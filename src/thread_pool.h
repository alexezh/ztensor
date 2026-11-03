#pragma once

// A pool of worker threads for the library's data-parallel loops.
//
// Work is passed as WorkItems -- a function pointer, its context and an
// index -- through one queue per worker. A submitter spreads its items over
// the queues; a worker runs its own queue first and steals from the others
// when it is empty. The thread that submitted works too: Completion counts
// the items outstanding, and ThreadPool::wait() runs queued items until the
// count reaches zero, so a caller never idles while its own work waits.
//
// The loops this is for are short (tens of microseconds), back to back. A
// worker with nothing to do spins for a while before it sleeps, so the next
// loop finds it awake; asleep, it waits on an atomic (a futex on Linux,
// __ulock on Apple, a WaitOnAddress on Windows -- whatever the standard
// library uses for std::atomic::wait).
//
// parallel_for() splits [0, n) into chunks that depend only on n and the
// grain, never on the number of threads, and hands each chunk its index. A
// loop that keeps one partial sum per chunk and adds them in chunk order
// gets the same bits on one thread as on twelve.
//
// Size: ThreadPool::global() takes most of the machine (see
// default_participants()); ZF_THREADS in the environment overrides it, and
// ZF_THREADS=1 runs everything on the caller.

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <memory>
#include <mutex>
#include <thread>
#include <type_traits>
#include <vector>

namespace zf {

class Completion;

// One item of work: run(ctx, index).
struct WorkItem {
  void (*run)(void* ctx, std::size_t index) = nullptr;
  void* ctx = nullptr;
  std::size_t index = 0;
  Completion* done = nullptr;  // told when the item has run
};

// Items outstanding. A submitter adds before it submits; each item finishes
// once it has run.
class Completion {
 public:
  void add(std::size_t n) { pending_.fetch_add(n, std::memory_order_relaxed); }
  void finish() {
    if (pending_.fetch_sub(1, std::memory_order_acq_rel) == 1) pending_.notify_all();
  }
  bool done() const { return pending_.load(std::memory_order_acquire) == 0; }
  void wait_idle() const {
    for (std::size_t p = pending_.load(std::memory_order_acquire); p != 0;
         p = pending_.load(std::memory_order_acquire)) {
      pending_.wait(p, std::memory_order_acquire);
    }
  }

 private:
  std::atomic<std::size_t> pending_{0};
};

// A bounded queue of WorkItems, any number of producers and consumers, no
// locks (D. Vyukov's bounded MPMC queue): each slot carries a sequence
// number that says whose turn it is, so a push and a pop claim a slot with
// one compare-and-swap on the head or the tail.
class WorkQueue {
 public:
  static constexpr std::size_t kCapacity = 256;  // a power of two

  WorkQueue();
  bool push(const WorkItem& item);  // false when full
  bool pop(WorkItem* item);         // false when empty
  bool empty() const;

 private:
  struct Slot {
    std::atomic<std::size_t> seq;
    WorkItem item;
  };
  static constexpr std::size_t kMask = kCapacity - 1;
  alignas(64) std::atomic<std::size_t> tail_{0};  // next push
  alignas(64) std::atomic<std::size_t> head_{0};  // next pop
  alignas(64) Slot slots_[kCapacity];
};

class ThreadPool {
 public:
  // `participants` counts the calling thread: participants - 1 workers.
  explicit ThreadPool(std::size_t participants = default_participants());
  ~ThreadPool();
  ThreadPool(const ThreadPool&) = delete;
  ThreadPool& operator=(const ThreadPool&) = delete;

  // Threads that work on a loop: the workers and the caller.
  std::size_t participants() const { return workers_.size() + 1; }

  // The library's pool, made on first use.
  static ThreadPool& global();

  // Most of the machine, counting the caller:
  //  - Apple: the performance cores less one (sysctl
  //    hw.perflevel0.physicalcpu), leaving that one to the GEMM's helper
  //    thread; efficiency cores make stragglers in a fork-join loop. Workers
  //    ask for the user-interactive QoS class, so the scheduler keeps them on
  //    performance cores.
  //  - Linux: the CPUs this process may run on (sched_getaffinity), less one.
  //  - Windows: the active processors in every group, less one.
  //  - elsewhere: std::thread::hardware_concurrency(), less one.
  // ZF_THREADS overrides it.
  static std::size_t default_participants();

  // Queues `items` (each has its `done` set and counted) and returns: item i
  // to worker (first + i) % workers, or the next one with room. Items that
  // find every queue full run on the caller. A loop passes the same `first`
  // every time, so a chunk goes to the same worker, whose cache already
  // holds that chunk's rows; with no `first`, submits rotate.
  void submit(const WorkItem* items, std::size_t n);
  void submit(const WorkItem* items, std::size_t n, std::size_t first);

  // Runs queued items until `done` has none outstanding.
  void wait(Completion& done);

  // True on one of this pool's workers (or any pool's): a loop started there
  // runs inline rather than queueing behind itself.
  static bool on_worker();

 private:
  struct Worker;
  bool run_one(std::size_t first);  // runs one queued item, if any
  bool any_queued() const;
  void worker_main(std::size_t id);
  void wake();

  std::vector<std::unique_ptr<Worker>> workers_;
  std::vector<std::thread> threads_;
  std::atomic<std::size_t> cursor_{0};  // where the next submit starts
  alignas(64) std::atomic<std::uint32_t> wake_{0};
  std::atomic<std::uint32_t> sleepers_{0};
  std::atomic<bool> stop_{false};
};

// --- parallel_for ---------------------------------------------------------

// Chunks never exceed this many, whatever the machine.
inline constexpr std::size_t kMaxChunks = 64;

// The chunks [0, n) splits into at `grain` items per chunk (at least): a
// function of n and grain only.
inline std::size_t chunk_count(std::size_t n, std::size_t grain) {
  if (n == 0) return 0;
  if (grain == 0) grain = 1;
  const std::size_t c = (n + grain - 1) / grain;
  return c < kMaxChunks ? c : kMaxChunks;
}
inline std::size_t chunk_begin(std::size_t n, std::size_t chunks, std::size_t c) {
  return n * c / chunks;
}

namespace detail {

template <typename F>
struct ForContext {
  F* f;
  std::size_t n, chunks;
  std::once_flag error_once;
  std::exception_ptr error;

  void run_chunk(std::size_t c) {
    try {
      (*f)(c, chunk_begin(n, chunks, c), chunk_begin(n, chunks, c + 1));
    } catch (...) {
      std::call_once(error_once, [&] { error = std::current_exception(); });
    }
  }
  static void run(void* ctx, std::size_t c) { static_cast<ForContext*>(ctx)->run_chunk(c); }
};

}  // namespace detail

// f(chunk, begin, end) for every chunk of [0, n), on `pool`'s threads and
// the caller's; returns when all have run. f must write disjoint data per
// chunk. With no pool, one chunk, or called from a worker, the chunks run in
// order on the caller -- the same chunks either way. The first exception a
// chunk throws is rethrown here, after every chunk has finished.
template <typename F>
void parallel_for(ThreadPool* pool, std::size_t n, std::size_t grain, F&& f) {
  const std::size_t chunks = chunk_count(n, grain);
  if (chunks == 0) return;
  if (chunks == 1 || pool == nullptr || pool->participants() == 1 || ThreadPool::on_worker()) {
    for (std::size_t c = 0; c < chunks; ++c) f(c, chunk_begin(n, chunks, c), chunk_begin(n, chunks, c + 1));
    return;
  }
  using Fn = std::remove_reference_t<F>;
  detail::ForContext<Fn> ctx{&f, n, chunks, {}, nullptr};
  Completion done;
  WorkItem items[kMaxChunks];
  done.add(chunks - 1);
  for (std::size_t c = 1; c < chunks; ++c) items[c - 1] = {&detail::ForContext<Fn>::run, &ctx, c, &done};
  pool->submit(items, chunks - 1, 0);
  ctx.run_chunk(0);
  pool->wait(done);
  if (ctx.error) std::rethrow_exception(ctx.error);
}

}  // namespace zf
