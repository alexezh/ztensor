#include "thread_pool.h"

#include <chrono>
#include <cstdlib>
#include <string>

#if defined(__APPLE__)
#include <pthread.h>
#include <pthread/qos.h>
#include <sys/sysctl.h>
#elif defined(__linux__)
#include <pthread.h>
#include <sched.h>
#elif defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#endif

#if defined(__aarch64__) || defined(_M_ARM64)
#if defined(_MSC_VER)
#include <intrin.h>
#endif
#elif defined(__x86_64__) || defined(_M_X64) || defined(__i386__)
#include <immintrin.h>
#endif

namespace zf {

namespace {

// One spin-wait step: tells the core it is spinning, so a sibling thread or
// the power manager gets the slack.
inline void cpu_relax() {
#if defined(__aarch64__)
  __asm__ __volatile__("yield");
#elif defined(_M_ARM64)
  __yield();
#elif defined(__x86_64__) || defined(_M_X64) || defined(__i386__)
  _mm_pause();
#endif
}

// How long an idle worker spins before it sleeps. Waking a sleeping thread
// costs several microseconds -- as much as a whole small loop -- and a batch
// step runs dozens of loops back to back.
const std::chrono::microseconds kSpin = [] {
  const char* env = std::getenv("ZF_SPIN_US");
  return std::chrono::microseconds(env != nullptr ? std::strtol(env, nullptr, 10) : 200);
}();

thread_local bool t_on_worker = false;

#if defined(__APPLE__)
int sysctl_int(const char* name) {
  int v = 0;
  std::size_t size = sizeof(v);
  return sysctlbyname(name, &v, &size, nullptr, 0) == 0 ? v : 0;
}
#endif

std::size_t machine_participants() {
#if defined(__APPLE__)
  // The performance cores, less one for the GEMM's helper thread (blas.h)
  // and everything else. The efficiency cores sit out: a chunk that lands
  // on one finishes last, and the whole loop waits for it. On an M2 Max (8
  // performance, 4 efficiency) a 4-level cross transformer took 9.1-9.3 s
  // an epoch with 5-7 threads, 10.9 s with 8 and 11.0 s with 11.
  const int p = sysctl_int("hw.perflevel0.physicalcpu");
  if (p > 0) return p > 1 ? static_cast<std::size_t>(p - 1) : 1;
  const int cpus_apple = sysctl_int("hw.activecpu");
  return cpus_apple > 1 ? static_cast<std::size_t>(cpus_apple - 1) : 1;
#else
  std::size_t cpus = 0;
#if defined(__linux__)
  cpu_set_t set;
  CPU_ZERO(&set);
  if (sched_getaffinity(0, sizeof(set), &set) == 0) cpus = static_cast<std::size_t>(CPU_COUNT(&set));
#elif defined(_WIN32)
  cpus = static_cast<std::size_t>(GetActiveProcessorCount(ALL_PROCESSOR_GROUPS));
#endif
  if (cpus == 0) cpus = std::thread::hardware_concurrency();
  // Leave one for the system and everything else the process does.
  return cpus > 1 ? cpus - 1 : 1;
#endif
}

void configure_worker_thread(std::size_t id) {
#if defined(__APPLE__)
  // User-interactive: what the scheduler keeps on performance cores. The
  // default class lets a worker drift onto an efficiency core, where its
  // chunk finishes last and the whole loop waits for it.
  pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0);
  pthread_setname_np(("zf-worker-" + std::to_string(id)).c_str());
#elif defined(__linux__)
  pthread_setname_np(pthread_self(), ("zf-worker-" + std::to_string(id)).substr(0, 15).c_str());
#else
  (void)id;
#endif
}

}  // namespace

// --- WorkQueue --------------------------------------------------------------

WorkQueue::WorkQueue() {
  for (std::size_t i = 0; i < kCapacity; ++i) slots_[i].seq.store(i, std::memory_order_relaxed);
}

bool WorkQueue::push(const WorkItem& item) {
  std::size_t pos = tail_.load(std::memory_order_relaxed);
  for (;;) {
    Slot& s = slots_[pos & kMask];
    const std::size_t seq = s.seq.load(std::memory_order_acquire);
    const auto diff = static_cast<std::ptrdiff_t>(seq) - static_cast<std::ptrdiff_t>(pos);
    if (diff == 0) {
      // The slot is free for this position: claim it.
      if (tail_.compare_exchange_weak(pos, pos + 1, std::memory_order_relaxed)) {
        s.item = item;
        s.seq.store(pos + 1, std::memory_order_release);
        return true;
      }
    } else if (diff < 0) {
      return false;  // full: the slot still holds an item a lap behind
    } else {
      pos = tail_.load(std::memory_order_relaxed);
    }
  }
}

bool WorkQueue::pop(WorkItem* item) {
  std::size_t pos = head_.load(std::memory_order_relaxed);
  for (;;) {
    Slot& s = slots_[pos & kMask];
    const std::size_t seq = s.seq.load(std::memory_order_acquire);
    const auto diff = static_cast<std::ptrdiff_t>(seq) - static_cast<std::ptrdiff_t>(pos + 1);
    if (diff == 0) {
      if (head_.compare_exchange_weak(pos, pos + 1, std::memory_order_relaxed)) {
        *item = s.item;
        s.seq.store(pos + kCapacity, std::memory_order_release);
        return true;
      }
    } else if (diff < 0) {
      return false;  // empty
    } else {
      pos = head_.load(std::memory_order_relaxed);
    }
  }
}

bool WorkQueue::empty() const {
  const std::size_t pos = head_.load(std::memory_order_relaxed);
  return slots_[pos & kMask].seq.load(std::memory_order_acquire) != pos + 1;
}

// --- ThreadPool -------------------------------------------------------------

struct ThreadPool::Worker {
  WorkQueue queue;
};

ThreadPool::ThreadPool(std::size_t participants) {
  if (participants == 0) participants = 1;
  const std::size_t n = participants - 1;
  workers_.reserve(n);
  for (std::size_t i = 0; i < n; ++i) workers_.push_back(std::make_unique<Worker>());
  threads_.reserve(n);
  for (std::size_t i = 0; i < n; ++i) threads_.emplace_back([this, i] { worker_main(i); });
}

ThreadPool::~ThreadPool() {
  stop_.store(true, std::memory_order_seq_cst);
  wake_.fetch_add(1, std::memory_order_seq_cst);
  wake_.notify_all();
  for (std::thread& t : threads_) t.join();
}

ThreadPool& ThreadPool::global() {
  static ThreadPool pool(default_participants());
  return pool;
}

std::size_t ThreadPool::default_participants() {
  if (const char* env = std::getenv("ZF_THREADS")) {
    const long v = std::strtol(env, nullptr, 10);
    if (v > 0) return static_cast<std::size_t>(v);
  }
  return machine_participants();
}

bool ThreadPool::on_worker() { return t_on_worker; }

void ThreadPool::wake() {
  // Pairs with the sleeper's seq_cst increment of sleepers_ and recheck of
  // the queues: either it sees this item, or this sees it asleep.
  if (sleepers_.load(std::memory_order_seq_cst) != 0) {
    wake_.fetch_add(1, std::memory_order_seq_cst);
    wake_.notify_all();
  }
}

void ThreadPool::submit(const WorkItem* items, std::size_t n) {
  submit(items, n, cursor_.fetch_add(n, std::memory_order_relaxed));
}

void ThreadPool::submit(const WorkItem* items, std::size_t n, std::size_t first) {
  if (n == 0) return;
  if (workers_.empty()) {
    for (std::size_t i = 0; i < n; ++i) {
      items[i].run(items[i].ctx, items[i].index);
      items[i].done->finish();
    }
    return;
  }
  const std::size_t w = workers_.size();
  std::size_t at = first;
  for (std::size_t i = 0; i < n; ++i, ++at) {
    bool queued = false;
    for (std::size_t k = 0; k < w && !queued; ++k) queued = workers_[(at + k) % w]->queue.push(items[i]);
    if (!queued) {
      items[i].run(items[i].ctx, items[i].index);
      items[i].done->finish();
    }
  }
  std::atomic_thread_fence(std::memory_order_seq_cst);
  wake();
}

bool ThreadPool::run_one(std::size_t first) {
  const std::size_t w = workers_.size();
  WorkItem item;
  for (std::size_t k = 0; k < w; ++k) {
    if (workers_[(first + k) % w]->queue.pop(&item)) {
      item.run(item.ctx, item.index);
      item.done->finish();
      return true;
    }
  }
  return false;
}

bool ThreadPool::any_queued() const {
  for (const auto& worker : workers_) {
    if (!worker->queue.empty()) return true;
  }
  return false;
}

void ThreadPool::wait(Completion& done) {
  // Help: run whatever is queued (ours or anyone's) until ours is done.
  while (!done.done()) {
    if (run_one(0)) continue;
    // Nothing queued: the rest is running on workers. Spin a little, then
    // sleep until the last one finishes.
    const auto until = std::chrono::steady_clock::now() + kSpin;
    while (!done.done() && !any_queued()) {
      for (int i = 0; i < 64; ++i) cpu_relax();
      if (std::chrono::steady_clock::now() > until) {
        done.wait_idle();
        break;
      }
    }
  }
}

void ThreadPool::worker_main(std::size_t id) {
  t_on_worker = true;
  configure_worker_thread(id);
  while (!stop_.load(std::memory_order_acquire)) {
    if (run_one(id)) continue;
    // Spin, so a loop that follows right away finds us awake.
    const auto until = std::chrono::steady_clock::now() + kSpin;
    bool found = false;
    while (!found && std::chrono::steady_clock::now() < until) {
      for (int i = 0; i < 64; ++i) cpu_relax();
      found = any_queued() || stop_.load(std::memory_order_relaxed);
    }
    if (found) continue;
    // Sleep until a submit (or the destructor) bumps wake_.
    const std::uint32_t seen = wake_.load(std::memory_order_seq_cst);
    sleepers_.fetch_add(1, std::memory_order_seq_cst);
    if (!any_queued() && !stop_.load(std::memory_order_seq_cst)) wake_.wait(seen, std::memory_order_seq_cst);
    sleepers_.fetch_sub(1, std::memory_order_seq_cst);
  }
}

}  // namespace zf
