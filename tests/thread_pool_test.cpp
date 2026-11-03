#include "thread_pool.h"

#include <atomic>
#include <numeric>
#include <stdexcept>
#include <thread>
#include <vector>

#include <gtest/gtest.h>

#include <random>

#include "layers.h"

namespace {

// Every index of [0, n) is visited once, by the chunk that owns it.
void check_cover(zf::ThreadPool* pool, std::size_t n, std::size_t grain) {
  std::vector<std::atomic<int>> seen(n);
  std::vector<std::size_t> owner(n, ~std::size_t{0});
  zf::parallel_for(pool, n, grain, [&](std::size_t c, std::size_t b, std::size_t e) {
    for (std::size_t i = b; i < e; ++i) {
      seen[i].fetch_add(1);
      owner[i] = c;
    }
  });
  const std::size_t chunks = zf::chunk_count(n, grain);
  for (std::size_t i = 0; i < n; ++i) {
    ASSERT_EQ(seen[i].load(), 1) << i;
    ASSERT_GE(i, zf::chunk_begin(n, chunks, owner[i]));
    ASSERT_LT(i, zf::chunk_begin(n, chunks, owner[i] + 1));
  }
}

}  // namespace

TEST(ThreadPool, CoversEveryIndexOnce) {
  zf::ThreadPool pool(4);
  for (std::size_t n : {0, 1, 7, 64, 1000, 4097}) {
    for (std::size_t grain : {1, 3, 16, 5000}) check_cover(&pool, n, grain);
  }
  check_cover(nullptr, 1000, 7);
}

// Chunks depend on n and the grain only: a per-chunk sum added in chunk
// order is the same bits on any pool.
TEST(ThreadPool, ChunkSumsAreTheSameOnAnyPool) {
  std::vector<float> x(10007);
  for (std::size_t i = 0; i < x.size(); ++i) x[i] = 1.0f / static_cast<float>(i + 1);
  const auto sum = [&](zf::ThreadPool* pool) {
    std::vector<float> part(zf::chunk_count(x.size(), 100), 0.0f);
    zf::parallel_for(pool, x.size(), 100, [&](std::size_t c, std::size_t b, std::size_t e) {
      for (std::size_t i = b; i < e; ++i) part[c] += x[i];
    });
    float s = 0.0f;
    for (float p : part) s += p;
    return s;
  };
  const float one = sum(nullptr);
  for (std::size_t n : {1, 2, 3, 8, 13}) {
    zf::ThreadPool pool(n);
    for (int rep = 0; rep < 20; ++rep) EXPECT_EQ(sum(&pool), one) << n;
  }
}

TEST(ThreadPool, RethrowsAfterEveryChunkFinished) {
  zf::ThreadPool pool(4);
  std::atomic<int> ran{0};
  EXPECT_THROW(zf::parallel_for(&pool, 32, 1,
                                [&](std::size_t c, std::size_t, std::size_t) {
                                  ran.fetch_add(1);
                                  if (c == 5) throw std::runtime_error("chunk 5");
                                }),
               std::runtime_error);
  EXPECT_EQ(ran.load(), 32);
}

// A loop started inside a chunk runs inline on that worker.
TEST(ThreadPool, NestedLoopsRunInline) {
  zf::ThreadPool pool(4);
  std::atomic<long> total{0};
  zf::parallel_for(&pool, 16, 1, [&](std::size_t, std::size_t, std::size_t) {
    zf::parallel_for(&pool, 100, 10, [&](std::size_t, std::size_t b, std::size_t e) {
      total.fetch_add(static_cast<long>(e - b));
    });
  });
  EXPECT_EQ(total.load(), 1600);
}

// Several threads share one pool, each waiting only for its own items.
TEST(ThreadPool, ConcurrentSubmitters) {
  zf::ThreadPool pool(6);
  std::vector<std::thread> callers;
  std::atomic<int> failures{0};
  for (int t = 0; t < 4; ++t) {
    callers.emplace_back([&, t] {
      for (int rep = 0; rep < 300; ++rep) {
        std::vector<int> out(257, 0);
        zf::parallel_for(&pool, out.size(), 4, [&](std::size_t, std::size_t b, std::size_t e) {
          for (std::size_t i = b; i < e; ++i) out[i] = static_cast<int>(i) + t;
        });
        for (std::size_t i = 0; i < out.size(); ++i) {
          if (out[i] != static_cast<int>(i) + t) failures.fetch_add(1);
        }
      }
    });
  }
  for (std::thread& c : callers) c.join();
  EXPECT_EQ(failures.load(), 0);
}

// Items that find every queue full run on the caller; none is lost.
TEST(ThreadPool, OverflowRunsOnTheCaller) {
  zf::ThreadPool pool(2);  // one worker, one queue
  zf::Completion done;
  std::atomic<int> ran{0};
  constexpr std::size_t kN = 3 * zf::WorkQueue::kCapacity;
  std::vector<zf::WorkItem> items(kN);
  struct Ctx {
    std::atomic<int>* ran;
    static void run(void* c, std::size_t) { static_cast<Ctx*>(c)->ran->fetch_add(1); }
  } ctx{&ran};
  done.add(kN);
  for (zf::WorkItem& it : items) it = {&Ctx::run, &ctx, 0, &done};
  pool.submit(items.data(), kN);
  pool.wait(done);
  EXPECT_EQ(ran.load(), static_cast<int>(kN));
}

TEST(ThreadPool, GlobalPoolTakesMostOfTheMachine) {
  EXPECT_GE(zf::ThreadPool::global().participants(), 1u);
  EXPECT_LE(zf::ThreadPool::global().participants(),
            std::max<std::size_t>(1, std::thread::hardware_concurrency()));
}

// Many tiny loops back to back: the pattern of a training step.
TEST(ThreadPool, ManySmallLoops) {
  zf::ThreadPool pool(8);
  std::vector<double> a(512, 1.0);
  for (int rep = 0; rep < 20000; ++rep) {
    zf::parallel_for(&pool, a.size(), 32, [&](std::size_t, std::size_t b, std::size_t e) {
      for (std::size_t i = b; i < e; ++i) a[i] += 1.0;
    });
  }
  for (double v : a) ASSERT_EQ(v, 20001.0);
}

// Without the GEMM (double here) a large stack splits over the global pool;
// it must still be the rows computed one at a time.
TEST(ThreadPool, SplitKernelsMatchOneRowAtATime) {
  constexpr std::size_t kIn = 256, kOut = 132, kRows = 300;  // leftover outputs
  static_assert(kRows * kIn * kOut >= zf::blas::kSplitWork);
  std::mt19937 rng(3);
  zf::Linear<double, kIn, kOut> split;
  split.randomize(rng);
  zf::Linear<double, kIn, kOut> single = split;
  std::normal_distribution<double> nd;
  std::vector<double> x(kRows * kIn), g(kRows * kOut);
  for (double& v : x) v = nd(rng);
  for (double& v : g) v = nd(rng);
  std::vector<double> y(kRows * kOut), y1(kRows * kOut), dx(kRows * kIn, 0.0), dx1(kRows * kIn, 0.0);
  split.forward_into(x.data(), y.data(), kRows);
  split.backward_into(x.data(), g.data(), dx.data(), kRows);
  for (std::size_t r = 0; r < kRows; ++r) {
    single.forward_into(x.data() + r * kIn, y1.data() + r * kOut, 1);
    single.backward_into(x.data() + r * kIn, g.data() + r * kOut, dx1.data() + r * kIn, 1);
  }
  for (std::size_t i = 0; i < y.size(); ++i) ASSERT_EQ(y[i], y1[i]) << i;
  for (std::size_t i = 0; i < dx.size(); ++i) ASSERT_EQ(dx[i], dx1[i]) << i;
  for (std::size_t i = 0; i < split.weight_grad().size(); ++i) {
    ASSERT_NEAR(split.weight_grad().data()[i], single.weight_grad().data()[i], 1e-9) << i;
  }
  for (std::size_t o = 0; o < kOut; ++o) ASSERT_NEAR(split.bias_grad()(o), single.bias_grad()(o), 1e-9);
}
