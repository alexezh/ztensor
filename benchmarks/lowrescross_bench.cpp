// LowResCross at a policy's size: rows 426 wide (174 values, 44 meta, 208
// state in 13 regions of 16), 11 queries over 12 memory rows, 4 levels of
// 4 heads x 16, feed-forward 128 (about 4.4 MB of weights).
//
// Times each way a step runs, and how busy the CPU and the GPU were:
//   cpu%  this process's CPU time over the wall time (100% = one core)
//   gpu%  the GPU's "Device Utilization %" (ioreg), whole system, sampled
//         every 100 ms while the case ran; idle shows the baseline
//
//   lowrescross_bench [seconds per case = 3]

#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <random>
#include <string>
#include <thread>
#include <vector>

#include <sys/resource.h>

#include "loss.h"
#include "lowrescross.h"
#include "thread_pool.h"

namespace {

constexpr std::size_t kRegions = 13, kRegionHidden = 16;
constexpr std::size_t kValues = 174, kMeta = 44, kStateAt = kValues + kMeta;
constexpr std::size_t kTagAt = kValues + 28, kRelAt = kValues + 36;  // id slots in meta
constexpr std::size_t kQ = 11, kM = 12, kLevels = 4;

std::size_t region_at(std::size_t r) { return r * 13; }  // 13 values each, the last 18
std::size_t region_width(std::size_t r) { return r + 1 < kRegions ? 13 : kValues - 12 * 13; }

struct Row {
  std::array<bool, kRegions> active{};
  std::array<float, kValues> values{};
  std::array<float, kRegions * kRegionHidden> state{};
  int tag = 0;
  std::array<int, 3> rel{-1, -1, -1};
};

struct Layout {
  using Input = Row;
  using StateGrad = std::array<float, kRegions * kRegionHidden>;
  static constexpr std::size_t kWidth = kStateAt + kRegions * kRegionHidden;
  static constexpr std::size_t kStateOffset = kStateAt;
  static constexpr std::size_t kState = kRegions * kRegionHidden;
  static constexpr std::size_t kIds = 17;
  static constexpr std::size_t kIdDim = 8;
  static constexpr std::size_t kIdSlots = 4;
  static constexpr std::array<std::size_t, kIdSlots> kIdAt = {kTagAt, kRelAt, kRelAt, kRelAt};

  template <typename T>
  static void put(T* row, const Row& t) {
    for (std::size_t r = 0; r < kRegions; ++r) {
      if (!t.active[r]) continue;
      row[kValues + 15 + r] = T{1};  // active flag
      for (std::size_t k = 0; k < region_width(r); ++k) row[region_at(r) + k] = t.values[region_at(r) + k];
      for (std::size_t k = 0; k < kRegionHidden; ++k) row[kStateAt + r * kRegionHidden + k] = t.state[r * kRegionHidden + k];
    }
  }
  template <typename T>
  static void put_mask(T* mask, const Row& t) {
    for (std::size_t k = 0; k < kMeta; ++k) mask[kValues + k] = T{1};
    for (std::size_t r = 0; r < kRegions; ++r) {
      if (!t.active[r]) continue;
      for (std::size_t k = 0; k < region_width(r); ++k) mask[region_at(r) + k] = T{1};
      for (std::size_t k = 0; k < kRegionHidden; ++k) mask[kStateAt + r * kRegionHidden + k] = T{1};
    }
  }
  template <typename T>
  static void put_write_mask(T* mask, const Row& t) {
    for (std::size_t r = 0; r < kRegions; ++r) {
      if (!t.active[r]) continue;
      for (std::size_t k = 0; k < kRegionHidden; ++k) mask[r * kRegionHidden + k] = T{1};
    }
  }
  static int id(const Row& t, std::size_t slot) { return slot == 0 ? t.tag : t.rel[slot - 1]; }
  template <typename T>
  static StateGrad state_grad(const T* d, const Row&) {
    StateGrad g{};
    for (std::size_t k = 0; k < kState; ++k) g[k] = static_cast<float>(d[k]);
    return g;
  }
};

using Cross = zf::LowResCross<float, Layout, kQ, kM, 4, 16, 128>;

Row random_row(std::mt19937& rng) {
  std::uniform_real_distribution<float> u(-1.0f, 1.0f);
  Row t;
  for (std::size_t r = 0; r < kRegions; ++r) t.active[r] = rng() % 13 < 5;  // about 5 of 13
  for (float& v : t.values) v = u(rng);
  for (float& v : t.state) v = 0.5f * u(rng);
  t.tag = static_cast<int>(rng() % 17);
  if (rng() % 2) t.rel[0] = static_cast<int>(1 + rng() % 16);
  return t;
}

// What a policy puts on top: a 16-way choice from readout 0, a pointer from
// readout 1 into the memory rows.
constexpr std::size_t kChoices = 16;

struct Heads {
  using Node = zf::apple::Node;
  zf::Linear<float, Layout::kState, kChoices> choice;
  zf::LowResPointer<float, Layout, kM, 16> pointer;
  std::vector<std::size_t> picked, target;

  template <typename Visit>
  void visit_layers(Visit&& visit) {
    visit(choice);
    visit(pointer);
  }
#if defined(ZTENSOR_APPLE_GPU)
  struct Nodes {
    Node live, hot_c, hot_t, logits;
  };
  Node build(zf::LowResHeadsGraph& hg, Nodes* n) {
    zf::apple::Graph& g = hg.g;
    const int b = hg.batch, h = static_cast<int>(Layout::kState);
    const Node cw = hg.params.next(), cb = hg.params.next();
    const Node aw = hg.params.next(), ab = hg.params.next(), bw = hg.params.next(), bb = hg.params.next();
    const Node h3 = g.reshape(hg.hidden, {b, hg.q, h});
    const Node r0 = g.reshape(g.slice(h3, 1, 0, 1), {b, h}), r1 = g.reshape(g.slice(h3, 1, 1, 1), {b, h});
    n->live = hg.in.add(g, {b, 1});
    n->hot_c = hg.in.add(g, {b, static_cast<int>(kChoices)});
    n->hot_t = hg.in.add(g, {b, hg.m});
    n->logits = zf::apple::linear(g, r0, cw, cb);
    hg.out.add(g, n->logits);
    const Node ce_c = g.sum_all(g.mul(g.reshape(n->live, {b}), g.softmax_ce(n->logits, n->hot_c)));
    const Node qa = g.reshape(zf::apple::linear(g, r1, aw, ab), {b, 1, 16});
    const Node keys = g.reshape(zf::apple::linear(g, hg.rows, bw, bb), {b, hg.m, 16});
    const Node logits = g.reshape(g.matmul(qa, g.transpose(keys, 1, 2)), {b, hg.m});
    const Node ce_t = g.sum_all(g.mul(g.reshape(n->live, {b}), g.softmax_ce(logits, n->hot_t)));
    return g.add(ce_c, ce_t);
  }
  void fill(const Nodes& n, zf::LowResGraphInputs& in) const {
    for (std::size_t b = 0; b < picked.size(); ++b) {
      in[n.live].data()[b] = 1.0f;
      in[n.hot_c].data()[b * kChoices + picked[b]] = 1.0f;
      in[n.hot_t].data()[b * kM + target[b]] = 1.0f;
    }
  }
  void read(const Nodes&, const zf::LowResGraphOutputs&) {}
#endif
};

double now() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }

double cpu_seconds() {
  rusage u{};
  getrusage(RUSAGE_SELF, &u);
  return u.ru_utime.tv_sec + u.ru_stime.tv_sec + 1e-6 * (u.ru_utime.tv_usec + u.ru_stime.tv_usec);
}

int gpu_utilization() {
  FILE* p = popen("ioreg -r -c AGXAccelerator -d 1 2>/dev/null | grep -o '\"Device Utilization %\"=[0-9]*'", "r");
  if (p == nullptr) return -1;
  char line[128] = {};
  const bool got = std::fgets(line, sizeof line, p) != nullptr;
  pclose(p);
  if (!got) return -1;
  const char* eq = std::strrchr(line, '=');
  return eq != nullptr ? std::atoi(eq + 1) : -1;
}

// Runs f for `seconds` after a warm-up, sampling the GPU alongside.
void measure(const char* name, std::size_t examples, double seconds, const std::function<void()>& f) {
  for (int w = 0; w < 3; ++w) f();
  std::atomic<bool> stop{false};
  std::vector<int> samples;
  std::thread sampler([&] {
    while (!stop) {
      const int u = gpu_utilization();
      if (u >= 0) samples.push_back(u);
      std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
  });
  const double c0 = cpu_seconds(), t0 = now();
  int n = 0;
  while (now() - t0 < seconds) {
    f();
    ++n;
  }
  const double wall = now() - t0, cpu = cpu_seconds() - c0;
  stop = true;
  sampler.join();
  double gpu = 0.0;
  for (int s : samples) gpu += s;
  gpu = samples.empty() ? -1.0 : gpu / samples.size();
  std::printf("%-34s %8.2f ms %9.0f ex/s %6.0f cpu%% %5.0f gpu%%\n", name, 1e3 * wall / n,
              examples * n / wall, 100.0 * cpu / wall, gpu);
}

}  // namespace

int main(int argc, char** argv) {
  const double seconds = argc > 1 ? std::atof(argv[1]) : 3.0;
  std::mt19937 rng(1);
  Cross cross(kLevels);
  cross.randomize(rng);
  std::size_t weights = 0;
  zf::visit_params(cross, [&](auto& t) { weights += t.size(); });
  Heads heads;
  heads.choice.randomize(rng);
  heads.pointer.randomize(rng);

  const std::size_t most = 256;
  std::vector<std::vector<Row>> queries(most), memory(most);
  for (std::size_t b = 0; b < most; ++b) {
    for (std::size_t i = 0; i < kQ; ++i) queries[b].push_back(random_row(rng));
    for (std::size_t j = 0; j < kM; ++j) memory[b].push_back(random_row(rng));
    heads.picked.push_back(rng() % kChoices);
    heads.target.push_back(rng() % kM);
  }
  const auto passes = [&](std::size_t n) {
    std::vector<zf::LowResPass<Row>> out;
    for (std::size_t b = 0; b < n; ++b) out.push_back({&queries[b], &memory[b]});
    return out;
  };
  std::vector<Cross::Hidden> d(most);
  for (auto& h : d) {
    for (std::size_t i = 0; i < h.size(); ++i) h.data()[i] = 0.01f * std::sin(static_cast<float>(i));
  }
  const std::vector<Cross::Hidden> d0;

  zf::ThreadPool& pool = zf::ThreadPool::global();
  std::printf("LowResCross %zu levels, %zu weights (%.1f MB), rows %zu (state %zu), Q %zu, M %zu; pool %zu\n",
              kLevels, weights, 4.0 * weights / 1e6, Layout::kWidth, Layout::kState, kQ, kM, pool.participants());
  std::printf("idle gpu%%: %d\n\n", gpu_utilization());
  std::printf("%-34s %11s %12s %10s %8s\n", "case", "per call", "", "", "");

  // The CPU: forward + backward (the heads not counted), one thread and the pool.
  cross.set_device(zf::LowResDevice::kCpu);
  for (std::size_t n : {16, 64}) {
    const auto batch = passes(n);
    std::vector<Cross::Hidden> dn(d.begin(), d.begin() + n);
    for (zf::ThreadPool* p : {static_cast<zf::ThreadPool*>(nullptr), &pool}) {
      cross.set_pool(p);
      const std::string name = "cpu fwd+bwd b" + std::to_string(n) + (p ? " pool" : " 1 thread");
      measure(name.c_str(), n, seconds, [&] {
        cross.forward_batch(batch);
        cross.backward_batch(dn);
      });
    }
  }
  cross.set_pool(nullptr);

#if defined(ZTENSOR_APPLE_GPU)
  cross.set_device(zf::LowResDevice::kApple);
  // Acting: forward only, the weights unchanged (uploaded once).
  for (std::size_t n : {64, 128, 256}) {
    const auto batch = passes(n);
    measure(("gpu forward (act) b" + std::to_string(n)).c_str(), n, seconds, [&] { cross.forward_batch(batch); });
  }
  // Training, two runs: forward, the heads (here a stand-in gradient), backward; then the step.
  for (std::size_t n : {64, 128, 256}) {
    const auto batch = passes(n);
    std::vector<Cross::Hidden> dn(d.begin(), d.begin() + n);
    measure(("gpu fwd+bwd+sgd b" + std::to_string(n)).c_str(), n, seconds, [&] {
      cross.zero_grad();
      cross.forward_batch(batch);
      cross.backward_batch(dn);
      cross.apply_sgd(1e-6f);
    });
  }
  // Training, one run: step_batch with the heads, then the step.
  for (std::size_t n : {64, 128, 256}) {
    const auto batch = passes(n);
    heads.picked.resize(n);
    heads.target.resize(n);
    measure(("gpu step_batch+sgd b" + std::to_string(n)).c_str(), n, seconds, [&] {
      cross.zero_grad();
      heads.visit_layers([](auto& l) { l.zero_grad(); });
      cross.step_batch(batch, heads);
      cross.apply_sgd(1e-6f);
      heads.visit_layers([](auto& l) { l.apply_sgd(1e-6f); });
    });
  }
#endif
  return 0;
}
