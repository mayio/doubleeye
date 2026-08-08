// The packed top-2 warp reduction in de_dense_cuda.cu, tested as an algorithm.
//
// The GPU's fused vertical-backward/top-2 pass reduces 64 candidates to the
// best two with a shfl_down tree, and the tree merges NON-ADJACENT k ranges --
// lane 0 combines lane 16 before lane 1 -- so the merge cannot assume anything
// about which side holds smaller k. The first version did, and produced 10
// wrong pixels in 407k on real data, every one an exact tie. This test is the
// simulation that found it, kept: it mirrors the kernel's packed
// (value + 32768) << 8 | (255 - k) representation and the exact shuffle-tree
// order, and compares against the sequential ascending-k strictly-greater scan
// that defines the semantics. Pure host C++, so it runs on both machines.
#include <cstdint>
#include <cstdio>
#include <random>
#include <string>

namespace {
int g_failures = 0;
void check(bool ok, const std::string& what) {
  std::printf("  [%s] %s\n", ok ? "PASS" : "FAIL", what.c_str());
  if (!ok) ++g_failures;
}

int32_t pack(int16_t v, int k) { return ((int32_t(v) + 32768) << 8) | (255 - k); }

struct P { int32_t pk0 = 0, pk1 = 0; };
P merge(P m, P o) {
  if (o.pk0 > m.pk0) { m.pk1 = m.pk0 > o.pk1 ? m.pk0 : o.pk1; m.pk0 = o.pk0; }
  else if (o.pk0 > m.pk1) { m.pk1 = o.pk0; }
  return m;
}

bool run_trials(std::mt19937& rng, int trials, int spread, long* checked) {
  for (int t = 0; t < trials; ++t) {
    int16_t v[256];
    const int G = 1 + int(rng() % 4);          // 64..256 padded k, D up to 220
    const int D = std::min(220, G * 64 - int(rng() % 5));
    for (int k = 0; k < 256; ++k) v[k] = int16_t(rng() % spread);
    const int kmax = int(rng() % (D + 1)) - 1;
    int16_t b0 = -32768, b1 = -32768;
    int i0 = -1, i1 = -1;
    for (int k = 0; k <= kmax && k < D; ++k) {   // the defining semantics
      if (v[k] <= b1) continue;
      if (v[k] > b0) { b1 = b0; i1 = i0; b0 = v[k]; i0 = k; }
      else           { b1 = v[k]; i1 = k; }
    }
    P lanes[32];
    for (int l = 0; l < 32; ++l) {               // lane partials, g-loop like the kernel
      P p;
      for (int g = 0; g < G; ++g)
        for (int j = 0; j < 2; ++j) {
          const int k = g * 64 + l * 2 + j;
          if (k > kmax || k >= D) continue;
          const int32_t c = pack(v[k], k);
          if (c > p.pk0)      { p.pk1 = p.pk0; p.pk0 = c; }
          else if (c > p.pk1) { p.pk1 = c; }
        }
      lanes[l] = p;
    }
    for (int off = 16; off > 0; off >>= 1) {     // the exact shfl_down tree
      P nl[32];
      for (int l = 0; l < 32; ++l)
        nl[l] = merge(lanes[l], l + off < 32 ? lanes[l + off] : lanes[l]);
      for (int l = 0; l < 32; ++l) lanes[l] = nl[l];
    }
    const P r = lanes[0];
    const int16_t rb0 = int16_t((r.pk0 >> 8) - 32768);
    const int ri0 = r.pk0 ? 255 - (r.pk0 & 255) : -1;
    const int16_t rb1 = int16_t((r.pk1 >> 8) - 32768);
    const int ri1 = r.pk1 ? 255 - (r.pk1 & 255) : -1;
    ++*checked;
    if (rb0 != b0 || ri0 != i0 || rb1 != b1 || ri1 != i1) return false;
  }
  return true;
}
}  // namespace

int main() {
  std::printf("packed top-2 warp-reduction semantics\n\n");
  std::mt19937 rng(20260809);
  long n = 0;
  check(run_trials(rng, 200000, 3, &n), "tie-saturated (3 values over 256 k)");
  check(run_trials(rng, 200000, 12, &n), "tie-heavy (12 values)");
  check(run_trials(rng, 100000, 30000, &n), "spread scores");
  std::printf("\n%s (%ld trials)\n", g_failures ? "FAILED" : "ALL PASSED", n);
  return g_failures ? 1 : 0;
}
