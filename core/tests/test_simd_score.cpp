// Tests for the NEON score kernel in doubleeye/simd_score.hpp.
//
// Everything here is a PERMUTATION argument -- which lane holds which disparity,
// which register becomes which plane -- and that is the class of mistake that
// produces plausible-looking output rather than an error. The addp lane order and
// the transpose permutation were each wrong once during development, and the
// symptom both times was a handful of differing pixels out of 168,750, which is
// indistinguishable at a glance from the tie-order noise the matcher produces
// anyway (see doc/09-matching.md).
//
// So each primitive is checked against a scalar reference, and the composed kernel
// against an independent scalar implementation of the whole group -- the check that
// catches a correct primitive wired to the wrong plane.
//
// On anything that is not AArch64 the kernel does not exist and this reports SKIPPED
// with zero assertions rather than passing vacuously. A test that cannot fail on the
// machine you are running it on should say so out loud.

#include "doubleeye/simd_score.hpp"

#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <vector>

using namespace doubleeye;

namespace {

int g_failures = 0;
int g_checks = 0;

void check(bool ok, const std::string& what) {
  std::printf("  [%s] %s\n", ok ? "PASS" : "FAIL", what.c_str());
  ++g_checks;
  if (!ok) ++g_failures;
}

#ifdef DE_HAVE_NEON

// The Census score table de_dense builds: 49 Hamming levels mapped to Q14.
void build_tbl(int16_t* t64, int half) {
  std::memset(t64, 0, 64 * sizeof(int16_t));
  const int SCORE_ONE = 1 << 14;
  for (int h = 0; h <= 2 * half; ++h)
    t64[h] = int16_t(((half - h) * SCORE_ONE) / half);
}

void build_adt(int16_t* a256, int T) {
  const int SCORE_ONE = 1 << 14;
  for (int v = 0; v < 256; ++v)
    a256[v] = int16_t(SCORE_ONE - (2 * SCORE_ONE * std::min(v, T)) / T);
}

// ---------------------------------------------------------------------------
void test_lut8() {
  int16_t tbl[64];
  build_tbl(tbl, 24);
  // Entries 49..63 are unreachable in the kernel but must still round-trip, since
  // a wrong split would corrupt the reachable ones the same way.
  for (int i = 49; i < 64; ++i) tbl[i] = int16_t(-30000 + 4000 * (i - 49));
  const SplitTable t = simd_split_table(tbl);
  bool ok = true;
  for (int base = 0; base + 8 <= 64; ++base) {
    uint8_t idxb[16];
    for (int l = 0; l < 16; ++l) idxb[l] = uint8_t(base + (l & 7));
    int16_t got[8];
    vst1q_s16(got, simd_lut8(t, vld1q_u8(idxb)));
    for (int l = 0; l < 8; ++l) ok = ok && got[l] == tbl[base + l];
  }
  check(ok, "simd_lut8 returns tbl[idx] for every index and every alignment");

  // Negative entries are the interesting half: the low/high byte split has to
  // reassemble two's complement, and half the Census table is negative.
  bool neg = true;
  for (int i = 0; i < 64; ++i) {
    uint8_t idxb[16];
    for (int l = 0; l < 16; ++l) idxb[l] = uint8_t(i);
    int16_t got[8];
    vst1q_s16(got, simd_lut8(t, vld1q_u8(idxb)));
    if (tbl[i] < 0) neg = neg && got[0] == tbl[i];
  }
  check(neg, "simd_lut8 reassembles negative int16 entries correctly");
}

// ---------------------------------------------------------------------------
void test_hamming8() {
  std::mt19937_64 rng(20260808);
  const uint64_t mask = 0x0000FFFFFFFFFFFFull;   // 48-bit descriptor
  bool ok = true;
  for (int trial = 0; trial < 2000; ++trial) {
    const uint64_t left = rng() & mask;
    uint64_t right[8];
    for (int j = 0; j < 8; ++j) right[j] = rng() & mask;
    uint8_t got[16];
    vst1q_u8(got, simd_hamming8(right, left));
    for (int j = 0; j < 8; ++j)
      ok = ok && got[j] == __builtin_popcountll(left ^ right[j]);
  }
  check(ok, "simd_hamming8 matches popcount, in rp[] address order, over 2000 trials");

  // The reduction sums eight bytes into one u8. 48 bits set is the worst case and
  // must not wrap; if the descriptor ever widens past 8 bytes this is the check
  // that fails first.
  uint64_t worst[8];
  for (int j = 0; j < 8; ++j) worst[j] = mask;
  uint8_t got[16];
  vst1q_u8(got, simd_hamming8(worst, 0));
  bool no_wrap = true;
  for (int j = 0; j < 8; ++j) no_wrap = no_wrap && got[j] == 48;
  check(no_wrap, "simd_hamming8 does not wrap u8 at the maximum 48 bits differing");
}

// ---------------------------------------------------------------------------
void test_transpose() {
  int16x8_t in[8], out[8];
  int16_t buf[8];
  for (int p = 0; p < 8; ++p) {
    for (int j = 0; j < 8; ++j) buf[j] = int16_t(100 * p + j);
    in[p] = vld1q_s16(buf);
  }
  simd_transpose8x8(in, out);
  bool ok = true;
  for (int j = 0; j < 8; ++j) {
    vst1q_s16(buf, out[j]);
    for (int p = 0; p < 8; ++p) ok = ok && buf[p] == int16_t(100 * p + j);
  }
  check(ok, "simd_transpose8x8 gives out[j] lane p == in[p] lane j");

  // Negative and sign-boundary values, in case a reinterpret ever goes unsigned.
  for (int p = 0; p < 8; ++p) {
    for (int j = 0; j < 8; ++j)
      buf[j] = int16_t((p + j) % 2 ? -32768 + 137 * (8 * p + j) : 32767 - 91 * (8 * p + j));
    in[p] = vld1q_s16(buf);
  }
  simd_transpose8x8(in, out);
  bool ok2 = true;
  for (int j = 0; j < 8; ++j) {
    int16_t got[8];
    vst1q_s16(got, out[j]);
    for (int p = 0; p < 8; ++p) {
      const int16_t want = int16_t((p + j) % 2 ? -32768 + 137 * (8 * p + j)
                                               : 32767 - 91 * (8 * p + j));
      ok2 = ok2 && got[p] == want;
    }
  }
  check(ok2, "simd_transpose8x8 is exact at the int16 sign boundaries");
}

// ---------------------------------------------------------------------------
void test_blend8() {
  bool ok = true;
  int n = 0;
  const int SCORE_ONE = 1 << 14;
  for (int wq = 0; wq <= 1024; wq += 61)
    for (int ci = -SCORE_ONE; ci <= SCORE_ONE; ci += 811)
      for (int ai = -SCORE_ONE; ai <= SCORE_ONE; ai += 743) {
        int16_t cs[8], as[8];
        for (int l = 0; l < 8; ++l) {
          cs[l] = int16_t(ci + l);         // vary per lane so a lane mix-up shows
          as[l] = int16_t(ai - l);
        }
        int16_t got[8];
        vst1q_s16(got, simd_blend8(vld1q_s16(cs), vld1q_s16(as),
                                   int16_t(1024 - wq), int16_t(wq)));
        for (int l = 0; l < 8; ++l) {
          const int16_t want =
              int16_t((int32_t(cs[l]) * (1024 - wq) + int32_t(as[l]) * wq) >> 10);
          ok = ok && got[l] == want;
        }
        ++n;
      }
  check(ok, "simd_blend8 is bit-identical to the scalar int32 expression (" +
                std::to_string(n) + " lane sets, includes negative operands)");
}

// ---------------------------------------------------------------------------
// An independent scalar implementation of the whole group. Deliberately written as
// plainly as possible: plane-major outer, one pixel-disparity at a time.
void score_group_ref(const uint64_t* cl, const uint64_t* cr,
                     const uint8_t* Ld, const uint8_t* Rd,
                     int16_t* planes, size_t WH, int W, int H, int d0,
                     const int16_t* tbl, const int16_t* adt, int32_t wq) {
  for (int g = 0; g < SIMD_G; ++g)
    for (int y = 3; y < H - 3; ++y)
      for (int x = 3 + d0 + g; x < W - 3; ++x) {
        const size_t i = size_t(y) * W + x;
        const int32_t c = tbl[__builtin_popcountll(cl[i] ^ cr[i - size_t(d0 + g)])];
        const int32_t a = adt[std::abs(int(Ld[i]) - int(Rd[i - size_t(d0 + g)]))];
        planes[size_t(g) * WH + i] = int16_t((c * (1024 - wq) + a * wq) >> 10);
      }
}

void test_kernel_against_reference() {
  struct Case { int W, H, d0, T; float ad; const char* what; };
  const Case cases[] = {
    {61, 23,  1, 10, 0.15f, "W=61 H=23 d0=1, the shipping cost weights"},
    {64, 20,  1, 10, 0.15f, "W=64, x range a multiple of the vector width"},
    {59, 19, 20, 10, 0.15f, "d0=20, a wide left strip the prologue must cover"},
    {61, 23,  1, 10, 0.00f, "ad=0, census only -- wq=0 must reduce to tbl[h]"},
    {61, 23,  1, 10, 1.00f, "ad=1, absolute difference only"},
    {61, 23,  1,  1, 0.15f, "ad_trunc=1, the narrowest legal clamp"},
    {61, 23,  5, 63, 0.15f, "ad_trunc=63, the widest legal table index"},
    {23, 17,  1, 10, 0.15f, "W=23, barely wide enough for one vector step"},
  };
  std::mt19937_64 rng(987654321);
  const uint64_t mask = 0x0000FFFFFFFFFFFFull;
  const int16_t SENTINEL = 12345;

  for (const Case& c : cases) {
    const size_t WH = size_t(c.W) * c.H;
    std::vector<uint64_t> cl(WH), cr(WH);
    std::vector<uint8_t> Ld(WH), Rd(WH);
    for (size_t i = 0; i < WH; ++i) {
      cl[i] = rng() & mask;
      cr[i] = rng() & mask;
      Ld[i] = uint8_t(rng());
      Rd[i] = uint8_t(rng());
    }
    int16_t tbl[64];
    build_tbl(tbl, 24);
    int16_t adt[256];
    build_adt(adt, c.T);
    const int32_t wq = int32_t(std::max(0.f, std::min(1.f, c.ad)) * 1024.f);

    std::vector<int16_t> got(WH * SIMD_G, SENTINEL), want(WH * SIMD_G, SENTINEL);
    score_group_neon(cl.data(), cr.data(), Ld.data(), Rd.data(), got.data(), WH,
                     c.W, c.H, c.d0, tbl, adt, wq, c.T);
    score_group_ref(cl.data(), cr.data(), Ld.data(), Rd.data(), want.data(), WH,
                    c.W, c.H, c.d0, tbl, adt, wq);

    size_t diff = 0, first = 0;
    for (size_t i = 0; i < got.size(); ++i)
      if (got[i] != want[i]) { if (!diff) first = i; ++diff; }
    if (diff)
      std::printf("       first mismatch at plane %zu, y %zu, x %zu: %d vs %d\n",
                  first / WH, (first % WH) / size_t(c.W), (first % WH) % size_t(c.W),
                  got[first], want[first]);
    check(diff == 0, std::string("score_group_neon == scalar reference, ") + c.what);
  }

  // The caller zeroes only the border and the left strip, on the promise that the
  // kernel writes nothing outside [3+d, W-3) x [3, H-3). If that promise breaks the
  // recursive filter aggregates over uninitialised memory, which is a silent fault.
  const int W = 61, H = 23, d0 = 3, T = 10;
  const size_t WH = size_t(W) * H;
  std::vector<uint64_t> cl(WH), cr(WH);
  std::vector<uint8_t> Ld(WH), Rd(WH);
  for (size_t i = 0; i < WH; ++i) {
    cl[i] = rng() & mask; cr[i] = rng() & mask;
    Ld[i] = uint8_t(rng()); Rd[i] = uint8_t(rng());
  }
  int16_t tbl[64]; build_tbl(tbl, 24);
  int16_t adt[256]; build_adt(adt, T);
  std::vector<int16_t> pl(WH * SIMD_G, SENTINEL);
  score_group_neon(cl.data(), cr.data(), Ld.data(), Rd.data(), pl.data(), WH, W, H,
                   d0, tbl, adt, 153, T);
  bool untouched = true;
  for (int g = 0; g < SIMD_G; ++g)
    for (int y = 0; y < H; ++y)
      for (int x = 0; x < W; ++x) {
        const bool inside = y >= 3 && y < H - 3 && x >= 3 + d0 + g && x < W - 3;
        if (!inside && pl[size_t(g) * WH + size_t(y) * W + x] != SENTINEL)
          untouched = false;
      }
  check(untouched, "score_group_neon writes nothing outside [3+d, W-3) x [3, H-3)");
}

#endif  // DE_HAVE_NEON

}  // namespace

int main() {
  std::printf("NEON score kernel tests\n\n");
#ifdef DE_HAVE_NEON
  test_lut8();
  test_hamming8();
  test_transpose();
  test_blend8();
  test_kernel_against_reference();
  std::printf("\n%s (%d check%s, %d failure%s)\n",
              g_failures ? "FAILED" : "ALL PASSED", g_checks,
              g_checks == 1 ? "" : "s", g_failures,
              g_failures == 1 ? "" : "s");
  return g_failures == 0 ? 0 : 1;
#else
  std::printf("  SKIPPED: no NEON kernel on this architecture, 0 checks run.\n"
              "  This suite is only meaningful on the Jetson. Run it there:\n"
              "    tools/deploy.sh && ssh jetson 'cd ~/doubleeye/core && make test'\n");
  std::printf("\nSKIPPED (0 checks)\n");
  return 0;
#endif
}
