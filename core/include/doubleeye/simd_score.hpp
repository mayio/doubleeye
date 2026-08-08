// The vector score kernel for the dense matcher: disparity in the LANES, planes
// still disparity-major.
//
// The question this settles is the `[d][x]` versus `[x][d]` one that arrived three
// times -- from the solver, from band fusion, and from SIMD. It is a false choice.
// The score wants disparity in the REGISTER FILE and the recursive filter wants it
// in MEMORY, and those are independent: nothing about the loads is disparity-minor
// either way. The right descriptors for eight consecutive disparities at pixel x are
// eight consecutive uint64 of the right census plane, and the left descriptor is a
// broadcast. So the group is computed with disparity in the lanes and TRANSPOSED IN
// REGISTERS before it is stored, and the filter, the top-2 insert and the solver see
// exactly the planes they saw before.
//
// This is not the write amplification the blocked transpose existed to manage. That
// wrote one 4-byte disparity per 64-byte line of a [x][d] volume -- 16x at line
// granularity. An in-register transpose deposits sixteen contiguous bytes per plane
// and consecutive x groups continue the same lines: eight sequential write streams,
// no partial lines.
//
// Why the previous attempt failed, from doc/09-matching.md: it vectorised eight
// PIXELS' popcounts and left the loop pixel-major, so `tbl[hamming]` and
// `adt[|L-R|]` stayed per-pixel gathers and took over the loop. With disparity in the
// lanes both become vector table lookups, because the eight Hamming distances land in
// one register and `vqtbl4q_u8` reads a 64-byte table in one instruction. 49 Census
// levels and ad_trunc+1 truncated-difference levels both fit 64 entries, so the
// arithmetic reformulation that file called necessary is not needed and bit-identity
// with the scalar loop survives.
//
// **Measured on the TX2: 1.59x on the score loop and 0.92x on the cost stage** at
// D=60 over six threads, because the eight-disparity work quantum costs more
// occupancy than the kernel saves. It is a net win only where the group count
// divides the core count comfortably (1.04x at D=96). Off by default; see
// doc/09-matching.md before spending time here.
//
// The pieces are separate inline functions rather than one loop body because the
// lane order and the transpose permutation are PERMUTATION ARGUMENTS, which is the
// class of mistake that produces plausible-looking output. Both were wrong once.
// tests/test_simd_score.cpp checks each against a scalar reference, and checks the
// composed kernel against an independent scalar implementation.

#ifndef DOUBLEEYE_SIMD_SCORE_HPP
#define DOUBLEEYE_SIMD_SCORE_HPP

#include <algorithm>
#include <cstdint>
#include <cstdlib>

// Deliberately not behind an -march flag: vcntq_u8, vpaddq_u8, vqtbl4q_u8 and the
// trn/zip permutes are all in the mandatory ARMv8-A base, so core/Makefile keeps
// ARCHFLAGS empty and the same source still builds on the desktop.
#ifdef __aarch64__
#include <arm_neon.h>
#define DE_HAVE_NEON 1
#endif

namespace doubleeye {

constexpr int SIMD_G = 8;   // disparities per group = int16 lanes per register

#ifdef DE_HAVE_NEON

// The two byte tables vqtbl4q_u8 reads, from one int16 table of 64 entries.
struct SplitTable {
  uint8x16x4_t lo, hi;
};

inline SplitTable simd_split_table(const int16_t* t64) {
  uint8_t lo[64], hi[64];
  for (int i = 0; i < 64; ++i) {
    lo[i] = uint8_t(uint16_t(t64[i]) & 0xFF);
    hi[i] = uint8_t(uint16_t(t64[i]) >> 8);
  }
  SplitTable s;
  for (int i = 0; i < 4; ++i) {
    s.lo.val[i] = vld1q_u8(lo + 16 * i);
    s.hi.val[i] = vld1q_u8(hi + 16 * i);
  }
  return s;
}

// Eight Hamming distances against a broadcast left descriptor, as BYTES 0..7 in the
// address order of rp[0..7]. Four XORs, four vcntq_u8, four vpaddq_u8: the addp tree
// pairs across registers, so it collapses 8 descriptors x 8 bytes to 8 sums without
// any widening. Max Hamming is 48 for a 48-bit descriptor, so nothing wraps u8, and
// the descriptor's top two bytes are zero so no masking is needed.
inline uint8x16_t simd_hamming8(const uint64_t* rp, uint64_t left) {
  const uint8x16_t lx = vreinterpretq_u8_u64(vdupq_n_u64(left));
  const uint8x16_t e0 = veorq_u8(vreinterpretq_u8_u64(vld1q_u64(rp + 0)), lx);
  const uint8x16_t e1 = veorq_u8(vreinterpretq_u8_u64(vld1q_u64(rp + 2)), lx);
  const uint8x16_t e2 = veorq_u8(vreinterpretq_u8_u64(vld1q_u64(rp + 4)), lx);
  const uint8x16_t e3 = veorq_u8(vreinterpretq_u8_u64(vld1q_u64(rp + 6)), lx);
  const uint8x16_t s0 = vpaddq_u8(vcntq_u8(e0), vcntq_u8(e1));
  const uint8x16_t s1 = vpaddq_u8(vcntq_u8(e2), vcntq_u8(e3));
  const uint8x16_t h0 = vpaddq_u8(s0, s1);
  return vpaddq_u8(h0, h0);
}

// Eight int16 table entries from eight byte indices in lanes 0..7. Indices >= 64
// would read zero, which is why the caller clamps.
inline int16x8_t simd_lut8(const SplitTable& t, uint8x16_t idx) {
  return vreinterpretq_s16_u8(
      vzip1q_u8(vqtbl4q_u8(t.lo, idx), vqtbl4q_u8(t.hi, idx)));
}

// (c*(1024-wq) + a*wq) >> 10, widened so the products are exact and the shift
// truncates exactly the way the scalar int32 expression does.
inline int16x8_t simd_blend8(int16x8_t c16, int16x8_t a16, int16_t wc, int16_t wa) {
  int32x4_t lo = vmull_n_s16(vget_low_s16(c16), wc);
  lo = vmlal_n_s16(lo, vget_low_s16(a16), wa);
  int32x4_t hi = vmull_n_s16(vget_high_s16(c16), wc);
  hi = vmlal_n_s16(hi, vget_high_s16(a16), wa);
  return vcombine_s16(vshrn_n_s32(lo, 10), vshrn_n_s32(hi, 10));
}

// 8x8 int16 transpose: out[j] lane p == in[p] lane j. Three levels of trn at 16, 32
// and 64-bit granularity, 24 instructions for 64 values.
inline void simd_transpose8x8(const int16x8_t* in, int16x8_t* out) {
  int16x8_t a[8], b[8];
  for (int i = 0; i < 8; i += 2) {
    a[i]     = vtrn1q_s16(in[i], in[i + 1]);
    a[i + 1] = vtrn2q_s16(in[i], in[i + 1]);
  }
  for (int i = 0; i < 8; i += 4)
    for (int j = 0; j < 2; ++j) {
      const int32x4_t p = vreinterpretq_s32_s16(a[i + j]);
      const int32x4_t q = vreinterpretq_s32_s16(a[i + j + 2]);
      b[i + j]     = vreinterpretq_s16_s32(vtrn1q_s32(p, q));
      b[i + j + 2] = vreinterpretq_s16_s32(vtrn2q_s32(p, q));
    }
  for (int j = 0; j < 4; ++j) {
    const int64x2_t p = vreinterpretq_s64_s16(b[j]);
    const int64x2_t q = vreinterpretq_s64_s16(b[j + 4]);
    out[j]     = vreinterpretq_s16_s64(vtrn1q_s64(p, q));
    out[j + 4] = vreinterpretq_s16_s64(vtrn2q_s64(p, q));
  }
}

// Score SIMD_G disparities into SIMD_G consecutive H*W int16 planes; plane g is
// disparity d0 + g. Writes x in [3 + d0 + g, W-3) for y in [3, H-3) and nothing
// else -- the border and the left strip are the caller's to zero, because the
// recursive filter propagates whatever it finds there into the valid window.
//
// `tbl` must have 64 entries and `adt` at least T+1, with 1 <= T <= 63 so the
// clamped difference is a legal vqtbl4q_u8 index. de_dense checks both.
inline void score_group_neon(const uint64_t* cl, const uint64_t* cr,
                             const uint8_t* Ld, const uint8_t* Rd,
                             int16_t* planes, size_t WH, int W, int H, int d0,
                             const int16_t* tbl, const int16_t* adt,
                             int32_t wq, int T) {
  const SplitTable TB = simd_split_table(tbl);
  int16_t aclamped[64];
  for (int i = 0; i < 64; ++i) aclamped[i] = adt[i <= T ? i : T];
  const SplitTable AB = simd_split_table(aclamped);

  const int16_t wc = int16_t(1024 - wq), wa = int16_t(wq);
  const uint8x8_t tdup = vdup_n_u8(uint8_t(T));
  const int dtop = d0 + SIMD_G - 1;   // lane j holds disparity dtop - j
  const int xv0 = 3 + dtop;           // first x at which all SIMD_G are in bounds
  const int xhi = W - 3;

  // One pixel-disparity, in exactly the form of the reference loop. Used for the
  // left strip where only part of the group is in bounds, and for the x tail.
  auto scalar_one = [&](int g, size_t row, int x) {
    const size_t i = row + x;
    const int32_t c = tbl[__builtin_popcountll(cl[i] ^ cr[i - size_t(d0 + g)])];
    const int32_t a = adt[std::abs(int(Ld[i]) - int(Rd[i - size_t(d0 + g)]))];
    planes[size_t(g) * WH + i] = int16_t((c * (1024 - wq) + a * wq) >> 10);
  };

  for (int y = 3; y < H - 3; ++y) {
    const size_t row = size_t(y) * W;
    for (int g = 0; g < SIMD_G; ++g)
      for (int x = 3 + d0 + g; x < std::min(xv0, xhi); ++x) scalar_one(g, row, x);
    int x = xv0;
    for (; x + SIMD_G <= xhi; x += SIMD_G) {
      int16x8_t v[SIMD_G];
      for (int p = 0; p < SIMD_G; ++p) {
        const size_t i = row + x + p;
        // rp[j] is the descriptor at disparity dtop - j: one contiguous load, and
        // the reversal is absorbed by the plane index at the store.
        const uint64_t* rp = cr + (i - size_t(dtop));
        const uint8x16_t hh = simd_hamming8(rp, cl[i]);
        const int16x8_t c16 = simd_lut8(TB, hh);
        // Truncated absolute difference: the same eight right pixels, one 8-byte
        // load, clamped to T so the 256-entry table collapses into 64.
        const uint8x8_t r8 = vld1_u8(Rd + (i - size_t(dtop)));
        const uint8x8_t ad8 = vmin_u8(vabd_u8(vdup_n_u8(Ld[i]), r8), tdup);
        const int16x8_t a16 = simd_lut8(AB, vcombine_u8(ad8, ad8));
        v[p] = simd_blend8(c16, a16, wc, wa);
      }
      int16x8_t out[SIMD_G];
      simd_transpose8x8(v, out);
      // Lane j held disparity dtop - j = d0 + (SIMD_G-1-j), so register j is that
      // plane's eight consecutive pixels.
      for (int j = 0; j < SIMD_G; ++j)
        vst1q_s16(planes + size_t(SIMD_G - 1 - j) * WH + row + x, out[j]);
    }
    for (; x < xhi; ++x)
      for (int g = 0; g < SIMD_G; ++g) scalar_one(g, row, x);
  }
}

#endif  // DE_HAVE_NEON

}  // namespace doubleeye

#endif  // DOUBLEEYE_SIMD_SCORE_HPP
