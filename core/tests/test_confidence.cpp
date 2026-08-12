// The per-pixel confidence, held to the fit that produced it.
//
// `confidence()` in dense_solve.hpp carries two hard-coded numbers from a logistic
// regression fitted in article/confidence.py. Nothing in the build connects the two,
// so a transcription slip -- a digit, a sign, a feature in the wrong place -- would
// produce a plausible number between 0 and 1 for every pixel and nothing would
// complain. The cloud would still look like a cloud.
//
// The expected values below were printed by that fit. They are not round numbers and
// they are not derived from the same expression twice: they came out of numpy on the
// eight Middlebury scenes, and this file only checks that the C++ agrees.
//
// It also checks the properties the number must have whatever the weights are, which
// is the part that survives a refit: monotone in each feature in the direction the
// feature argues, bounded, and defined where a pixel has no runner-up at all.

#include "doubleeye/dense_solve.hpp"

#include <cmath>
#include <cstdio>
#include <string>

using doubleeye::confidence;
using doubleeye::confidence_u8;

namespace {

int failures = 0;
const float LAMBDA = -0.1f;

void check(bool ok, const std::string& what) {
  if (!ok) { std::printf("  [FAIL] %s\n", what.c_str()); ++failures; }
  else       std::printf("  [PASS] %s\n", what.c_str());
}

void close(float got, float want, float tol, const std::string& what) {
  const bool ok = std::fabs(got - want) <= tol;
  if (!ok)
    std::printf("  [FAIL] %s: got %.6f, want %.6f\n", what.c_str(), got, want);
  else
    std::printf("  [PASS] %s (%.6f)\n", what.c_str(), got);
  failures += ok ? 0 : 1;
}

}  // namespace

int main() {
  std::printf("confidence(): the fitted probability, against article/confidence.py\n");

  // Five points from the fit. The last has no second candidate, which the solver
  // signals with -1e30 and the model floors to lambda.
  close(confidence(0.9f, 0.3f, LAMBDA),     1.000000f, 1e-5f, "s1 0.90, s2 0.30");
  close(confidence(0.5f, 0.49f, LAMBDA),    0.700536f, 1e-5f, "s1 0.50, s2 0.49");
  close(confidence(0.682f, 0.537f, LAMBDA), 0.975261f, 1e-5f, "the eight-scene mean");
  close(confidence(0.2f, -0.5f, LAMBDA),    0.961586f, 1e-5f, "s2 below lambda");
  close(confidence(0.3f, -1e30f, LAMBDA),   0.986346f, 1e-5f, "no second candidate");

  // The worked example in the header must be the number the code returns, or the
  // comment is a claim rather than a derivation.
  close(confidence(0.682f, 0.537f, LAMBDA), 0.975f, 5e-4f, "header's worked example");

  std::printf("\nproperties that must hold whatever the weights are\n");

  check(confidence(-1e30f, -1e30f, LAMBDA) == 0.f,
        "no candidate at all returns exactly 0");

  // A REGION WITH NO TEXTURE. The Census descriptor is constant there, so every
  // disparity matches equally and s1 == s2 at whatever level the region sits at.
  // Both cues correctly say "no idea", and the answer must not depend on how good
  // that meaningless perfect score was.
  //
  // This is not hypothetical. A three-feature version of this model that also read
  // the winning score returned 0.94 here, and on a synthetic frame the black padding
  // scored HIGHER than the real image beside it. Nothing in the benchmark caught it,
  // because Middlebury is textured almost everywhere.
  bool flat_ok = true, flat_same = true;
  const float flat_ref = confidence(0.95f, 0.95f, LAMBDA);
  for (float lv = -0.05f; lv < 0.99f; lv += 0.05f) {
    const float p = confidence(lv, lv, LAMBDA);
    if (p >= confidence(0.7f, 0.2f, LAMBDA)) flat_ok = false;   // below a real match
    if (p > 0.75f) flat_ok = false;                             // and below any gate
    if (std::fabs(p - flat_ref) > 1e-5f) flat_same = false;
  }
  check(flat_ok, "a textureless region scores below 0.75 at every brightness");
  check(flat_same, "and scores the SAME whatever the brightness, since s1 says nothing");

  // Monotone in the margin: widen the gap to the runner-up, holding the winner, and
  // confidence must rise. This is the direction --min-margin has always assumed.
  bool mono = true;
  for (float s2 = 0.6f; s2 > -0.2f; s2 -= 0.05f)
    if (confidence(0.7f, s2, LAMBDA) < confidence(0.7f, s2 + 0.05f, LAMBDA))
      mono = false;
  check(mono, "rises as the runner-up falls away, s1 = 0.7");

  // Monotone in the winner's own score at a fixed margin. A pixel that matches well
  // AND beats its rival is worth more than one that merely beats its rival.
  mono = true;
  for (float s1 = -0.2f; s1 < 0.9f; s1 += 0.05f)
    if (confidence(s1 + 0.05f, s1 - 0.05f, LAMBDA)
        < confidence(s1, s1 - 0.1f, LAMBDA))
      mono = false;
  check(mono, "rises with the winning score at a fixed margin");

  // Bounded, and the byte form must not wrap. -32768 through 32767 as scores is far
  // outside anything Q14 produces, which is the point: a NaN or a garbage read must
  // saturate rather than alias onto a confident value.
  bool bounded = true;
  for (float s1 = -3.f; s1 <= 3.f; s1 += 0.1f)
    for (float s2 = -3.f; s2 <= 3.f; s2 += 0.1f) {
      const float p = confidence(s1, s2, LAMBDA);
      if (!(p >= 0.f && p <= 1.f)) bounded = false;
      const int u = confidence_u8(s1, s2, LAMBDA);
      if (u < 0 || u > 255) bounded = false;
    }
  check(bounded, "stays in [0, 1] and [0, 255] over scores far outside Q14");

  check(confidence_u8(0.9f, 0.3f, LAMBDA) == 255 &&
        confidence_u8(0.5f, 0.49f, LAMBDA) == 179,
        "the byte form rounds rather than truncates");

  std::printf("\nthe reverse match, which is a cap and not a weight\n");

  // The case the cap exists for, and the one Middlebury cannot fit: a match with no
  // competitor scores a confident ratio, and the reverse match is the only cue that
  // objects. Obstacle 24a's ghost. A linear model handed this returns 0.998.
  close(confidence(0.9f, 0.3f, LAMBDA, 9.0f), 0.35f, 1e-6f,
        "a strong ratio the reverse match rejects");
  close(confidence(0.9f, 0.3f, LAMBDA, 0.0f), 1.0f, 1e-5f,
        "the same pixel when the reverse match agrees");

  // It can only lower. A cue that could raise confidence would need its own
  // calibration, and this one is measured as a population that is 31.8% correct.
  bool only_down = true;
  for (float s = -0.2f; s < 0.95f; s += 0.05f)
    for (float g = -0.3f; g < s; g += 0.1f)
      if (confidence(s, g, LAMBDA, 9.0f) > confidence(s, g, LAMBDA, 0.0f))
        only_down = false;
  check(only_down, "a rejection never raises confidence, at any score");

  // A poor absolute match with no runner-up: s1 = -0.4 is at the low end of what the
  // Q14 scores actually reach, and it lands at 0.23, below the cap.
  check(confidence(-0.4f, -1e30f, LAMBDA) < 0.35f &&
        confidence(-0.4f, -1e30f, LAMBDA, 9.0f)
            == confidence(-0.4f, -1e30f, LAMBDA, 0.f),
        "and does not touch a pixel already below the cap");

  // Not-a-number is what the solver writes where the right pixel had no claimant at
  // all. That is not a disagreement, and treating it as one would flag the border of
  // every frame.
  check(confidence(0.9f, 0.3f, LAMBDA, std::nanf("")) ==
        confidence(0.9f, 0.3f, LAMBDA, 0.0f),
        "NaN means unanswered, not rejected");

  std::printf(failures ? "\nFAILED (%d failures)\n" : "\nALL PASSED (%d failures)\n",
              failures);
  return failures ? 1 : 0;
}
