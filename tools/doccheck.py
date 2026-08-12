#!/usr/bin/env python3
"""The mechanical half of doc/12-writing.md. Standard library only.

Checks what a script can check -- links, anchors, images, math delimiters, retired
values, banned phrasing -- and reports what it can only measure -- sentence length.
The rules a script cannot check are in doc/12-writing.md and are the more important
half.

    tools/doccheck.py                 # doc/, CLAUDE.md, article/*.py docstrings
    tools/doccheck.py --blog          # also the posts and the glossary
    tools/doccheck.py --self-test     # prove every check can still fail
    tools/doccheck.py --long 30       # lower the sentence-length report threshold

Exit status is 1 if anything failed, so it can sit in front of a commit.

The anchor rule below is GitHub's and kramdown-with-GFM's, which are the same one:
lowercase, delete everything that is not alphanumeric / space / hyphen, then replace
each space with a hyphen -- EACH space, so "form -- this" becomes "form--this". It was
confirmed against the published HTML rather than taken from documentation, because
stock kramdown uses a different rule and the difference is silent (rule 6).
"""

import argparse
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BLOG = os.path.expanduser("~/src/mayio.github.io")

# Values that were retired, so they do not creep back in. Add a row whenever a default
# changes: (regex, what to say). Keep the pattern narrow -- a bare "60" would match a
# thousand innocent things. Rule 5.
RETIRED = [
    (r"--min-range\s+0\.4\b",
     "--min-range is 0.335 (dmax 64); 0.4 m was obstacle 24"),
    (r"--dmax\s+53\b",
     "dmax 53 was the 0.4 m near limit; the search quantises in blocks of 64"),
    (r"--dmax\s+60\b",
     "--dmax 60 buys a block of 64 and discards four disparities; use 64. "
     "A measurement TAKEN at D=60 should still say so -- that is rule 1, not rule 5"),
    (r"\b(98|49)\s*MB\b.{0,40}(cost )?volume|volume.{0,40}\b(98|49)\s*MB\b",
     "state the element type and D: 104 MB float / 52 MB int16 at D=64"),
]

# Rule 4: documentation describes the present. 03-obstacles.md is the one file allowed
# to be chronological, and TODO.md records dated decisions by design.
CHRONOLOGICAL_OK = {"03-obstacles.md", "TODO.md", "12-writing.md", "doccheck.py"}
BANNED = [
    (r"\bhonest(ly)?\b", "say the thing instead of claiming honesty"),
    (r"^\s*\*?\*?Update[,:]", "documentation describes the present (rule 4)"),
    (r"\bas (mentioned|noted|described) (before|above|earlier)\b",
     "link the section instead (rule 6)"),
    (r"\bpreviously\b|\bused to be\b|\bhas since been (changed|updated)\b",
     "documentation describes the present (rule 4)"),
]


def anchor(heading):
    t = heading.replace("`", "").strip().lower()
    t = "".join(c for c in t if c.isalnum() or c in " -")
    return t.replace(" ", "-")


def headings(text):
    return {anchor(h) for h in re.findall(r"^#{1,6}\s+(.+?)\s*$", text, re.M)}


def strip_code(text):
    """Blank code blocks out rather than delete them: the checks report line numbers,
    and deleting shifts every line after the first fence."""
    return re.sub(r"```.*?```",
                  lambda m: re.sub(r"[^\n]", " ", m.group(0)), text, flags=re.S)


class Report:
    def __init__(self):
        self.fails = []
        self.notes = []

    def fail(self, where, msg):
        self.fails.append(f"{where}: {msg}")

    def note(self, where, msg):
        self.notes.append(f"{where}: {msg}")


def check_file(path, rep, universe, long_words):
    """universe maps an absolute .md path -> its set of anchors, for cross-file links."""
    name = os.path.basename(path)
    raw = open(path, encoding="utf-8").read()
    body = strip_code(raw)
    own = headings(raw)
    rel = os.path.relpath(path, ROOT if path.startswith(ROOT) else BLOG)

    # --- math delimiters -----------------------------------------------------------
    if body.count("$$") % 2:
        rep.fail(rel, "odd number of $$ delimiters -- a formula is unterminated")

    # --- reference-style definitions that nothing uses -----------------------------
    for ref in re.findall(r"^\[([^\]^]+)\]:\s", body, re.M):
        if not re.search(r"\]\[" + re.escape(ref) + r"\]", body):
            rep.fail(rel, f"unused link definition [{ref}]")

    # --- links ---------------------------------------------------------------------
    for target in re.findall(r"(?<!!)\[[^\]]*\]\(([^)\s]+)\)", body):
        if target.startswith(("http://", "https://", "mailto:")):
            continue
        if target.startswith("#"):
            if target[1:] not in own:
                rep.fail(rel, f"broken intra-file anchor {target}")
            continue
        if target.startswith("/") and not target.startswith("/assets/"):
            continue                      # a Jekyll permalink, not a path on disk
        file_part, _, frag = target.partition("#")
        dest = os.path.normpath(os.path.join(os.path.dirname(path), file_part))
        if not os.path.exists(dest):
            rep.fail(rel, f"link target does not exist: {file_part}")
        elif frag and dest in universe and frag not in universe[dest]:
            rep.fail(rel, f"broken anchor {file_part}#{frag}")

    # --- images --------------------------------------------------------------------
    for img in re.findall(r"!\[[^\]]*\]\(([^)\s]+)\)", body):
        if img.startswith(("http://", "https://")):
            continue
        base = BLOG if img.startswith("/assets/") else os.path.dirname(path)
        p = os.path.normpath(os.path.join(base, img.lstrip("/") if
                                          img.startswith("/assets/") else img))
        if not os.path.exists(p):
            rep.fail(rel, f"image not found: {img}")

    # --- retired values ------------------------------------------------------------
    if name not in CHRONOLOGICAL_OK:
        for pat, why in RETIRED:
            for m in re.finditer(pat, body):
                line = body[:m.start()].count("\n") + 1
                rep.fail(f"{rel}:{line}", f"retired value {m.group(0)!r} -- {why}")

    # --- banned phrasing -----------------------------------------------------------
    for pat, why in BANNED:
        if name in CHRONOLOGICAL_OK and "rule 4" in why:
            continue
        for m in re.finditer(pat, body, re.M | re.I):
            line = body[:m.start()].count("\n") + 1
            rep.fail(f"{rel}:{line}", f"{m.group(0)!r} -- {why}")

    # --- sentence length: reported, not failed (rule 10) ----------------------------
    prose = re.sub(r"\$\$.*?\$\$", " M ", body, flags=re.S)
    prose = re.sub(r"^\s*[|>].*$", "", prose, flags=re.M)
    prose = re.sub(r"^\[.*?\]:.*$", "", prose, flags=re.M)
    prose = re.sub(r"^---$.*?^---$", "", prose, flags=re.M | re.S)     # front matter
    prose = re.sub(r"!\[[^\]]*\]\([^)]*\)", "", prose)
    prose = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", prose)
    prose = re.sub(r"\[([^\]]*)\]\[[^\]]*\]", r"\1", prose)
    n = 0
    for sent in re.split(r"(?<=[.!?])\s+(?=[A-Z*`])", prose.replace("\n", " ")):
        if len(sent.split()) >= long_words:
            n += 1
    if n:
        rep.note(rel, f"{n} sentence(s) of {long_words}+ words")


def collect(paths):
    return {p: headings(open(p, encoding="utf-8").read()) for p in paths}


def gather(blog):
    out = [os.path.join(ROOT, "CLAUDE.md")]
    d = os.path.join(ROOT, "doc")
    out += [os.path.join(d, f) for f in sorted(os.listdir(d)) if f.endswith(".md")]
    if blog:
        p = os.path.join(BLOG, "_posts")
        if not os.path.isdir(p):
            sys.exit(f"--blog: {BLOG} is not there")
        out += [os.path.join(p, f) for f in sorted(os.listdir(p))
                if f.endswith(".md") and f.startswith("2026-08-0")]
        out.append(os.path.join(BLOG, "masda-glossary.md"))
    return [p for p in out if os.path.exists(p)]


SELF_TEST = """# A heading

See [nowhere](#no-such-anchor) and [gone](./does-not-exist.md).
Run it with --min-range 0.4 which is retired.
Update, 2026-01-01: this reads as a changelog.
To be honest, this word is banned.
$$ x = 1
![missing](/assets/img/nope/nope.png)

[unused]: https://example.com
"""


def self_test():
    """Rule 12: a check that cannot be made to fail is not evidence."""
    import tempfile
    want = ["broken intra-file anchor", "link target does not exist", "retired value",
            "rule 4", "honest", "odd number of $$", "image not found",
            "unused link definition"]
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, "self-test.md")
        open(p, "w").write(SELF_TEST)
        rep = Report()
        check_file(p, rep, {}, 34)
    blob = "\n".join(rep.fails).lower()
    missing = [w for w in want if w.lower() not in blob]
    for f in rep.fails:
        print("  caught:", f)
    if missing:
        print(f"\nSELF-TEST FAILED -- these checks did not fire: {missing}")
        return 1
    print(f"\nself-test passed: all {len(want)} checks fired on a document built to "
          f"break them")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--blog", action="store_true", help="also check the posts")
    ap.add_argument("--long", type=int, default=34, metavar="N",
                    help="report sentences of N+ words (default 34)")
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args()
    if a.self_test:
        return self_test()

    files = gather(a.blog)
    universe = collect(files)
    rep = Report()
    for f in files:
        check_file(f, rep, universe, a.long)

    for n in rep.notes:
        print("  note   ", n)
    for f in rep.fails:
        print("  FAIL   ", f)
    print(f"\n{len(files)} files, {len(rep.fails)} failure(s), {len(rep.notes)} note(s)")
    if not rep.fails:
        print("the mechanical rules pass. The rest of doc/12-writing.md is judgement.")
    return 1 if rep.fails else 0


if __name__ == "__main__":
    sys.exit(main())
