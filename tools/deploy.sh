#!/usr/bin/env bash
# Sync the Jetson sources and build them, as one step.
#
# This exists because separating the two costs real time: running make on the
# Jetson without syncing first prints "Built target" and relinks a stale binary,
# so a just-written feature appears simply not to work while every signal points
# at the code. See doc/03-obstacles.md, obstacle 10.
#
# rsync is not installed on the TX2, hence tar over ssh.

set -euo pipefail

HOST="${DOUBLEEYE_HOST:-jetson}"
REMOTE_DIR="${DOUBLEEYE_REMOTE_DIR:-doubleeye}"
JOBS="${DOUBLEEYE_JOBS:-6}"

cd "$(dirname "$0")/.."

usage() {
  cat <<EOF
usage: tools/deploy.sh [--probe] [--capture ARGS...]

  (no args)          sync sources, then build
  --probe            sync, build, then run rs_probe
  --capture ARGS...  sync, build, then run rs_ir_capture with ARGS

Environment:
  DOUBLEEYE_HOST        ssh alias or host   (default: jetson)
  DOUBLEEYE_REMOTE_DIR  path under \$HOME    (default: doubleeye)
  DOUBLEEYE_JOBS        parallel build jobs (default: 6)
EOF
}

case "${1:-}" in
  -h|--help) usage; exit 0 ;;
esac

echo "==> syncing jetson/ to ${HOST}:~/${REMOTE_DIR}/"
ssh "$HOST" "mkdir -p ~/${REMOTE_DIR}"
tar czf - jetson | ssh "$HOST" "tar xzf - -C ~/${REMOTE_DIR}"

# cmake on the TX2 is 3.10.2, which supports NEITHER `-S`/`-B` (3.13+) NOR
# `--build -j` (3.12+). Passing either makes cmake print its usage text and exit
# without building, which is silent enough to look like success. So: configure
# in-directory the old way, and drive make directly. Errors are propagated, not
# filtered -- a build stage that can fail quietly is what obstacle 10 was about.
echo "==> building on ${HOST}"
if ! ssh "$HOST" "set -e
  cd ~/${REMOTE_DIR}/jetson
  mkdir -p build
  cd build
  cmake .. >cmake.log 2>&1 || { echo '--- cmake configure failed ---'; cat cmake.log; exit 1; }
  make -j${JOBS}"; then
  echo "==> BUILD FAILED" >&2
  exit 1
fi

case "${1:-}" in
  --probe)
    echo "==> rs_probe"
    ssh -t "$HOST" "cd ~/${REMOTE_DIR}/jetson/build && ./rs_probe ${*:2}"
    ;;
  --capture)
    if [[ $# -lt 2 ]]; then
      echo "--capture needs at least an output directory" >&2
      exit 2
    fi
    echo "==> rs_ir_capture ${*:2}"
    ssh -t "$HOST" "cd ~/${REMOTE_DIR}/jetson/build && ./rs_ir_capture ${*:2}"
    ;;
  "")
    ;;
  *)
    echo "unknown argument: $1" >&2
    usage >&2
    exit 2
    ;;
esac

echo "==> done"
