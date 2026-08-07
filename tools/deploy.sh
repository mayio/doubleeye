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
usage: tools/deploy.sh [--probe] [--capture NAME [ARGS...]] [--pull NAME]

  (no args)               sync sources, then build
  --probe                 sync, build, then run rs_probe
  --capture NAME [ARGS]   sync, build, then record to ~/bags/NAME on the Jetson
  --pull NAME             copy ~/bags/NAME from the Jetson into ./bags/NAME

NAME is a bare name, not a path. It always lands in ~/bags/NAME on the Jetson.
That is deliberate: writing '~/bags/x' here would expand against the DESKTOP
home before ssh ever sees it, and the Jetson would then fail to create
/home/<desktop-user>/bags/x. Taking a name removes the whole class of mistake.

Environment:
  DOUBLEEYE_HOST        ssh alias or host   (default: jetson)
  DOUBLEEYE_REMOTE_DIR  path under \$HOME    (default: doubleeye)
  DOUBLEEYE_JOBS        parallel build jobs (default: 6)
EOF
}

# Reject anything that is a path rather than a name, so a stray ~/ or / cannot
# silently resolve against the wrong machine's filesystem.
check_name() {
  case "$1" in
    "" ) echo "error: missing NAME" >&2; exit 2 ;;
    */*|~*) echo "error: '$1' looks like a path. Pass a bare name; it goes to" >&2
            echo "       ~/bags/$(basename "$1") on the Jetson." >&2; exit 2 ;;
  esac
}

case "${1:-}" in
  -h|--help) usage; exit 0 ;;
  --pull)
    check_name "${2:-}"
    NAME="$2"
    echo "==> pulling ~/bags/$NAME from ${HOST}"
    mkdir -p "bags/$NAME"
    ssh "$HOST" "test -d ~/bags/$NAME" || {
      echo "error: ~/bags/$NAME does not exist on ${HOST}" >&2
      echo "available:" >&2; ssh "$HOST" "ls ~/bags 2>/dev/null" >&2; exit 1; }
    ssh "$HOST" "cd ~/bags/$NAME && tar czf - ." | tar xzf - -C "bags/$NAME"
    echo "==> bags/$NAME"
    du -sh "bags/$NAME"
    exit 0
    ;;
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
    check_name "${2:-}"
    NAME="$2"
    echo "==> rs_ir_capture -> ${HOST}:~/bags/$NAME ${*:3}"
    # ~ is quoted so it expands on the REMOTE side, not here.
    ssh -t "$HOST" "mkdir -p ~/bags && rm -rf ~/bags/$NAME && \
      cd ~/${REMOTE_DIR}/jetson/build && ./rs_ir_capture ~/bags/$NAME ${*:3}"
    echo
    echo "==> pull it with:  ./tools/deploy.sh --pull $NAME"
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
