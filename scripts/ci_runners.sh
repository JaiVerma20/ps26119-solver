#!/usr/bin/env bash
# ci_runners.sh — self-hosted GitHub Actions runners for this repository on a team laptop
# (Apple Silicon Mac + Docker): one macOS runner (build-test macos-latest) and one Linux runner in
# Docker (build-test ubuntu-latest, asan-ubsan, tsan). Self-hosted runners are not billed, so CI
# keeps working when GitHub-hosted minutes are unavailable.
#
#   scripts/ci_runners.sh setup     download (checksum-verified), register, start; sets CI_RUNNER
#   scripts/ci_runners.sh start     start both (after a reboot / Docker restart)
#   scripts/ci_runners.sh stop      stop both (jobs then wait in the queue)
#   scripts/ci_runners.sh status    local processes + what GitHub sees
#   scripts/ci_runners.sh hosted    route CI back to GitHub-hosted runners (repo variable)
#   scripts/ci_runners.sh remove    deregister and delete both runners
#
# Needs: gh (logged in, admin on the repo), Docker running, curl, shasum. The workflow chooses the
# runners from the repository variable CI_RUNNER (self-hosted | anything else = GitHub-hosted).
# Registration tokens are fetched from GitHub with gh and never printed; the Linux one stays in the
# container's metadata until it expires (1 hour; it can only register runners to this repository).
set -euo pipefail
cd "$(dirname "$0")/.."

VERSION=2.337.0
MAC_SHA256=5a2cd92908a93d7276a194e1de6008099f3e7946f3f8e14aa7a1a7b4a31fdec2  # osx-arm64, release notes
REPO=$(gh repo view --json nameWithOwner -q .nameWithOwner)
HOME_DIR="${PS26119_RUNNER_HOME:-$HOME/ps26119-runners}"
MAC_DIR="$HOME_DIR/mac"
HOST=$(hostname -s)
MAC_NAME="ps26119-mac-$HOST"
LINUX_NAME="ps26119-linux-$HOST"
IMAGE=ps26119-runner-linux
CONTAINER=ps26119-runner-linux

token() { gh api -X POST "repos/$REPO/actions/runners/$1" --jq .token; }

# The runner is run.sh -> run-helper.sh -> bin/Runner.Listener (-> Runner.Worker per job): judge
# and stop it by its processes, not by a pid file — killing only run.sh leaves the listener alive,
# and a second start then runs two listeners on one work folder (jobs collide and fail).
mac_pids() { pgrep -f "$MAC_DIR/(run.sh|run-helper.sh|bin/Runner.Listener|bin/Runner.Worker)" || true; }
mac_running() { pgrep -f "$MAC_DIR/bin/Runner.Listener" >/dev/null; }

start_mac() {
  if mac_running; then echo "macOS runner already running (listener pid $(pgrep -f "$MAC_DIR/bin/Runner.Listener" | tr '\n' ' '))"; return; fi
  (cd "$MAC_DIR" && nohup ./run.sh >> run.log 2>&1 & echo $! > "$MAC_DIR/run.pid")
  echo "macOS runner started (log $MAC_DIR/run.log)"
}

start_linux() {
  docker info >/dev/null 2>&1 || { echo "Docker is not running: open -a Docker, then retry"; exit 1; }
  docker start "$CONTAINER" >/dev/null && echo "Linux runner started (docker logs -f $CONTAINER)"
}

case "${1:-status}" in
  setup)
    [ "$(uname -sm)" = "Darwin arm64" ] || { echo "setup expects an Apple Silicon Mac"; exit 1; }
    docker info >/dev/null 2>&1 || { echo "Docker is not running: open -a Docker, then retry"; exit 1; }
    mkdir -p "$MAC_DIR"
    if [ ! -f "$MAC_DIR/.runner" ]; then
      tgz="$MAC_DIR/runner.tgz"
      curl -fsSL -o "$tgz" "https://github.com/actions/runner/releases/download/v$VERSION/actions-runner-osx-arm64-$VERSION.tar.gz"
      echo "$MAC_SHA256  $tgz" | shasum -a 256 -c -
      tar xzf "$tgz" -C "$MAC_DIR" && rm "$tgz"
      (cd "$MAC_DIR" && ./config.sh --unattended --url "https://github.com/$REPO" --token "$(token registration-token)" \
        --name "$MAC_NAME" --labels ps26119 --work _work --replace)
    fi
    start_mac
    if ! docker container inspect "$CONTAINER" >/dev/null 2>&1; then
      docker build -t "$IMAGE" ci/runner
      docker create --name "$CONTAINER" --restart unless-stopped --memory 6g \
        -e RUNNER_TOKEN="$(token registration-token)" -e REPO_URL="https://github.com/$REPO" \
        -e RUNNER_NAME="$LINUX_NAME" -e RUNNER_LABELS=ps26119 "$IMAGE" >/dev/null
    fi
    start_linux
    gh variable set CI_RUNNER --body self-hosted --repo "$REPO"
    echo "CI_RUNNER=self-hosted: the workflow now runs on these runners"
    ;;
  start) start_mac; start_linux ;;
  stop)
    pids=$(mac_pids)
    if [ -n "$pids" ]; then
      kill $pids 2>/dev/null || true
      for _ in 1 2 3 4 5 6 7 8 9 10; do [ -z "$(mac_pids)" ] && break; sleep 1; done
      [ -n "$(mac_pids)" ] && kill -9 $(mac_pids) 2>/dev/null || true
      echo "macOS runner stopped"
    fi
    rm -f "$MAC_DIR/run.pid"
    docker stop "$CONTAINER" >/dev/null 2>&1 && echo "Linux runner stopped" || true
    ;;
  status)
    n=$(pgrep -f "$MAC_DIR/bin/Runner.Listener" | wc -l | tr -d ' ')
    if [ "$n" -gt 1 ]; then echo "macOS runner: $n listeners running — run: $0 stop; $0 start"
    elif [ "$n" -eq 1 ]; then echo "macOS runner: running (listener pid $(pgrep -f "$MAC_DIR/bin/Runner.Listener"))"; else echo "macOS runner: stopped"; fi
    echo "Linux runner: $(docker inspect -f '{{.State.Status}}' "$CONTAINER" 2>/dev/null || echo 'not created')"
    echo "CI_RUNNER=$(gh variable get CI_RUNNER --repo "$REPO" 2>/dev/null || echo '(unset: GitHub-hosted)')"
    gh api "repos/$REPO/actions/runners" --jq '.runners[] | "GitHub sees \(.name): \(.status)\(if .busy then " (busy)" else "" end)"'
    ;;
  hosted)
    gh variable set CI_RUNNER --body github-hosted --repo "$REPO"
    echo "CI_RUNNER=github-hosted: the workflow runs on GitHub's runners again (billed)"
    ;;
  remove)
    "$0" stop || true
    if [ -f "$MAC_DIR/.runner" ]; then (cd "$MAC_DIR" && ./config.sh remove --token "$(token remove-token)"); fi
    docker rm -f "$CONTAINER" >/dev/null 2>&1 || true
    for id in $(gh api "repos/$REPO/actions/runners" --jq ".runners[] | select(.name==\"$LINUX_NAME\") | .id"); do
      gh api -X DELETE "repos/$REPO/actions/runners/$id"
    done
    rm -rf "$MAC_DIR"
    gh variable set CI_RUNNER --body github-hosted --repo "$REPO"
    echo "runners removed; CI routed back to GitHub-hosted runners"
    ;;
  *) sed -n '2,20p' "$0"; exit 2 ;;
esac
