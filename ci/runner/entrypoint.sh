#!/usr/bin/env bash
# Registers the runner on first start (RUNNER_TOKEN: a short-lived registration token from
# scripts/ci_runners.sh), then runs it. The registration lives in the container's filesystem,
# so `docker restart` / Docker Desktop restarts need no new token.
set -euo pipefail
cd /home/runner/actions-runner
if [ ! -f .runner ]; then
  : "${RUNNER_TOKEN:?first start needs RUNNER_TOKEN}" "${REPO_URL:?}" "${RUNNER_NAME:?}" "${RUNNER_LABELS:?}"
  ./config.sh --unattended --url "$REPO_URL" --token "$RUNNER_TOKEN" --name "$RUNNER_NAME" \
    --labels "$RUNNER_LABELS" --work _work --replace
fi
unset RUNNER_TOKEN
exec ./run.sh
