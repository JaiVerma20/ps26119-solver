#!/usr/bin/env bash
# setup_github.sh — create the PRIVATE GitHub repo for the team and push everything.
# Run once, on the Mac, by the repo owner:
#   brew install gh        (or: download from https://cli.github.com)   then:  gh auth login
#   scripts/setup_github.sh [repo-name] [teammate1 teammate2 ...]
# Example: scripts/setup_github.sh ps26119-solver alice-gh bob-gh
set -euo pipefail
cd "$(dirname "$0")/.."
NAME="${1:-ps26119-solver}"; shift || true
command -v gh >/dev/null || { echo "install GitHub CLI first: https://cli.github.com"; exit 1; }
gh auth status >/dev/null 2>&1 || { echo "run: gh auth login"; exit 1; }

if ! git remote get-url origin >/dev/null 2>&1; then
  gh repo create "$NAME" --private --source . --remote origin --description "SIH 2026 PS26119 — GPU-native LP solver" --push
else
  git push -u origin main
fi
OWNER=$(gh api user --jq .login)
# Teammates get push access (they still work through pull requests, see docs/CONTRIBUTING.md).
for u in "$@"; do
  gh api -X PUT "repos/$OWNER/$NAME/collaborators/$u" -f permission=push >/dev/null && echo "invited $u (push)"
done
# Protect main: PRs + green CI required (works on private repos only with a paid plan;
# ignore the error on a free plan and follow the convention in CONTRIBUTING.md instead).
gh api -X PUT "repos/$OWNER/$NAME/branches/main/protection" --input - <<JSON >/dev/null 2>&1 && echo "main protected" || echo "branch protection not available on this plan (fine: follow CONTRIBUTING.md)"
{"required_status_checks":{"strict":false,"contexts":["build-test (ubuntu-latest)","build-test (macos-latest)"]},
 "enforce_admins":false,"required_pull_request_reviews":{"required_approving_review_count":0},
 "restrictions":null}
JSON
echo "repo: https://github.com/$OWNER/$NAME"
