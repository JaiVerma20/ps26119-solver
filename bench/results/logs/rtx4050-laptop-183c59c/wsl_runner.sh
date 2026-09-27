#!/usr/bin/env bash
# Runs Jai's gpu_check.sh inside WSL, directly on the Windows friend-code folder,
# with a Linux-only PATH (no Windows tools).
# Build dir and Python venv live in the WSL home (fast disk, not synced by OneDrive).
# Usage: run_gpu_check.sh quick|full
export PATH=/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/lib/wsl/lib
cd /mnt/c/Users/vatss/OneDrive/Desktop/gpu_optimizer/friend-code || exit 1
# friend-code is a Windows git worktree; its .git file holds a C:/ path Linux git can't follow.
export GIT_DIR=/mnt/c/Users/vatss/OneDrive/Desktop/gpu_optimizer/.git/worktrees/friend-code
export GIT_WORK_TREE="$PWD"
export BUILD=/home/vatsshivanshu/ps26119-build-gpu
# gpu_check.sh uses "$ROOT/$VENV", so VENV must be relative to the repo root.
export VENV=../../../../../../../../home/vatsshivanshu/.venv-ps26119-gpu
if [ "$1" = quick ]; then export QUICK=1; fi
PS26119_MACHINE=rtx4050-laptop scripts/gpu_check.sh > ~/gpu_$1.out 2>&1
echo "EXIT=$?"
tail -60 ~/gpu_$1.out
