# ci/runner/Dockerfile — a self-hosted GitHub Actions runner for the Linux jobs of
# .github/workflows/ci.yml (build-test ubuntu, asan-ubsan, tsan), run in Docker on a team laptop.
# Used when GitHub-hosted runners are unavailable (self-hosted runners are not billed).
# Built and started by scripts/ci_runners.sh. Ubuntu 24.04 like ubuntu-latest; architecture is the
# host's (arm64 on Apple Silicon — GitHub's ubuntu-latest is x86_64).
FROM ubuntu:24.04

ARG RUNNER_VERSION=2.337.0
ARG RUNNER_ARCH=arm64
# from the release notes of https://github.com/actions/runner/releases/tag/v2.337.0
ARG RUNNER_SHA256=9b1dc70626422526e3c94767cf024896beb15da5342a3f4819bf2feac13e0393

ENV DEBIAN_FRONTEND=noninteractive
# what the workflow needs beyond the runner: compilers (gcc for build-test, clang for the
# sanitizers — libclang-rt holds the ASan/UBSan/TSan runtimes — and the CUDA syntax check), git, python
RUN apt-get update && apt-get install -y --no-install-recommends \
      ca-certificates curl git jq sudo unzip xz-utils zstd lsb-release tzdata \
      build-essential clang libclang-rt-18-dev lld llvm binutils python3 python3-venv python3-pip libicu74 \
    && rm -rf /var/lib/apt/lists/*

RUN useradd -m -s /bin/bash runner && echo "runner ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/runner
USER runner
WORKDIR /home/runner/actions-runner
RUN curl -fsSL -o runner.tgz \
      "https://github.com/actions/runner/releases/download/v${RUNNER_VERSION}/actions-runner-linux-${RUNNER_ARCH}-${RUNNER_VERSION}.tar.gz" \
    && echo "${RUNNER_SHA256}  runner.tgz" | sha256sum -c - \
    && tar xzf runner.tgz && rm runner.tgz \
    && sudo ./bin/installdependencies.sh

COPY --chown=runner:runner entrypoint.sh /home/runner/entrypoint.sh
ENTRYPOINT ["/home/runner/entrypoint.sh"]
