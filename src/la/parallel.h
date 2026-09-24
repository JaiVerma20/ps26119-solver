// parallel.h — a tiny fixed-size thread pool with a deterministic parallel_for.
//
// Determinism: work is split into chunks by index ranges that depend only on n and the
// chunk count (never on timing), and every output element is written by exactly one task
// in a fixed order. Row-parallel SpMV and elementwise updates are therefore bit-identical
// for any number of threads. Reductions use per-chunk partial sums over a FIXED chunk
// count (kReduceChunks) combined in order, so they are identical for any thread count too.
//
// No OpenMP dependency (Apple clang ships without libomp). Threads = 1 runs inline.
#pragma once

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <functional>
#include <mutex>
#include <thread>
#include <vector>

namespace ps26119::la {

class ThreadPool {
 public:
  static ThreadPool& instance() {
    static ThreadPool pool;
    return pool;
  }
  // 0 = hardware concurrency. Changing the size restarts the workers.
  void set_threads(int t) {
    if (t <= 0) t = static_cast<int>(std::max(1u, std::thread::hardware_concurrency()));
    if (t == threads_) return;
    stop();
    threads_ = t;
    for (int i = 1; i < threads_; ++i) workers_.emplace_back([this] { loop(); });
  }
  int threads() const { return threads_; }

  // Runs f(chunk) for chunk = 0..chunks-1, spread over the pool; returns when all are done.
  // Chunks are claimed with an atomic counter; idle workers spin briefly (then sleep), so a
  // dispatch costs microseconds, not a condition-variable round trip per call.
  void run(int chunks, const std::function<void(int)>& f) {
    if (threads_ <= 1 || chunks <= 1) {
      for (int c = 0; c < chunks; ++c) f(c);
      return;
    }
    // No worker may still be inside the previous job while we publish a new one.
    while (active_.load(std::memory_order_acquire) != 0) std::this_thread::yield();
    job_ = &f;
    chunks_ = chunks;
    next_.store(0, std::memory_order_relaxed);
    done_.store(0, std::memory_order_relaxed);
    {
      std::lock_guard<std::mutex> lk(m_);
      generation_.fetch_add(1, std::memory_order_release);  // publishes the job
    }
    cv_.notify_all();
    work(job_, chunks_);  // the calling thread helps
    while (done_.load(std::memory_order_acquire) != chunks) std::this_thread::yield();
    // f lives on the caller's stack: wait until every worker has left before returning.
    while (active_.load(std::memory_order_acquire) != 0) std::this_thread::yield();
    job_ = nullptr;
  }

  ~ThreadPool() { stop(); }

 private:
  ThreadPool() = default;
  void work(const std::function<void(int)>* f, int chunks) {
    for (;;) {
      const int c = next_.fetch_add(1, std::memory_order_acq_rel);
      if (c >= chunks) return;
      (*f)(c);
      done_.fetch_add(1, std::memory_order_acq_rel);
    }
  }
  // Worker protocol: announce yourself (active_++), THEN read the generation. The dispatcher
  // only (re)publishes when active_ == 0, so a worker that sees a new generation sees a fully
  // published job, and a worker that sees the old one simply backs off.
  void loop() {
    std::uint64_t seen = generation_.load(std::memory_order_acquire);
    for (;;) {
      for (int spin = 0; spin < 20000; ++spin) {  // iterations dispatch back to back: spin briefly
        if (quit_.load(std::memory_order_acquire)) return;
        if (generation_.load(std::memory_order_acquire) != seen) break;
        if ((spin & 63) == 63) std::this_thread::yield();
      }
      if (generation_.load(std::memory_order_acquire) == seen) {
        std::unique_lock<std::mutex> lk(m_);
        cv_.wait(lk, [&] { return quit_.load() || generation_.load() != seen; });
        if (quit_.load()) return;
      }
      active_.fetch_add(1, std::memory_order_acq_rel);
      const std::uint64_t g = generation_.load(std::memory_order_acquire);
      if (g != seen) {
        seen = g;
        work(job_, chunks_);
      }
      active_.fetch_sub(1, std::memory_order_acq_rel);
    }
  }
  void stop() {
    {
      std::lock_guard<std::mutex> lk(m_);
      quit_.store(true);
    }
    cv_.notify_all();
    for (auto& w : workers_) w.join();
    workers_.clear();
    quit_.store(false);
  }

  int threads_ = 1;
  std::vector<std::thread> workers_;
  std::mutex m_;
  std::condition_variable cv_;
  const std::function<void(int)>* job_ = nullptr;
  int chunks_ = 0;
  std::atomic<int> next_{0}, done_{0}, active_{0};
  std::atomic<std::uint64_t> generation_{0};
  std::atomic<bool> quit_{false};
};

// Fixed chunk count for reductions (independent of the thread count ⇒ same rounding).
inline constexpr int kReduceChunks = 64;

// Parallel loop over [0, n) in contiguous chunks; small n runs inline.
template <class F>
void parallel_for(std::int64_t n, F&& f, std::int64_t min_per_chunk = 16384) {
  ThreadPool& pool = ThreadPool::instance();
  const int t = pool.threads();
  if (t <= 1 || n < 2 * min_per_chunk) {
    f(std::int64_t{0}, n);
    return;
  }
  const int chunks = static_cast<int>(std::min<std::int64_t>(4 * t, n / min_per_chunk));
  pool.run(chunks, [&](int c) {
    const std::int64_t b = n * c / chunks, e = n * (c + 1) / chunks;
    f(b, e);
  });
}

// Deterministic sum of g(i) over [0, n): fixed chunking, partial sums combined in order.
template <class G>
double parallel_sum(std::int64_t n, G&& g) {
  double part[kReduceChunks] = {};
  const int chunks = static_cast<int>(std::min<std::int64_t>(kReduceChunks, std::max<std::int64_t>(1, n)));
  auto body = [&](int c) {
    const std::int64_t b = n * c / chunks, e = n * (c + 1) / chunks;
    double s = 0;
    for (std::int64_t i = b; i < e; ++i) s += g(i);
    part[c] = s;
  };
  ThreadPool& pool = ThreadPool::instance();
  if (pool.threads() <= 1 || n < 65536) {
    for (int c = 0; c < chunks; ++c) body(c);
  } else {
    pool.run(chunks, body);
  }
  double s = 0;
  for (int c = 0; c < chunks; ++c) s += part[c];
  return s;
}

}  // namespace ps26119::la
