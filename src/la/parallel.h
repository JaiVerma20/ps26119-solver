// parallel.h — a tiny fixed-size thread pool with a deterministic parallel_for.
//
// Determinism: work is split into chunks by index ranges that depend only on n and the
// chunk count (never on timing), and every output element is written by exactly one task
// in a fixed order. Row-parallel SpMV and elementwise updates are therefore bit-identical
// for any number of threads. Reductions use per-chunk partial sums over a FIXED chunk
// count (kReduceChunks) combined in order, so they are identical for any thread count too.
//
// No OpenMP dependency (Apple clang ships without libomp). Threads = 1 runs inline.
//
// Concurrent callers (several solve() calls on different threads of one process): run() and
// set_threads() are serialised by use_m_, so one job is in the pool at a time. That is correct
// for any interleaving and keeps results bit-identical (they never depend on the thread
// count); concurrent solves simply share the pool. run() is not re-entrant (never nested).
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
    std::lock_guard<std::mutex> use(use_m_);
    if (t == threads_.load()) return;
    stop();
    threads_.store(t);
    for (int i = 1; i < t; ++i) workers_.emplace_back([this] { loop(); });
  }
  int threads() const { return threads_.load(); }

  // Runs f(chunk) for chunk = 0..chunks-1, spread over the pool; returns when all are done.
  //
  // Hand-off protocol (race-free by construction):
  //   * a job is PUBLISHED under m_ (job_, chunks_, counters, generation_), and only when no
  //     worker is active and no job is published;
  //   * a worker JOINS under m_ only if a job is published and its generation is new, and it
  //     registers itself (active_++) inside that same critical section;
  //   * when all chunks are done the job is RETIRED under m_ (job_ = nullptr), after which no
  //     worker can join it; run() then waits for active_ == 0, so `f` (on the caller's stack)
  //     outlives every use.
  // Chunks are claimed with an atomic counter; idle workers spin briefly before sleeping.
  void run(int chunks, const std::function<void(int)>& f) {
    std::lock_guard<std::mutex> use(use_m_);
    if (threads_.load() <= 1 || chunks <= 1) {
      for (int c = 0; c < chunks; ++c) f(c);
      return;
    }
    {
      std::lock_guard<std::mutex> lk(m_);
      job_ = &f;
      chunks_ = chunks;
      next_.store(0, std::memory_order_relaxed);
      done_.store(0, std::memory_order_relaxed);
      generation_.fetch_add(1, std::memory_order_release);
    }
    cv_.notify_all();
    work(&f, chunks);  // the calling thread helps
    while (done_.load(std::memory_order_acquire) != chunks) std::this_thread::yield();
    {
      std::lock_guard<std::mutex> lk(m_);
      job_ = nullptr;  // retired: nobody can join any more
    }
    while (active_.load(std::memory_order_acquire) != 0) std::this_thread::yield();
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
  void loop() {
    std::uint64_t seen = 0;
    for (;;) {
      for (int spin = 0; spin < 20000; ++spin) {  // iterations dispatch back to back: spin briefly
        if (quit_.load(std::memory_order_acquire)) return;
        if (generation_.load(std::memory_order_acquire) != seen) break;
        if ((spin & 63) == 63) std::this_thread::yield();
      }
      const std::function<void(int)>* f = nullptr;
      int chunks = 0;
      {
        std::unique_lock<std::mutex> lk(m_);
        if (generation_.load() == seen)
          cv_.wait(lk, [&] { return quit_.load() || generation_.load() != seen; });
        if (quit_.load()) return;
        seen = generation_.load();
        if (job_ != nullptr) {  // join only a published, not yet retired job
          f = job_;
          chunks = chunks_;
          active_.fetch_add(1, std::memory_order_acq_rel);
        }
      }
      if (f) {
        work(f, chunks);
        active_.fetch_sub(1, std::memory_order_acq_rel);
      }
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

  std::atomic<int> threads_{1};
  std::mutex use_m_;  // one job (or resize) at a time, see the header comment
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
