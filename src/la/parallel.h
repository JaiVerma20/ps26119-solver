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
    for (int i = 1; i < threads_; ++i) workers_.emplace_back([this, i] { loop(i); });
  }
  int threads() const { return threads_; }

  // Runs f(chunk) for chunk = 0..chunks-1, spread over the pool; returns when all are done.
  void run(int chunks, const std::function<void(int)>& f) {
    if (threads_ <= 1 || chunks <= 1) {
      for (int c = 0; c < chunks; ++c) f(c);
      return;
    }
    {
      std::unique_lock<std::mutex> lk(m_);
      job_ = &f;
      chunks_ = chunks;
      next_ = 0;
      done_ = 0;
      ++generation_;
    }
    cv_.notify_all();
    work();  // the calling thread helps
    std::unique_lock<std::mutex> lk(m_);
    finished_.wait(lk, [&] { return done_ == chunks_; });
    job_ = nullptr;
  }

  ~ThreadPool() { stop(); }

 private:
  ThreadPool() = default;
  void work() {
    for (;;) {
      int c;
      const std::function<void(int)>* f;
      {
        std::lock_guard<std::mutex> lk(m_);
        if (!job_ || next_ >= chunks_) return;
        c = next_++;
        f = job_;
      }
      (*f)(c);
      {
        std::lock_guard<std::mutex> lk(m_);
        if (++done_ == chunks_) finished_.notify_all();
      }
    }
  }
  void loop(int) {
    std::uint64_t seen = 0;
    for (;;) {
      {
        std::unique_lock<std::mutex> lk(m_);
        cv_.wait(lk, [&] { return quit_ || (generation_ != seen && job_ != nullptr); });
        if (quit_) return;
        seen = generation_;
      }
      work();
    }
  }
  void stop() {
    {
      std::lock_guard<std::mutex> lk(m_);
      quit_ = true;
    }
    cv_.notify_all();
    for (auto& w : workers_) w.join();
    workers_.clear();
    quit_ = false;
  }

  int threads_ = 1;
  std::vector<std::thread> workers_;
  std::mutex m_;
  std::condition_variable cv_, finished_;
  const std::function<void(int)>* job_ = nullptr;
  int chunks_ = 0, next_ = 0, done_ = 0;
  std::uint64_t generation_ = 0;
  bool quit_ = false;
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
