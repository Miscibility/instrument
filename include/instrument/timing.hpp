/**
 * @file timing.hpp
 * @brief A small, scoped timing library for HPC / numerical code.
 *
 * Features:
 *  - RAII scoped timers (miscibility::instrument::Timer) building a per-thread call tree.
 *  - A non-RAII span API (miscibility::instrument::start / miscibility::instrument::stop) for regions that
 *    do not match lexical scope (e.g. spanning an MPI_Isend / MPI_Wait pair).
 *  - Compile-time timing levels: a `Timer<Level>` (or `start<Level>`) compiles to
 *    nothing when `Level > MISCIBILITY_INSTRUMENT_TIMING_LEVEL`. `MISCIBILITY_INSTRUMENT_TIMING_LEVEL` is intended to be set by CMake.
 *  - Statistics: call count, inclusive/exclusive time, time per iteration, percentage
 *    of total and of parent; queryable by path.
 *  - Human-readable reports (aligned ASCII or GitHub Markdown) to a stream or file.
 *  - MPI rank-aware reduction reporting (enable with `-DMISCIBILITY_INSTRUMENT_WITH_MPI`),
 *    surfacing load imbalance (max/mean across ranks) per region.
 *
 * @par Overhead
 * The steady-state hot path is lock-free and allocation-free: region nodes are
 * created once on first encounter and reused thereafter; accumulators are integer
 * nanoseconds. A disabled `Timer<Level>` emits code identical to no timer at all.
 *
 * @par Header-only
 * Globals use `inline` storage; include this header anywhere.
 *
 * C++23.
 */

#pragma once

#include <algorithm>
#include <array>
#include <charconv>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <ostream>
#include <sstream>
#include <string>
#include <string_view>
#include <type_traits>
#include <vector>

#ifdef MISCIBILITY_INSTRUMENT_WITH_MPI
#include <mpi.h>
#endif

/**
 * @def MISCIBILITY_INSTRUMENT_TIMING_LEVEL
 * @brief Compile-time threshold controlling which timers are active.
 *
 * A timer with level `L` is compiled in iff `L <= MISCIBILITY_INSTRUMENT_TIMING_LEVEL`. Give coarse,
 * always-on regions a *low* level and fine hot-loop regions a *high* level; then a
 * production build sets a low `MISCIBILITY_INSTRUMENT_TIMING_LEVEL` (fine timers vanish) and a profiling
 * build sets a high one. Override from CMake with `-DMISCIBILITY_INSTRUMENT_TIMING_LEVEL=<n>`.
 */
#ifndef MISCIBILITY_INSTRUMENT_TIMING_LEVEL
#define MISCIBILITY_INSTRUMENT_TIMING_LEVEL 2 // default: a moderate level that avoids timing inside loops
#endif

/// @namespace miscibility::instrument
/// @brief Instrument library for the Miscibility project; this header adds timing facilities.
namespace miscibility::instrument {

/// @brief Clock used for all measurements (monotonic, not subject to wall-clock jumps).
using clock = std::chrono::steady_clock;

/**
 * @brief Snapshot of a single region's statistics, returned by query().
 *
 * All times are in seconds. "Inclusive" time covers the region and everything it
 * called; "exclusive"/self time is inclusive minus the time attributed to children.
 */
struct Stats {
    std::string name;         ///< Region (leaf) name.
    std::uint64_t calls = 0;  ///< Number of times the region was entered (incl. recursion).
    double total_seconds = 0; ///< Inclusive time.
    double self_seconds = 0;  ///< Exclusive time (inclusive minus children).
    double avg_seconds = 0;   ///< total / (number of outermost completions).
    double pct_total = 0;     ///< Percentage of the whole measured program.
    double pct_parent = 0;    ///< Percentage of the immediate parent region.
};

/// @brief Output format for reports.
enum class Format : std::uint8_t {
    Plain,   ///< Aligned ASCII table for terminals.
    Markdown ///< GitHub-flavoured Markdown table.
};

/// @namespace miscibility::instrument::detail
/// @brief Implementation details. Not part of the public API; documented for maintainers.
namespace detail {

/**
 * @internal
 * @brief Compare a std::string to a std::string_view without allocating.
 * @param a Owning string.
 * @param b View to compare against.
 * @return true iff the contents are equal.
 */
inline bool eq(const std::string& a, std::string_view b) { return a.size() == b.size() && std::string_view(a) == b; }

/**
 * @internal
 * @brief A node in a per-thread call tree. One node per (parent, name) pair.
 *
 * Nodes are created lazily on first entry and then reused, so steady-state entry
 * costs no allocation. Identity is structural: the path from the root names a region.
 */
struct Node {
    std::string name;                            ///< Region name ("<root>" for the root).
    Node* parent = nullptr;                      ///< Owning parent, or nullptr for the root.
    std::vector<std::unique_ptr<Node>> children; ///< Child regions, owned.

    std::uint64_t calls = 0;    ///< Entries, including recursive re-entries.
    std::uint64_t samples = 0;  ///< Outermost completions (denominator for avg).
    std::uint64_t total_ns = 0; ///< Inclusive time in nanoseconds.
    int active = 0;             ///< Re-entrancy depth; >1 means recursion in progress.

    /**
     * @brief Find the child with the given name, creating it if absent.
     * @param n Child region name.
     * @return Pointer to the (existing or newly created) child node.
     * @note Linear scan; sibling counts are small in practice and comparisons are
     *       cheap (length check then memcmp).
     */
    Node* child(std::string_view n)
    {
        for (auto& c : children) {
            if (eq(c->name, n)) {
                return c.get();
            }
        }
        auto up = std::make_unique<Node>();
        Node* c = up.get();
        c->name.assign(n);
        c->parent = this;
        children.push_back(std::move(up));
        return c;
    }
};

/**
 * @internal
 * @brief A live region activation: the node plus when it started.
 *
 * Returned by begin_region() and consumed by end_region(). For RAII timers a Frame
 * lives inside the Timer object on the C++ stack; for the manual API it lives on a
 * per-thread stack (see ThreadReg::manual_stack).
 */
struct Frame {
    Node* node = nullptr;    ///< The node this activation refers to.
    clock::time_point start; ///< Timestamp captured at entry.
    bool descended = false;  ///< True if entry moved `current` (false for folded recursion).
};

/**
 * @internal
 * @brief Program-lifetime registry holding one tree root per thread.
 *
 * Per-thread roots live here (not in thread-local storage) so the data survives
 * thread join and is available at report time. The mutex is taken only when a new
 * thread first registers a root and at report/query time — never on the hot path.
 */
struct Global {
    std::mutex mtx;                           ///< Guards `roots` and `t0`.
    std::vector<std::unique_ptr<Node>> roots; ///< One root per thread that used a timer.
    clock::time_point t0 = clock::now();      ///< Reference time for wall-clock reporting.

    /// @brief Meyers singleton accessor (thread-safe initialization).
    static Global& instance()
    {
        static Global g;
        return g;
    }

    /// @brief Allocate and register a fresh root for a thread. @return The new root.
    Node* new_root()
    {
        std::lock_guard lk(mtx);
        auto up = std::make_unique<Node>();
        up->name = "<root>";
        Node* r = up.get();
        roots.push_back(std::move(up));
        return r;
    }
};

/**
 * @internal
 * @brief Per-thread handle into the global registry.
 *
 * Holds the current position in this thread's call tree plus the manual-span stack.
 * The hot path manipulates only plain pointers here; no synchronization is needed
 * because each thread owns its own tree.
 */
struct ThreadReg {
    Node* root;                      ///< This thread's tree root (owned by Global).
    Node* current;                   ///< Current open region (== root when idle).
    std::vector<Frame> manual_stack; ///< Pending start()/stop() activations (LIFO).

    /// @brief Register this thread's root on construction.
    ThreadReg() : root(Global::instance().new_root()), current(root) {}

    /// @brief Access the calling thread's handle (created on first use per thread).
    static ThreadReg& local()
    {
        thread_local ThreadReg t;
        return t;
    }
};

/**
 * @internal
 * @brief Enter a region: descend (or fold direct recursion) and start timing.
 * @param reg  The calling thread's handle.
 * @param name Region name.
 * @return A Frame describing this activation, to be passed to end_region().
 *
 * Direct recursion (re-entering the region that is already current) is folded into
 * a single node: the entry is counted but wall time is accumulated only when the
 * outermost activation completes, preventing unbounded tree growth.
 */
inline Frame begin_region(ThreadReg& reg, std::string_view name)
{
    Node* cur = reg.current;
    Node* node = nullptr;
    bool descended = false;
    if (cur != reg.root && eq(cur->name, name)) {
        node = cur; // direct recursion -> same node
        descended = false;
    }
    else {
        node = cur->child(name);
        reg.current = node;
        descended = true;
    }
    ++node->calls;
    ++node->active;
    return Frame{.node = node, .start = clock::now(), .descended = descended}; // start AFTER bookkeeping
}

/**
 * @internal
 * @brief Leave a region: accumulate time (outermost only) and pop.
 * @param reg The calling thread's handle.
 * @param f   The Frame returned by the matching begin_region().
 */
inline void end_region(ThreadReg& reg, const Frame& f)
{
    const auto end = clock::now(); // stop BEFORE bookkeeping
    Node* node = f.node;
    if (--node->active == 0) {
        node->total_ns +=
            static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(end - f.start).count());
        ++node->samples;
    }
    if (f.descended) {
        reg.current = node->parent;
    }
}

/**
 * @internal
 * @brief Sum structurally-identical paths from every thread into one aggregate tree.
 * @param dst Destination (aggregate) node.
 * @param src Source (per-thread) node.
 */
inline void merge(Node& dst, const Node& src)
{
    dst.calls += src.calls;
    dst.samples += src.samples;
    dst.total_ns += src.total_ns;
    for (const auto& c : src.children) {
        merge(*dst.child(c->name), *c);
    }
}

/**
 * @internal
 * @brief Build the single-process aggregate tree across all threads.
 * @return Owning aggregate root whose total_ns is set to the sum of top-level times
 *         (used as the "% of total" denominator).
 */
inline std::unique_ptr<Node> aggregate()
{
    auto root = std::make_unique<Node>();
    root->name = "<root>";
    Global& g = Global::instance();
    std::lock_guard lk(g.mtx);
    for (auto& r : g.roots) {
        merge(*root, *r);
    }
    std::uint64_t top = 0;
    for (auto& c : root->children) {
        top += c->total_ns;
    }
    root->total_ns = top;
    return root;
}

/**
 * @internal
 * @brief Format a duration (seconds) with an adaptive unit (ns/us/ms/s).
 * @param seconds Duration in seconds.
 * @return Human-readable string, e.g. "3.767 ms".
 */
inline std::string humanize(double seconds)
{
    std::ostringstream os;
    os << std::fixed << std::setprecision(3);
    if (seconds >= 1.0) {
        os << seconds << " s";
    }
    else if (seconds >= 1e-3) {
        os << seconds * 1e3 << " ms";
    }
    else if (seconds >= 1e-6) {
        os << seconds * 1e6 << " us";
    }
    else {
        os << seconds * 1e9 << " ns";
    }
    return os.str();
}

/**
 * @internal
 * @brief Render a table to a stream in either Plain (aligned) or Markdown form.
 * @param os   Output stream.
 * @param hdr  Column headers.
 * @param rows Row cells (each row must have hdr.size() entries).
 * @param fmt  Output format.
 * @note The first column is left-aligned (names); the rest are right-aligned (numbers).
 */
inline void render(std::ostream& os, const std::vector<std::string>& hdr,
                   const std::vector<std::vector<std::string>>& rows, Format fmt)
{
    const std::size_t n = hdr.size();
    if (fmt == Format::Markdown) {
        os << '|';
        for (const auto& h : hdr) {
            os << ' ' << h << " |";
        }
        os << "\n|";
        for (std::size_t i = 0; i < n; ++i) {
            os << (i == 0 ? ":--" : "--:") << '|';
        }
        os << '\n';
        for (const auto& r : rows) {
            os << '|';
            for (const auto& c : r) {
                os << ' ' << c << " |";
            }
            os << '\n';
        }
        return;
    }
    std::vector<std::size_t> w(n);
    for (std::size_t i = 0; i < n; ++i) {
        w[i] = hdr[i].size();
    }
    for (const auto& r : rows) {
        for (std::size_t i = 0; i < n; ++i) {
            w[i] = std::max(w[i], r[i].size());
        }
    }
    auto rule = [&] {
        for (std::size_t i = 0; i < n; ++i) {
            os << '+' << std::string(w[i] + 2, '-');
        }
        os << "+\n";
    };
    auto line = [&](const std::vector<std::string>& r) {
        for (std::size_t i = 0; i < n; ++i) {
            os << "| " << (i == 0 ? std::left : std::right) << std::setw(static_cast<int>(w[i])) << r[i] << ' ';
        }
        os << "|\n";
    };
    rule();
    line(hdr);
    rule();
    for (const auto& r : rows) {
        line(r);
    }
    rule();
}

/// @internal @brief Indent prefix for a tree row at the given depth and format.
inline std::string indent_for(int depth, Format fmt)
{
    std::string s;
    if (depth <= 1) {
        return s;
    }
    if (fmt == Format::Markdown) {
        for (int d = 0; d < depth - 1; ++d) {
            s += "&nbsp;&nbsp;&nbsp;&nbsp;";
        }
    }
    else {
        // s.assign(static_cast<std::size_t>(2 * (depth - 1)), ' ');
        s.assign(long{2} * (depth - 1), ' ');
    }
    return s + "- ";
}

} // namespace detail

/**
 * @brief Build a Stats snapshot from a node.
 * @param n        The node.
 * @param denom_ns "% of total" denominator (whole-program inclusive ns).
 * @param parent   The node's parent (for "% of parent"), or nullptr.
 * @return Populated Stats.
 */
inline Stats to_stats(const detail::Node& n, std::uint64_t denom_ns, const detail::Node* parent)
{
    Stats s;
    s.name = n.name;
    s.calls = n.calls;
    std::uint64_t child_ns = 0;
    for (const auto& c : n.children) {
        child_ns += c->total_ns;
    }
    s.total_seconds = static_cast<double>(n.total_ns) / 1e9;
    s.self_seconds = static_cast<double>((n.total_ns - child_ns)) / 1e9;
    s.avg_seconds = n.samples ? (static_cast<double>(n.total_ns) / static_cast<double>(n.samples)) / 1e9 : 0;
    s.pct_total = denom_ns ? 100.0 * static_cast<double>(n.total_ns) / static_cast<double>(denom_ns) : 0;
    s.pct_parent = (parent && parent->total_ns)
                       ? 100.0 * static_cast<double>(n.total_ns) / static_cast<double>(parent->total_ns)
                       : 0;
    return s;
}

/**
 * @brief Look up a region's aggregated statistics by slash-separated path.
 * @param path e.g. "solve/spmv". Leading/trailing/double slashes are tolerated.
 * @return Stats for the region, or std::nullopt if no such path exists.
 *
 * @code
 * if (auto s = miscibility::instrument::query("solve/spmv"))
 *     std::cout << s->calls << " calls, " << s->pct_total << "% of total\n";
 * @endcode
 */
inline std::optional<Stats> query(std::string_view path)
{
    auto root = detail::aggregate();
    const std::uint64_t denom = root->total_ns;
    detail::Node* cur = root.get();
    detail::Node* parent = nullptr;
    std::size_t i = 0;
    while (i <= path.size()) {
        std::size_t j = path.find('/', i);
        if (j == std::string_view::npos) {
            j = path.size();
        }
        std::string_view seg = path.substr(i, j - i);
        if (!seg.empty()) {
            detail::Node* next = nullptr;
            for (auto& c : cur->children) {
                if (detail::eq(c->name, seg)) {
                    next = c.get();
                    break;
                }
            }
            if (!next) {
                return std::nullopt;
            }
            parent = cur;
            cur = next;
        }
        if (j == path.size()) {
            break;
        }
        i = j + 1;
    }
    if (cur == root.get()) {
        return std::nullopt;
    }
    return to_stats(*cur, denom, parent);
}

// ===========================================================================
//  RAII timer
// ===========================================================================
/**
 * @brief RAII scoped timer. Times the enclosing scope and records into the call tree.
 * @tparam Level Timing level. The timer is active iff `Level <= MISCIBILITY_INSTRUMENT_TIMING_LEVEL`; when
 *               inactive the class is empty and all members are no-ops (zero overhead,
 *               verified: identical codegen to having no timer).
 *
 * Construct a *named* object (not a temporary) so it lives for the whole scope.
 * Nesting is automatic: a timer constructed while another is alive becomes its child.
 *
 * @par Basic usage
 * @code
 * void solve() {
 *     miscibility::instrument::Timer<> t{"solve"};        // level 1 (default)
 *     assemble();
 *     {
 *         miscibility::instrument::Timer<> s{"spmv"};      // nested under "solve"
 *         for (int i = 0; i < iters; ++i) {
 *             miscibility::instrument::Timer<3> it{"spmv_iter"};  // level 3: off if MISCIBILITY_INSTRUMENT_TIMING_LEVEL<3
 *             matvec();
 *         }
 *     }
 * }
 * @endcode
 *
 * @par Pitfall: temporaries
 * @code
 * miscibility::instrument::Timer<>{"x"};   // BUG: temporary dies at the ';', times nothing
 * miscibility::instrument::Timer<> t{"x"}; // OK: named, lives until end of scope
 * auto t = miscibility::instrument::scoped_timer("x");  // OK: factory + [[nodiscard]] guard
 * @endcode
 *
 * @par Conditional compilation from CMake
 * @code
 * // production build:  -DMISCIBILITY_INSTRUMENT_TIMING_LEVEL=1  -> Timer<3> regions compile away entirely
 * // profiling  build:  -DMISCIBILITY_INSTRUMENT_TIMING_LEVEL=9  -> all timers active
 * @endcode
 */
template<int Level = 1> class Timer {
    /// @brief Whether this timer is compiled in.
    static constexpr bool On = (Level <= MISCIBILITY_INSTRUMENT_TIMING_LEVEL);

    /// @brief State carried by an active timer.
    struct State {
        detail::ThreadReg* reg = nullptr; ///< Cached thread handle (one TLS lookup).
        detail::Frame frame;              ///< This activation.
    };
    /// @brief Empty placeholder used when the timer is inactive.
    struct Empty {};
    /// @brief Active timers carry State; inactive ones carry nothing (no_unique_address).
    [[no_unique_address]] std::conditional_t<On, State, Empty> s_;

public:
    /**
     * @brief Enter and start timing the region @p name.
     * @param name Region name (typically a string literal).
     */
    explicit Timer(std::string_view name)
    {
        if constexpr (On) {
            s_.reg = &detail::ThreadReg::local();
            s_.frame = detail::begin_region(*s_.reg, name);
        }
    }

    /// @brief Stop timing and pop the region.
    ~Timer()
    {
        if constexpr (On) {
            detail::end_region(*s_.reg, s_.frame);
        }
    }

    Timer(const Timer&) = delete;            ///< Non-copyable (owns a live activation).
    Timer& operator=(const Timer&) = delete; ///< Non-copyable.
    Timer(Timer&&) = delete;                 ///< Non-movable (frame must not migrate).
    Timer& operator=(Timer&&) = delete;      ///< Non-movable.
};

/**
 * @brief Create a scoped Timer to bind to a local variable.
 * @tparam Level Timing level (same semantics as Timer); inactive when Level > MISCIBILITY_INSTRUMENT_TIMING_LEVEL.
 * @param name Region name.
 * @return A Timer<Level> that times until the bound variable goes out of scope.
 *
 * This is the function-form alternative to a scoped-timer macro. Bind the result to a
 * named variable; thanks to guaranteed copy elision (C++17) the returned Timer is built
 * directly in that variable even though Timer is non-movable. The `[[nodiscard]]` guards
 * the temporary pitfall: discarding the result is diagnosed at compile time.
 *
 * @code
 * void solve() {
 *     auto t = miscibility::instrument::scoped_timer("solve");        // level 1 (default)
 *     for (int i = 0; i < iters; ++i) {
 *         auto it = miscibility::instrument::scoped_timer<3>("spmv");  // off if MISCIBILITY_INSTRUMENT_TIMING_LEVEL < 3
 *         matvec();
 *     }
 * }
 *
 * miscibility::instrument::scoped_timer("oops");   // WARNING ([[nodiscard]]): times nothing
 * @endcode
 */
template<int Level = 1> [[nodiscard]] inline Timer<Level> scoped_timer(std::string_view name)
{
    return Timer<Level>{name};
}

// ===========================================================================
//  Non-RAII span API
// ===========================================================================
/**
 * @brief Begin a timed region not bound to lexical scope. Pair with stop().
 * @tparam Level Timing level (same semantics as Timer); inactive levels are no-ops.
 * @param name Region name.
 *
 * Spans must nest LIFO with each other and with any active Timer objects (the call
 * tree is a stack). Use this when start and stop straddle a non-scope boundary.
 *
 * @code
 * miscibility::instrument::start("halo_exchange");
 * MPI_Isend(...); MPI_Irecv(...);
 * compute_interior();                  // overlap
 * MPI_Waitall(...);
 * miscibility::instrument::stop("halo_exchange");  // name is validated against the open span
 * @endcode
 */
template<int Level = 1> inline void start(std::string_view name)
{
    if constexpr (Level <= MISCIBILITY_INSTRUMENT_TIMING_LEVEL) {
        auto& reg = detail::ThreadReg::local();
        reg.manual_stack.push_back(detail::begin_region(reg, name));
    }
}

/**
 * @brief End the most recently started span at this @p Level.
 * @tparam Level Must match the corresponding start() level.
 * @param expected Optional region name; if non-empty it is checked against the open
 *                 span and a warning is written to std::cerr on mismatch (mis-nesting).
 */
template<int Level = 1> inline void stop(std::string_view expected = {})
{
    if constexpr (Level <= MISCIBILITY_INSTRUMENT_TIMING_LEVEL) {
        auto& reg = detail::ThreadReg::local();
        if (reg.manual_stack.empty()) {
            std::cerr << "[miscibility::instrument::timing] stop() with no open span\n";
            return;
        }
        detail::Frame f = reg.manual_stack.back();
        reg.manual_stack.pop_back();
        if (!expected.empty() && !detail::eq(f.node->name, expected)) {
            std::cerr << "[miscibility::instrument::timing] stop(\"" << expected << "\") closed \"" << f.node->name
                      << "\" (mis-nested spans?)\n";
        }
        detail::end_region(reg, f);
    }
}

// ===========================================================================
//  Reporting (single process)
// ===========================================================================
namespace detail {

/**
 * @internal
 * @brief Depth-first emit table rows for a subtree, hottest child first.
 * @param node  Current node.
 * @param denom "% of total" denominator (ns).
 * @param depth Tree depth (root == 0; the root itself is skipped).
 * @param rows  Accumulated output rows.
 * @param fmt   Output format (controls indentation style).
 */
inline void emit_rows(const Node& node, std::uint64_t denom, int depth, std::vector<std::vector<std::string>>& rows,
                      Format fmt)
{
    if (node.parent != nullptr) {
        Stats s = to_stats(node, denom, node.parent);
        std::ostringstream pt;
        std::ostringstream pp;
        pt << std::fixed << std::setprecision(1) << s.pct_total;
        pp << std::fixed << std::setprecision(1) << s.pct_parent;
        rows.push_back({indent_for(depth, fmt) + s.name, std::to_string(s.calls), humanize(s.total_seconds),
                        humanize(s.self_seconds), humanize(s.avg_seconds), pt.str(), pp.str()});
    }
    std::vector<const Node*> kids;
    kids.reserve(node.children.size());
    for (const auto& c : node.children) {
        kids.push_back(c.get());
    }
    std::ranges::sort(kids, [](const Node* a, const Node* b) { return a->total_ns > b->total_ns; });
    for (const Node* c : kids) {
        emit_rows(*c, denom, depth + 1, rows, fmt);
    }
}

} // namespace detail

/**
 * @brief Write a timing report (all threads aggregated) to a stream.
 * @param os  Output stream.
 * @param fmt Output format (default Plain).
 */
inline void report(std::ostream& os, Format fmt = Format::Plain)
{
    auto root = detail::aggregate();
    const std::uint64_t denom = root->total_ns;
    const double wall =
        std::chrono::duration_cast<std::chrono::duration<double>>(clock::now() - detail::Global::instance().t0).count();
    const std::vector<std::string> hdr = {"Region", "Calls", "Total", "Self", "Avg", "%Total", "%Parent"};
    std::vector<std::vector<std::string>> rows;
    detail::emit_rows(*root, denom, 0, rows, fmt);

    if (fmt == Format::Markdown) {
        os << "## Timing report\n\nMeasured total: **" << detail::humanize(static_cast<double>(denom) / 1e9)
           << "**, wall clock since init: " << detail::humanize(wall) << ".\n\n";
        detail::render(os, hdr, rows, fmt);
        os << "\n*Total = inclusive; Self = exclusive (region minus children).*\n";
    }
    else {
        os << "Timing report  (measured total " << detail::humanize(static_cast<double>(denom) / 1e9) << ", wall "
           << detail::humanize(wall) << ")\n";
        detail::render(os, hdr, rows, fmt);
    }
}

/**
 * @brief Write a timing report to a file.
 * @param path Output path.
 * @param fmt  Output format (default Markdown).
 */
inline void report(const std::filesystem::path& path, Format fmt = Format::Markdown)
{
    std::ofstream f(path);
    report(f, fmt);
}

/// @brief Write a Plain report to std::cout.
inline void report(Format fmt = Format::Plain) { report(std::cout, fmt); }

/**
 * @brief Clear all collected timing data and reset the wall-clock reference.
 * @warning Intended for single-threaded / benchmark use while no timers are active.
 *          Resetting with live timers or concurrent threads leaves dangling state.
 */
inline void reset()
{
    // Materialize this thread's handle *before* locking: a first-time
    // ThreadReg construction acquires the same mutex internally, which would
    // self-deadlock if we already held it.
    auto& tr = detail::ThreadReg::local();
    auto& g = detail::Global::instance();
    std::lock_guard lk(g.mtx);
    for (auto& r : g.roots) {
        r->children.clear();
        r->calls = r->samples = r->total_ns = 0;
        r->active = 0;
    }
    tr.current = tr.root;
    tr.manual_stack.clear();
    g.t0 = clock::now();
}

// ===========================================================================
//  MPI rank-aware reduction (transport-agnostic core; testable without MPI)
// ===========================================================================
namespace detail {

/// @internal @brief Append a uint64 in decimal to @p out.
inline void append_u64(std::string& out, std::uint64_t v) { out += std::to_string(v); }

/**
 * @internal
 * @brief Serialize a per-process aggregate tree to a text blob (one region per line).
 *
 * Line format (tab-separated): `path \t calls \t total_ns`. Region names must not
 * contain '/', tab, or newline (paths are slash-joined).
 * @param n    Current node (call with the aggregate root).
 * @param path Scratch path string (call with "").
 * @param out  Destination blob.
 */
inline void serialize_rec(const Node& n, std::string& path, std::string& out)
{
    if (n.parent) {
        const std::size_t base = path.size();
        if (!path.empty()) {
            path += '/';
        }
        path += n.name;
        out += path;
        out += '\t';
        append_u64(out, n.calls);
        out += '\t';
        append_u64(out, n.total_ns);
        out += '\n';
        for (const auto& c : n.children) {
            serialize_rec(*c, path, out);
        }
        path.resize(base);
    }
    else {
        for (const auto& c : n.children) {
            serialize_rec(*c, path, out);
        }
    }
}

/// @internal @brief Convenience: serialize a whole tree to a blob. @param root Tree root.
inline std::string serialize(const Node& root)
{
    std::string path;
    std::string out;
    serialize_rec(root, path, out);
    return out;
}

/// @internal @brief Cross-rank accumulation for one region.
struct RankAgg {
    std::uint64_t calls_sum = 0;             ///< Sum of entries over all ranks.
    std::uint64_t total_ns_sum = 0;          ///< Sum of inclusive ns over ranks (for mean).
    std::uint64_t total_ns_min = UINT64_MAX; ///< Min inclusive ns over ranks.
    std::uint64_t total_ns_max = 0;          ///< Max inclusive ns over ranks (for imbalance).
    int ranks_present = 0;                   ///< Number of ranks that recorded this region.
};

/// @internal @brief Node of the reduced cross-rank tree (built on the root rank).
struct MpiNode {
    std::string name;                               ///< Region name.
    MpiNode* parent = nullptr;                      ///< Owning parent or nullptr.
    std::vector<std::unique_ptr<MpiNode>> children; ///< Children, owned.
    RankAgg agg;                                    ///< Cross-rank stats for this region.

    /// @brief Find/create a child by name. @param n Name. @return Child node.
    MpiNode* child(std::string_view n)
    {
        for (auto& c : children) {
            if (eq(c->name, n)) {
                return c.get();
            }
        }
        auto up = std::make_unique<MpiNode>();
        MpiNode* c = up.get();
        c->name.assign(n);
        c->parent = this;
        children.push_back(std::move(up));
        return c;
    }
};

/// @internal @brief Result of reducing per-rank blobs.
struct MpiReduce {
    std::unique_ptr<MpiNode> root; ///< Reduced tree root.
    std::uint64_t denom = 0;       ///< "% of total" denominator (sum of top-level total_ns_sum).
};

/// @internal @brief Parse a decimal uint64 from a token. @param sv Token. @return Value.
inline std::uint64_t parse_u64(std::string_view sv)
{
    std::uint64_t v = 0;
    std::from_chars(sv.data(), sv.data() + sv.size(), v);
    return v;
}

/**
 * @internal
 * @brief Reduce per-rank serialized blobs into one cross-rank tree.
 * @param blobs  One blob per rank (as produced by serialize()).
 * @param nranks Communicator size (used as the mean denominator).
 * @return Reduced tree and the "% of total" denominator.
 */
inline MpiReduce reduce_blobs(const std::vector<std::string>& blobs, int nranks)
{
    // TODO: clangd marks this as too high of congitive complexity. See if it can be simplified.
    MpiReduce out;
    out.root = std::make_unique<MpiNode>();
    out.root->name = "<root>";
    for (const auto& blob : blobs) {
        std::size_t pos = 0;
        while (pos < blob.size()) {
            std::size_t nl = blob.find('\n', pos);
            if (nl == std::string::npos) {
                nl = blob.size();
            }
            std::string_view line(blob.data() + pos, nl - pos);
            pos = nl + 1;
            if (line.empty()) {
                continue;
            }
            std::array<std::string_view, 3> f{};
            std::size_t fi = 0;
            std::size_t start = 0;
            for (std::size_t i = 0; i <= line.size() && fi < 3; ++i) {
                if (i == line.size() || line[i] == '\t') {
                    f[fi] = line.substr(start, i - start);
                    ++fi;
                    start = i + 1;
                }
            }
            if (fi < 3) {
                continue;
            }
            MpiNode* node = out.root.get();
            std::string_view p = f[0];
            std::size_t s = 0;
            while (s <= p.size()) {
                std::size_t sl = p.find('/', s);
                if (sl == std::string_view::npos) {
                    sl = p.size();
                }
                std::string_view seg = p.substr(s, sl - s);
                if (!seg.empty()) {
                    node = node->child(seg);
                }
                if (sl == p.size()) {
                    break;
                }
                s = sl + 1;
            }
            const std::uint64_t total = parse_u64(f[2]);
            node->agg.calls_sum += parse_u64(f[1]);
            node->agg.total_ns_sum += total;
            node->agg.total_ns_min = std::min(node->agg.total_ns_min, total);
            node->agg.total_ns_max = std::max(node->agg.total_ns_max, total);
            ++node->agg.ranks_present;
        }
    }
    for (auto& c : out.root->children) {
        out.denom += c->agg.total_ns_sum;
    }
    (void)nranks;
    return out;
}

/**
 * @internal
 * @brief Emit cross-rank table rows, hottest child first.
 * @param node   Current MpiNode.
 * @param nranks Communicator size (mean denominator).
 * @param denom  "% of total" denominator.
 * @param depth  Tree depth (root skipped).
 * @param rows   Output rows.
 * @param fmt    Format (indentation style).
 */
inline void emit_mpi_rows(const MpiNode& node, int nranks, std::uint64_t denom, int depth,
                          std::vector<std::vector<std::string>>& rows, Format fmt)
{
    if (node.parent != nullptr) {
        const double mean = nranks ? static_cast<double>(node.agg.total_ns_sum) / static_cast<double>(nranks) : 0;
        const double maxs = static_cast<double>(node.agg.total_ns_max) / 1e9;
        const double mins =
            static_cast<double>(
                (node.agg.total_ns_min == std::numeric_limits<std::uint64_t>::max() ? 0 : node.agg.total_ns_min)) /
            1e9;
        const double imbal = mean > 0 ? ((static_cast<double>(node.agg.total_ns_max) / mean) - 1.0) * 100.0 : 0;
        const double pct = denom ? 100.0 * static_cast<double>(node.agg.total_ns_sum) / static_cast<double>(denom) : 0;
        std::ostringstream im;
        std::ostringstream pc;
        im << std::fixed << std::setprecision(1) << imbal;
        pc << std::fixed << std::setprecision(1) << pct;
        rows.push_back({indent_for(depth, fmt) + node.name, std::to_string(node.agg.ranks_present),
                        std::to_string(node.agg.calls_sum), humanize(mean / 1e9), humanize(mins), humanize(maxs),
                        im.str(), pc.str()});
    }
    std::vector<const MpiNode*> kids;
    kids.reserve(node.children.size());
    for (const auto& c : node.children) {
        kids.push_back(c.get());
    }
    std::ranges::sort(kids,
                      [](const MpiNode* a, const MpiNode* b) { return a->agg.total_ns_sum > b->agg.total_ns_sum; });

    for (const MpiNode* c : kids) {
        emit_mpi_rows(*c, nranks, denom, depth + 1, rows, fmt);
    }
}

/**
 * @internal
 * @brief Print a reduced cross-rank tree as a report.
 * @param os     Output stream.
 * @param red    Reduced tree.
 * @param nranks Communicator size.
 * @param fmt    Format.
 */
inline void print_mpi(std::ostream& os, const MpiReduce& red, int nranks, Format fmt)
{
    const std::vector<std::string> hdr = {"Region", "Ranks", "Calls", "Mean/rank", "Min", "Max", "Imbal%", "%Total"};
    std::vector<std::vector<std::string>> rows;
    emit_mpi_rows(*red.root, nranks, red.denom, 0, rows, fmt);
    if (fmt == Format::Markdown) {
        os << "## MPI timing report (" << nranks << " ranks)\n\nMean total across ranks: **"
           << humanize(static_cast<double>(red.denom) / 1e9 / (nranks ? nranks : 1))
           << "** per rank.  Imbal% = (max/mean - 1)*100.\n\n";
        render(os, hdr, rows, fmt);
    }
    else {
        os << "MPI timing report (" << nranks << " ranks)  Imbal% = (max/mean-1)*100\n";
        render(os, hdr, rows, fmt);
    }
}

} // namespace detail

#ifdef MISCIBILITY_INSTRUMENT_WITH_MPI
/**
 * @brief Gather every rank's timing tree to @p root and print a cross-rank report.
 * @param comm Communicator (default MPI_COMM_WORLD).
 * @param root Root rank that prints (default 0).
 * @param os   Output stream on the root rank (default std::cout).
 * @param fmt  Output format (default Plain).
 *
 * Collective: must be called by all ranks in @p comm. Each rank serializes its
 * (thread-aggregated) tree; the root reduces them and reports per-region mean/min/max
 * across ranks plus load imbalance. Requires `-DMISCIBILITY_INSTRUMENT_WITH_MPI` and linking MPI.
 *
 * @code
 * int main(int argc, char** argv) {
 *     MPI_Init(&argc, &argv);
 *     { miscibility::instrument::Timer<> t{"solve"}; run(); }
 *     miscibility::instrument::report_mpi();        // rank 0 prints the cross-rank table
 *     MPI_Finalize();
 * }
 * @endcode
 */
inline void report_mpi(MPI_Comm comm = MPI_COMM_WORLD, int root = 0, std::ostream& os = std::cout,
                       Format fmt = Format::Plain)
{
    int rank = 0;
    int nranks = 1;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &nranks);

    auto agg = detail::aggregate();
    const std::string blob = detail::serialize(*agg);
    const int len = static_cast<int>(blob.size());

    std::vector<int> lens(rank == root ? static_cast<std::size_t>(nranks) : 0);
    MPI_Gather(&len, 1, MPI_INT, lens.data(), 1, MPI_INT, root, comm);

    std::vector<int> displs;
    std::string all;
    if (rank == root) {
        displs.resize(static_cast<std::size_t>(nranks));
        int tot = 0;
        for (int i = 0; i < nranks; ++i) {
            displs[static_cast<std::size_t>(i)] = tot;
            tot += lens[static_cast<std::size_t>(i)];
        }
        all.resize(static_cast<std::size_t>(tot));
    }
    MPI_Gatherv(blob.data(), len, MPI_CHAR, all.data(), lens.data(), displs.data(), MPI_CHAR, root, comm);

    if (rank == root) {
        std::vector<std::string> blobs(static_cast<std::size_t>(nranks));
        for (int i = 0; i < nranks; ++i) {
            blobs[static_cast<std::size_t>(i)].assign(all.data() + displs[static_cast<std::size_t>(i)],
                                                      static_cast<std::size_t>(lens[static_cast<std::size_t>(i)]));
        }
        detail::print_mpi(os, detail::reduce_blobs(blobs, nranks), nranks, fmt);
    }
}
#endif // MISCIBILITY_INSTRUMENT_WITH_MPI

} // namespace miscibility::instrument