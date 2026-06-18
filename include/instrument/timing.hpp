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

#ifndef MISCIBILITY_INSTRUMENT_TIMING_LEVEL
#define MISCIBILITY_INSTRUMENT_TIMING_LEVEL 2 // default: a moderate level that avoids timing inside loops
#endif

namespace miscibility::instrument {

/// The clock used for all timing measurements: a monotonic steady clock.
using clock = std::chrono::steady_clock;

/// A snapshot of the timing measured for one region.
///
/// "Total" time is inclusive (the region and everything called within it);
/// "self" time is exclusive (total minus the time attributed to child regions).
/// Returned by :cpp:`query` and used internally to build the report tables.
struct Stats {
    /// The region's name (its last path segment, not the full path).
    std::string name;
    /// Number of times the region was entered.
    std::uint64_t calls = 0;
    /// Inclusive wall time spent in the region, in seconds.
    double total_seconds = 0;
    /// Exclusive wall time (total minus children), in seconds.
    double self_seconds = 0;
    /// Mean inclusive time per top-level entry, in seconds.
    double avg_seconds = 0;
    /// Inclusive time as a percentage of the whole measured program.
    double pct_total = 0;
    /// Inclusive time as a percentage of the parent region.
    double pct_parent = 0;
};

/// Output style for a timing report.
enum class Format : std::uint8_t {
    Plain,    ///< A box-drawn ASCII table for console output.
    Markdown, ///< A GitHub-flavored Markdown table for files and docs.
};

namespace detail {

inline bool eq(const std::string& a, std::string_view b) { return a.size() == b.size() && std::string_view(a) == b; }

struct Node {
    std::string name;
    Node* parent = nullptr;
    std::vector<std::unique_ptr<Node>> children;

    std::uint64_t calls = 0;
    std::uint64_t samples = 0;
    std::uint64_t total_ns = 0;
    int active = 0;

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

struct Frame {
    Node* node = nullptr;
    clock::time_point start;
    bool descended = false;
};

struct Global {
    std::mutex mtx;
    std::vector<std::unique_ptr<Node>> roots;
    clock::time_point t0 = clock::now();

    static Global& instance()
    {
        static Global g;
        return g;
    }

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

struct ThreadReg {
    Node* root;
    Node* current;
    std::vector<Frame> manual_stack;

    ThreadReg() : root(Global::instance().new_root()), current(root) {}

    static ThreadReg& local()
    {
        thread_local ThreadReg t;
        return t;
    }
};

inline Frame begin_region(ThreadReg& reg, std::string_view name)
{
    Node* cur = reg.current;
    Node* node = nullptr;
    bool descended = false;
    if (cur != reg.root && eq(cur->name, name)) {
        node = cur;
        descended = false;
    }
    else {
        node = cur->child(name);
        reg.current = node;
        descended = true;
    }
    ++node->calls;
    ++node->active;
    return Frame{.node = node, .start = clock::now(), .descended = descended};
}

inline void end_region(ThreadReg& reg, const Frame& f)
{
    const auto end = clock::now();
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

inline void merge(Node& dst, const Node& src)
{
    dst.calls += src.calls;
    dst.samples += src.samples;
    dst.total_ns += src.total_ns;
    for (const auto& c : src.children) {
        merge(*dst.child(c->name), *c);
    }
}

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
        s.assign(long{2} * (depth - 1), ' ');
    }
    return s + "- ";
}

} // namespace detail

/// Computes the derived :cpp:`Stats` (self time, averages, percentages) for one node.
///
/// :param n: The region node to summarize.
/// :param denom_ns: Total measured nanoseconds used as the denominator for ``pct_total``.
/// :param parent: The node's parent, or ``nullptr`` at the top level, used for ``pct_parent``.
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

/// Looks up the aggregated stats for a single region by its path.
///
/// The path is a ``/``-separated sequence of region names from a root region
/// down to the region of interest, for example ``"solve/assemble"``. Results
/// are aggregated across all threads.
///
/// .. code-block:: cpp
///
///     if (auto s = query("solve/assemble")) {
///         std::cout << s->calls << " calls, " << s->total_seconds << " s\n";
///     }
///
/// :param path: Slash-separated region path to resolve.
/// :returns: The region's stats, or ``std::nullopt`` if no such region was recorded.
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

/// A scoped guard that times the region between its construction and destruction.
///
/// Construct a ``Timer`` at the start of the scope you want to measure; it
/// records the elapsed time when it goes out of scope. Nested timers form a
/// call tree, so a region's time is attributed beneath its enclosing region.
///
/// ``Level`` gates the timer at compile time against
/// ``MISCIBILITY_INSTRUMENT_TIMING_LEVEL``: when ``Level`` exceeds the configured
/// level the timer compiles to nothing, so fine-grained instrumentation can be
/// left in place and switched off without runtime cost. Lower levels are
/// coarser; higher levels are reserved for hot, inner-loop regions.
///
/// .. code-block:: cpp
///
///     void solve() {
///         Timer t{"solve"};
///         // ... work timed as the "solve" region ...
///     }
///
/// The timer is neither copyable nor movable; it is meant to live on the stack.
///
/// :tparam Level: Verbosity level at which this timer is active.
template<int Level = 1> class Timer {
    static constexpr bool On = (Level <= MISCIBILITY_INSTRUMENT_TIMING_LEVEL);

    struct State {
        detail::ThreadReg* reg = nullptr;
        detail::Frame frame;
    };
    struct Empty {};
    [[no_unique_address]] std::conditional_t<On, State, Empty> s_;

public:
    /// Opens a timed region named ``name`` (a no-op when gated off by ``Level``).
    explicit Timer(std::string_view name)
    {
        if constexpr (On) {
            s_.reg = &detail::ThreadReg::local();
            s_.frame = detail::begin_region(*s_.reg, name);
        }
    }

    ~Timer()
    {
        if constexpr (On) {
            detail::end_region(*s_.reg, s_.frame);
        }
    }

    Timer(const Timer&) = delete;
    Timer& operator=(const Timer&) = delete;
    Timer(Timer&&) = delete;
    Timer& operator=(Timer&&) = delete;
};

/// Creates a :cpp:`Timer` for the current scope.
///
/// A convenience factory so the region can be opened without naming the
/// ``Level`` twice. Bind the result to a local variable so it lives for the
/// scope you want to measure:
///
/// .. code-block:: cpp
///
///     auto t = scoped_timer("assemble");
///
/// :param name: Name of the region.
/// :tparam Level: Verbosity level at which the timer is active.
template<int Level = 1> [[nodiscard]] inline Timer<Level> scoped_timer(std::string_view name)
{
    return Timer<Level>{name};
}

/// Opens a timed region that is closed by a later :cpp:`stop`.
///
/// Use this when the region does not align with a C++ scope. Calls must nest:
/// the most recently started region is the one a bare :cpp:`stop` closes. When
/// gated off by ``Level`` this does nothing.
///
/// :param name: Name of the region to open.
/// :tparam Level: Verbosity level at which the region is recorded.
template<int Level = 1> inline void start(std::string_view name)
{
    if constexpr (Level <= MISCIBILITY_INSTRUMENT_TIMING_LEVEL) {
        auto& reg = detail::ThreadReg::local();
        reg.manual_stack.push_back(detail::begin_region(reg, name));
    }
}

/// Closes the most recently started region (see :cpp:`start`).
///
/// If ``expected`` is given, it is checked against the region actually being
/// closed and a warning is written to ``std::cerr`` on a mismatch, which
/// usually signals mis-nested ``start``/``stop`` pairs. Calling ``stop`` with
/// no open region also warns rather than throwing.
///
/// :param expected: Optional name the closing region is expected to have.
/// :tparam Level: Must match the ``Level`` used for the paired :cpp:`start`.
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

namespace detail {

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

/// Writes the full timing report to ``os``.
///
/// The report is a table of every recorded region, sorted by total time and
/// indented to show the call hierarchy, aggregated across all threads.
///
/// :param os: Stream to write to.
/// :param fmt: Table style (plain ASCII or Markdown).
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

/// Writes the timing report to a file, defaulting to Markdown.
///
/// :param path: Destination file; overwritten if it exists.
/// :param fmt: Table style (Markdown by default).
inline void report(const std::filesystem::path& path, Format fmt = Format::Markdown)
{
    std::ofstream f(path);
    report(f, fmt);
}

/// Writes the timing report to standard output.
///
/// :param fmt: Table style (plain ASCII by default).
inline void report(Format fmt = Format::Plain) { report(std::cout, fmt); }

/// Clears all recorded timing data and restarts the wall-clock baseline.
///
/// Use this to time a fresh phase after discarding earlier measurements, for
/// example to exclude a warm-up pass from the reported numbers.
inline void reset()
{
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

namespace detail {

inline void append_u64(std::string& out, std::uint64_t v) { out += std::to_string(v); }

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

inline std::string serialize(const Node& root)
{
    std::string path;
    std::string out;
    serialize_rec(root, path, out);
    return out;
}

struct RankAgg {
    std::uint64_t calls_sum = 0;
    std::uint64_t total_ns_sum = 0;
    std::uint64_t total_ns_min = UINT64_MAX;
    std::uint64_t total_ns_max = 0;
    int ranks_present = 0;
};

struct MpiNode {
    std::string name;
    MpiNode* parent = nullptr;
    std::vector<std::unique_ptr<MpiNode>> children;
    RankAgg agg;

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

struct MpiReduce {
    std::unique_ptr<MpiNode> root;
    std::uint64_t denom = 0;
};

inline std::uint64_t parse_u64(std::string_view sv)
{
    std::uint64_t v = 0;
    std::from_chars(sv.data(), sv.data() + sv.size(), v);
    return v;
}

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

/// Gathers per-rank timings across an MPI communicator and reports them on one rank.
///
/// Every rank must call this collectively. Each rank's region tree is sent to
/// ``root``, which combines them and prints a table showing, per region, the
/// mean time per rank, the min and max across ranks, and a load-imbalance
/// percentage ``(max/mean - 1) * 100``. Only ``root`` writes to ``os``.
///
/// Available only when the library is built with
/// ``MISCIBILITY_INSTRUMENT_WITH_MPI`` defined.
///
/// :param comm: Communicator over which timings are gathered.
/// :param root: Rank that aggregates and prints the report.
/// :param os: Stream the root rank writes to.
/// :param fmt: Table style (plain ASCII or Markdown).
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