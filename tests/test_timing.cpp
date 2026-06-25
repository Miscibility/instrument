// test_timing.cpp -- unit tests for miscibility::instrument timing (boost/ut).

#include <boost/ut.hpp>
#include <cmath>
#include <functional>
// #include <thread>

#include "instrument/timing.hpp"

namespace mi = miscibility::instrument;

namespace {
/// Spin for approximately @p ms milliseconds of monotonic time.
void busy(double ms)
{
    const auto t0 = mi::clock::now();
    volatile double x = 0;
    while (std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(mi::clock::now() - t0).count() < ms) {
        x += std::sin(x + 1.0);
    }
}
} // namespace

int main()
{
    using namespace boost::ut;

    suite<"Timer"> timer_suite = [] {
        test("single region records one call with positive time") = [] {
            mi::reset();
            {
                mi::Timer<> t{"A"};
                busy(3.0);
            }
            auto s = mi::query("A");
            expect(s.has_value());
            expect(s->calls == 1_u);
            expect(s->total_seconds > 0.0);
            expect(s->pct_total == 100.0_d); // only region -> whole of measured total
        };

        test("repeated entries accumulate calls and average") = [] {
            mi::reset();
            for (int i = 0; i < 5; ++i) {
                mi::Timer<> t{"loop"};
                busy(1.0);
            }
            auto s = mi::query("loop");
            expect(s.has_value());
            expect(s->calls == 5_u);
            expect(s->avg_seconds > 0.0);
            expect(s->total_seconds >= s->avg_seconds); // 5 samples summed
        };

        test("nested regions form a parent/child tree") = [] {
            mi::reset();
            {
                mi::Timer<> outer{"solve"};
                busy(2.0);
                {
                    mi::Timer<> inner{"spmv"};
                    busy(4.0);
                }
            }
            auto outer = mi::query("solve");
            auto inner = mi::query("spmv"); // top-level lookup should miss
            auto nested = mi::query("solve/spmv");
            expect(outer.has_value());
            expect(not inner.has_value()) << "spmv is nested, not top-level";
            expect(nested.has_value());
            expect(nested->calls == 1_u);
            // inner is a strict subset of outer's inclusive time.
            expect(nested->total_seconds < outer->total_seconds);
            // child %parent is meaningful and below 100.
            expect(nested->pct_parent > 0.0);
            expect(nested->pct_parent < 100.0_d);
            // outer self time excludes the child.
            expect(outer->self_seconds < outer->total_seconds);
        };

        test("direct recursion folds into one node") = [] {
            mi::reset();
            std::function<void(int)> rec = [&](int n) {
                mi::Timer<> t{"rec"};
                busy(1.0);
                if (n > 0) {
                    rec(n - 1);
                }
            };
            rec(3); // entered 4 times, single outermost completion
            auto s = mi::query("rec");
            expect(s.has_value());
            expect(s->calls == 4_u) << "every entry is counted";
            // One outermost sample => avg == total (folded), not total/4.
            expect(std::abs(s->avg_seconds - s->total_seconds) < 1e-9) << "recursion folded to a single timed interval";
        };

        test("query of a missing path returns nullopt") = [] {
            mi::reset();
            {
                mi::Timer<> t{"present"};
            }
            expect(not mi::query("absent").has_value());
            expect(not mi::query("present/absent").has_value());
        };

        test("timing level gates compilation") = [] {
            mi::reset();
            {
                mi::Timer<1> on{"active"};    // 1 <= TIMING_LEVEL(2): active
                mi::Timer<3> off{"inactive"}; // 3 >  TIMING_LEVEL(2): no-op
                busy(1.0);
            }
            expect(mi::query("active").has_value());
            expect(not mi::query("inactive").has_value()) << "Timer<3> must record nothing when TIMING_LEVEL==2";
            // Inactive timer is a truly empty object.
            expect(sizeof(mi::Timer<3>) == 1_u);
        };
    };

    suite<"ManualSpan"> span_suite = [] {
        test("start/stop times a region like a scoped Timer") = [] {
            mi::reset();
            mi::start("phase");
            busy(3.0);
            mi::stop("phase");
            auto s = mi::query("phase");
            expect(s.has_value());
            expect(s->calls == 1_u);
            expect(s->total_seconds > 0.0);
        };

        test("manual spans nest with scoped timers") = [] {
            mi::reset();
            {
                mi::Timer<> t{"region"};
                mi::start("inner_span");
                busy(2.0);
                mi::stop("inner_span");
            }
            auto outer = mi::query("region");
            auto nested = mi::query("region/inner_span");
            expect(outer.has_value());
            expect(nested.has_value());
            expect(nested->calls == 1_u);
        };

        test("inactive level makes start/stop a no-op") = [] {
            mi::reset();
            mi::start<3>("vanished"); // 3 > TIMING_LEVEL(2)
            busy(1.0);
            mi::stop<3>("vanished");
            expect(not mi::query("vanished").has_value());
        };
    };

    suite<"ManualSpanErrors"> span_error_suite = [] {
        test("stop with no open span warns and is a no-op") = [] {
            mi::reset();
            mi::stop("nothing"); // manual stack empty -> warning, returns
            expect(not mi::query("nothing").has_value());
        };

        test("stop with a mismatched name still closes the span") = [] {
            mi::reset();
            mi::start("opened");
            busy(1.0);
            mi::stop("something_else"); // name mismatch -> warning, closes anyway
            auto s = mi::query("opened");
            expect(s.has_value());
            expect(s->calls == 1_u);
        };
    };

    suite<"QueryEdges"> query_edge_suite = [] {
        test("query of an empty path returns nullopt") = [] {
            mi::reset();
            {
                mi::Timer<> t{"region"};
            }
            expect(not mi::query("").has_value()) << "empty path resolves to the root";
            expect(not mi::query("/").has_value());
        };
    };

    suite<"Detail"> detail_suite = [] {
        test("humanize scales across seconds/ms/us/ns") = [] {
            using mi::detail::humanize;
            expect(humanize(2.5).find(" s") != std::string::npos);    // >= 1 s
            expect(humanize(2.5e-3).find("ms") != std::string::npos); // milliseconds
            expect(humanize(2.5e-6).find("us") != std::string::npos); // microseconds
            expect(humanize(2.5e-9).find("ns") != std::string::npos); // nanoseconds
        };

        test("indent_for emits nested markdown and plain indentation") = [] {
            using mi::detail::indent_for;
            using mi::Format;
            expect(indent_for(1, Format::Markdown).empty()) << "top level has no indent";
            expect(indent_for(2, Format::Markdown).find("&nbsp;") != std::string::npos);
            expect(indent_for(2, Format::Plain).find("  ") != std::string::npos);
        };
    };

    // MPI tests live in test_timing_mpi.cpp (guarded by MISCIBILITY_INSTRUMENT_WITH_MPI).

    // The unit suite must run before main() returns so coverage counters are flushed.
    return ::boost::ut::cfg<>.run();
}