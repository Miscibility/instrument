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

    suite<"MPIReduce"> mpi_suite = [] {
        test("serialize round-trips through a single-rank reduce") = [] {
            mi::reset();
            {
                mi::Timer<> t{"solve"};
                busy(2.0);
                {
                    mi::Timer<> s{"spmv"};
                    busy(3.0);
                }
            }
            auto agg = mi::detail::aggregate();
            const std::string blob = mi::detail::serialize(*agg);
            auto red = mi::detail::reduce_blobs({blob}, 1);
            // top-level node "solve" present, with nested "spmv".
            expect(red.root->children.size() == 1_u);
            expect(red.root->children[0]->name == std::string("solve"));
            expect(red.root->children[0]->children.size() == 1_u);
            expect(red.root->children[0]->children[0]->name == std::string("spmv"));
            expect(red.denom > 0_u);
        };

        test("cross-rank reduce sums times and exposes imbalance") = [] {
            // Two ranks, same region, lopsided times: rank0=100ns, rank1=300ns.
            // Blob line format is: name \t calls \t total_ns
            const std::string r0 = "solve\t1\t100\n";
            const std::string r1 = "solve\t1\t300\n";
            auto red = mi::detail::reduce_blobs({r0, r1}, 2);
            expect(red.root->children.size() == 1_u);
            const auto& a = red.root->children[0]->agg;
            expect(a.ranks_present == 2_i);
            expect(a.calls_sum == 2_u);
            expect(a.total_ns_sum == 400_u);
            expect(a.total_ns_min == 100_u);
            expect(a.total_ns_max == 300_u);
            expect(red.denom == 400_u);
            // mean = 400/2 = 200; imbalance = (max/mean - 1) = 0.5 => 50%.
            const double mean = static_cast<double>(a.total_ns_sum) / 2.0;
            const double imbal = ((static_cast<double>(a.total_ns_max) / mean) - 1.0) * 100.0;
            expect(std::abs(imbal - 50.0) < 1e-9);
        };

        test("reduce unions regions that differ across ranks") = [] {
            const std::string r0 = "assemble\t1\t50\n";
            const std::string r1 = "exchange\t1\t70\n";
            auto red = mi::detail::reduce_blobs({r0, r1}, 2);
            expect(red.root->children.size() == 2_u) << "union of both ranks' regions";
            expect(red.denom == 120_u);
        };

        test("reduce builds nested paths across ranks") = [] {
            const std::string r0 = "solve\t1\t100\nsolve/spmv\t1\t60\n";
            const std::string r1 = "solve\t1\t200\nsolve/spmv\t1\t120\n";
            auto red = mi::detail::reduce_blobs({r0, r1}, 2);
            expect(red.root->children.size() == 1_u);
            const auto* solve = red.root->children[0].get();
            expect(solve->name == std::string("solve"));
            expect(solve->children.size() == 1_u);
            expect(solve->children[0]->name == std::string("spmv"));
            expect(solve->children[0]->agg.total_ns_sum == 180_u); // 60 + 120
        };
    };

    return 0;
}