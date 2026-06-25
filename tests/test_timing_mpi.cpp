// test_timing_mpi.cpp -- MPI unit tests for miscibility::instrument timing (boost/ut).
//
// Every test here is guarded by MISCIBILITY_INSTRUMENT_WITH_MPI: nothing of
// substance compiles unless the library was built with MPI support. The CMake
// test harness launches this executable on several ranks (see tests/CMakeLists.txt).

#ifdef MISCIBILITY_INSTRUMENT_WITH_MPI

#include "instrument/timing.hpp"

#include <boost/ut.hpp>
#include <cmath>
#include <mpi.h>
#include <sstream>
#include <string>

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

int world_rank()
{
    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    return rank;
}

int world_size()
{
    int n = 1;
    MPI_Comm_size(MPI_COMM_WORLD, &n);
    return n;
}
} // namespace

int main(int argc, char** argv)
{
    using namespace boost::ut;

    MPI_Init(&argc, &argv);

    // Pure reduce/serialize logic: independent of the number of ranks, these
    // verify the building blocks report_mpi() relies on.
    suite<"MPIReduce"> reduce_suite = [] {
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

        test("reduce tolerates malformed and unterminated blob lines") = [] {
            // No trailing newline on the last line; a blank line; a short line
            // with too few fields. All must be parsed/skipped without error.
            const std::string blob = "solve\t1\t100\n\nbad_line_two_fields\t1\nfinal\t2\t50";
            auto red = mi::detail::reduce_blobs({blob}, 1);
            expect(red.root->children.size() == 2_u) << "only the two well-formed lines survive";
            expect(red.denom == 150_u); // 100 + 50
        };
    };

    // Collective report_mpi() over the real communicator. Every rank must enter
    // these tests so the MPI_Gather/MPI_Gatherv calls match up.
    suite<"ReportMpi"> report_suite = [] {
        test("report_mpi gathers timings and only root writes") = [] {
            mi::reset();
            {
                mi::Timer<> t{"solve"};
                // Lopsided work so the imbalance column is non-trivial.
                busy(2.0 * (world_rank() + 1));
                {
                    mi::Timer<> s{"spmv"};
                    busy(1.0);
                }
            }
            std::ostringstream os;
            mi::report_mpi(MPI_COMM_WORLD, 0, os, mi::Format::Plain);
            if (world_rank() == 0) {
                const std::string out = os.str();
                expect(not out.empty());
                expect(out.find("solve") != std::string::npos);
                expect(out.find("spmv") != std::string::npos);
                expect(out.find("MPI timing report") != std::string::npos);
            }
            else {
                expect(os.str().empty()) << "only the root rank writes the report";
            }
        };

        test("report_mpi renders a Markdown report on root") = [] {
            mi::reset();
            {
                mi::Timer<> t{"assemble"};
                busy(1.0 * (world_rank() + 1));
            }
            std::ostringstream os;
            mi::report_mpi(MPI_COMM_WORLD, 0, os, mi::Format::Markdown);
            if (world_rank() == 0) {
                const std::string out = os.str();
                expect(out.find("## MPI timing report") != std::string::npos);
                expect(out.find('|') != std::string::npos) << "markdown table";
            }
            else {
                expect(os.str().empty());
            }
        };

        test("report_mpi sorts sibling regions on the root") = [] {
            mi::reset();
            {
                mi::Timer<> a{"alpha"};
                busy(1.0);
            }
            {
                mi::Timer<> b{"bravo"};
                busy(2.0);
            }
            std::ostringstream os;
            mi::report_mpi(MPI_COMM_WORLD, 0, os, mi::Format::Plain);
            if (world_rank() == 0) {
                const std::string out = os.str();
                // Two top-level siblings force the rank-sum comparator to run.
                expect(out.find("alpha") != std::string::npos);
                expect(out.find("bravo") != std::string::npos);
            }
        };

        test("report_mpi accepts a non-zero root rank") = [] {
            mi::reset();
            {
                mi::Timer<> t{"exchange"};
                busy(1.0);
            }
            const int root = world_size() - 1; // last rank aggregates
            std::ostringstream os;
            mi::report_mpi(MPI_COMM_WORLD, root, os, mi::Format::Plain);
            if (world_rank() == root) {
                expect(not os.str().empty());
            }
            else {
                expect(os.str().empty());
            }
        };
    };

    // Run the suites before MPI_Finalize so gcov counters are flushed at exit.
    const auto result = ::boost::ut::cfg<>.run();
    MPI_Finalize();
    return result;
}

#else // MISCIBILITY_INSTRUMENT_WITH_MPI

// Built without MPI: nothing to test, but the translation unit must still link.
int main() { return 0; }

#endif // MISCIBILITY_INSTRUMENT_WITH_MPI
