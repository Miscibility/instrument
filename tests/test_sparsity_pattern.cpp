// test_sparsity_pattern.cpp -- unit tests for MatrixEntry and SparsityPattern:
// validation, sorting into canonical row-major order, duplicate coalescing by
// summing, and the emitted CSR arrays. boost/ut.
//
// SparsityPattern's constructor currently throws std::runtime_error{"not
// implemented"}, so every test below fails until tdd-3-implement fills it in.

#include "instrument/sparsity_pattern.hpp"

#include <boost/ut.hpp>
#include <cstddef>
#include <stdexcept>
#include <vector>

namespace mi = miscibility::instrument;

namespace {

template<class V, class W> bool eq(const V& a, const W& b)
{
    if (a.size() != b.size()) {
        return false;
    }
    for (std::size_t i = 0; i < a.size(); ++i) {
        if (a[i] != static_cast<typename V::value_type>(b[i])) {
            return false;
        }
    }
    return true;
}

} // namespace

int main()
{
    using namespace boost::ut;

    suite<"SparsityPattern"> sp_suite = [] {
        test("happy path: dense 2x2 yields canonical CSR arrays") = [] {
            mi::SparsityPattern<double> p(2, 2, {{.row=0, .col=0, .value=1}, {.row=0, .col=1, .value=2}, {.row=1, .col=0, .value=3}, {.row=1, .col=1, .value=4}});
            expect(p.rows() == 2_u);
            expect(p.columns() == 2_u);
            expect(p.nonzeros() == 4_u);
            expect(eq(p.row_offsets(), std::vector<std::size_t>{0, 2, 4}));
            expect(eq(p.column_indices(), std::vector<std::size_t>{0, 1, 0, 1}));
            expect(eq(p.values(), std::vector<double>{1, 2, 3, 4}));
        };

        test("unsorted input produces the same canonical arrays as sorted") = [] {
            mi::SparsityPattern<double> p(2, 2, {{.row=1, .col=1, .value=4}, {.row=0, .col=0, .value=1}, {.row=1, .col=0, .value=3}, {.row=0, .col=1, .value=2}});
            expect(p.nonzeros() == 4_u);
            expect(eq(p.row_offsets(), std::vector<std::size_t>{0, 2, 4}));
            expect(eq(p.column_indices(), std::vector<std::size_t>{0, 1, 0, 1}));
            expect(eq(p.values(), std::vector<double>{1, 2, 3, 4}));
        };

        test("duplicate (row,col) entries coalesce by summing into one slot") = [] {
            mi::SparsityPattern<double> p(2, 2, {{.row=0, .col=0, .value=1.5}, {.row=0, .col=0, .value=2.5}, {.row=1, .col=1, .value=7}});
            expect(p.nonzeros() == 2_u);
            expect(eq(p.row_offsets(), std::vector<std::size_t>{0, 1, 2}));
            expect(eq(p.column_indices(), std::vector<std::size_t>{0, 1}));
            expect(eq(p.values(), std::vector<double>{4.0, 7.0}));
        };

        test("entries coalescing to zero are retained as a structural slot") = [] {
            mi::SparsityPattern<double> p(1, 1, {{.row=0, .col=0, .value=1}, {.row=0, .col=0, .value=-1}});
            expect(p.nonzeros() == 1_u);
            expect(eq(p.row_offsets(), std::vector<std::size_t>{0, 1}));
            expect(eq(p.column_indices(), std::vector<std::size_t>{0}));
            expect(eq(p.values(), std::vector<double>{0.0}));
        };

        test("rectangular: 3x5 with entries only in row 0; trailing rows empty") = [] {
            mi::SparsityPattern<double> p(3, 5, {{.row=0, .col=1, .value=9}, {.row=0, .col=4, .value=8}});
            expect(p.rows() == 3_u);
            expect(p.columns() == 5_u);
            expect(p.nonzeros() == 2_u);
            // row 0 has 2 entries; rows 1 and 2 are empty
            expect(eq(p.row_offsets(), std::vector<std::size_t>{0, 2, 2, 2}));
            expect(eq(p.column_indices(), std::vector<std::size_t>{1, 4}));
            expect(eq(p.values(), std::vector<double>{9, 8}));
        };

        test("empty middle row: consecutive row_offsets entries are equal there") = [] {
            // row 0: 1 entry, row 1: empty, row 2: 1 entry
            mi::SparsityPattern<double> p(3, 3, {{.row=0, .col=0, .value=1}, {.row=2, .col=2, .value=5}});
            expect(p.nonzeros() == 2_u);
            expect(eq(p.row_offsets(), std::vector<std::size_t>{0, 1, 1, 2}));
            expect(eq(p.column_indices(), std::vector<std::size_t>{0, 2}));
            expect(eq(p.values(), std::vector<double>{1, 5}));
        };

        test("empty pattern: no nonzeros, row_offsets all zero of length rows+1") = [] {
            mi::SparsityPattern<double> p(4, 4, {});
            expect(p.rows() == 4_u);
            expect(p.columns() == 4_u);
            expect(p.nonzeros() == 0_u);
            expect(eq(p.row_offsets(), std::vector<std::size_t>{0, 0, 0, 0, 0}));
            expect(p.column_indices().empty());
            expect(p.values().empty());
        };

        test("within each row, column indices are strictly increasing") = [] {
            mi::SparsityPattern<double> p(1, 4, {{.row=0, .col=3, .value=1}, {.row=0, .col=0, .value=2}, {.row=0, .col=2, .value=3}});
            expect(eq(p.column_indices(), std::vector<std::size_t>{0, 2, 3}));
            expect(eq(p.values(), std::vector<double>{2, 3, 1}));
        };

        test("out-of-range row throws out_of_range") = [] {
            expect(throws<std::out_of_range>([] { mi::SparsityPattern<double> p(2, 2, {{.row=2, .col=0, .value=1}}); }));
        };

        test("out-of-range column throws out_of_range") = [] {
            expect(throws<std::out_of_range>([] { mi::SparsityPattern<double> p(2, 2, {{.row=0, .col=2, .value=1}}); }));
        };

        test("float instantiation") = [] {
            mi::SparsityPattern<float> p(2, 2, {{.row=0, .col=0, .value=1.5F}, {.row=1, .col=1, .value=2.5F}});
            expect(p.nonzeros() == 2_u);
            expect(eq(p.values(), std::vector<float>{1.5F, 2.5F}));
        };
    };
}
