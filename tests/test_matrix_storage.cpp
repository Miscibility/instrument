// test_matrix_storage.cpp -- unit tests for the DenseMatrix container's storage,
// layout, element access and elementwise surface (boost/ut).
// See specs/matrix-storage.md.
//
// Every test is expected to FAIL until tdd-3-implement fills in the stubs: every
// DenseMatrix constructor currently throws std::runtime_error, so each test fails
// at the construction line before it can assert real behaviour.

#include <boost/ut.hpp>

#include <array>
#include <cstddef>
#include <cstdint>
#include <stdexcept>

#include "instrument/matrix.hpp"

namespace mi = miscibility::instrument;

namespace {

// SIMD lanes for the build's static-dispatch target (matches the Vector tests).
template<class T> std::size_t lane_count()
{
    return mi::detail::hn::Lanes(mi::detail::hn::ScalableTag<T>{});
}

// The backing Vector's storage invariant: aligned buffer, capacity a whole
// multiple of the lane count, capacity >= size, every pad slot zero.
template<class V> void check_aligned_and_padded(const V& v)
{
    using namespace boost::ut;
    using T = typename V::value_type;
    expect((reinterpret_cast<std::uintptr_t>(v.data()) % mi::detail::alignment) == 0_u);
    const std::size_t lanes = lane_count<T>();
    expect((v.capacity() % lanes) == 0u) << "single-loop, no remainder";
    expect(v.capacity() >= v.size());
    for (std::size_t i = v.size(); i < v.capacity(); ++i) {
        expect(v.data()[i] == T{}) << "pad slot" << i << "must be zero";
    }
}

} // namespace

int main()
{
    using namespace boost::ut;

    suite<"DenseMatrixStorage"> storage_suite = [] {
        test("static matrix default-constructs to zeros with the static shape") = [] {
            mi::DenseMatrix<double, 2, 3> m;
            expect(m.rows() == 2_u);
            expect(m.columns() == 3_u);
            expect(m.size() == 6_u);
            for (std::size_t j = 0; j < m.columns(); ++j) {
                for (std::size_t i = 0; i < m.rows(); ++i) {
                    expect(m(i, j) == 0.0_d);
                }
            }
        };

        test("dynamic matrix takes a runtime shape and starts zero") = [] {
            mi::DenseMatrix<double> m(2, 3);
            expect(m.rows() == 2_u);
            expect(m.columns() == 3_u);
            expect(m.size() == 6_u);
            for (std::size_t j = 0; j < m.columns(); ++j) {
                for (std::size_t i = 0; i < m.rows(); ++i) {
                    expect(m(i, j) == 0.0_d);
                }
            }
        };

        test("dynamic matrix (rows, cols, value) fills every element") = [] {
            mi::DenseMatrix<double> m(2, 3, 7.0);
            expect(m.size() == 6_u);
            for (std::size_t j = 0; j < m.columns(); ++j) {
                for (std::size_t i = 0; i < m.rows(); ++i) {
                    expect(m(i, j) == 7.0_d);
                }
            }
        };

        test("column-major layout is pinned: indexing, raw buffer and leading dimension") = [] {
            mi::DenseMatrix<double, 2, 3> m{{1, 2, 3}, {4, 5, 6}};
            // Row-major reading order in the initializer; column-major in storage.
            expect(m(0, 0) == 1.0_d);
            expect(m(0, 2) == 3.0_d);
            expect(m(1, 0) == 4.0_d);
            expect(m(1, 2) == 6.0_d);
            // Raw buffer is column-major: {col0, col1, col2} = {1,4, 2,5, 3,6}.
            const std::array<double, 6> expected{1, 4, 2, 5, 3, 6};
            for (std::size_t k = 0; k < expected.size(); ++k) {
                expect(m.data()[k] == expected[k]) << "column-major slot" << k;
            }
            // Leading dimension is the row count.
            expect(m.rows() == 2_u);
        };

        test("row-major nested initializer builds the same matrix, static and dynamic") = [] {
            mi::DenseMatrix<double, 2, 3> s{{1, 2, 3}, {4, 5, 6}};
            mi::DenseMatrix<double> d{{1, 2, 3}, {4, 5, 6}};
            expect(d.rows() == 2_u);
            expect(d.columns() == 3_u);
            for (std::size_t j = 0; j < 3; ++j) {
                for (std::size_t i = 0; i < 2; ++i) {
                    expect(s(i, j) == d(i, j)) << "entry (" << i << "," << j << ")";
                }
            }
        };

        test("at() is bounds-checked, operator() reaches every in-range entry") = [] {
            mi::DenseMatrix<double, 2, 3> m{{1, 2, 3}, {4, 5, 6}};
            expect(m.at(0, 0) == 1.0_d);
            expect(m.at(1, 2) == 6.0_d);
            expect(throws<std::out_of_range>([&] { (void)m.at(2, 0); }));
            expect(throws<std::out_of_range>([&] { (void)m.at(0, 3); }));
            expect(throws<std::out_of_range>([&] { (void)m.at(2, 3); }));
        };

        test("ragged initializer throws invalid_argument") = [] {
            expect(throws<std::invalid_argument>(
                [] { mi::DenseMatrix<double> m{{1, 2}, {3}}; }));
        };

        test("static shape mismatch in the initializer throws invalid_argument") = [] {
            expect(throws<std::invalid_argument>(
                [] { mi::DenseMatrix<double, 2, 2> m{{1, 2, 3}, {4, 5, 6}}; }));
        };

        test("as_vector exposes a length-size, aligned, zero-padded backing Vector") = [] {
            mi::DenseMatrix<double, 2, 3> m{{1, 2, 3}, {4, 5, 6}};
            expect(m.as_vector().size() == m.size());
            check_aligned_and_padded(m.as_vector());
        };

        test("elementwise scalar scale and matrix add match a per-element reference") = [] {
            mi::DenseMatrix<double, 2, 2> m{{1, 2}, {3, 4}};
            mi::DenseMatrix<double, 2, 2> n{{10, 20}, {30, 40}};
            m *= 2.0;     // {{2,4},{6,8}}
            m += n;       // {{12,24},{36,48}}
            const std::array<std::array<double, 2>, 2> expected{{{12, 24}, {36, 48}}};
            for (std::size_t i = 0; i < 2; ++i) {
                for (std::size_t j = 0; j < 2; ++j) {
                    expect(m(i, j) == expected[i][j]);
                }
            }
            check_aligned_and_padded(m.as_vector());
        };

        test("elementwise Hadamard product matches a per-element reference") = [] {
            mi::DenseMatrix<double, 2, 2> m{{1, 2}, {3, 4}};
            mi::DenseMatrix<double, 2, 2> n{{5, 6}, {7, 8}};
            m.elementwise_product(n); // {{5,12},{21,32}}
            expect(m(0, 0) == 5.0_d);
            expect(m(0, 1) == 12.0_d);
            expect(m(1, 0) == 21.0_d);
            expect(m(1, 1) == 32.0_d);
            check_aligned_and_padded(m.as_vector());
        };

        test("shape mismatch on += throws invalid_argument (not merely a size check)") = [] {
            // 2x3 and 3x2 have equal size() == 6 but different shapes.
            mi::DenseMatrix<double> a(2, 3, 1.0);
            mi::DenseMatrix<double> b(3, 2, 1.0);
            expect(throws<std::invalid_argument>([&] { a += b; }));
            expect(throws<std::invalid_argument>([&] { a.elementwise_product(b); }));
            expect(throws<std::invalid_argument>([&] { a.add_scaled(2.0, b); }));
        };

        test("swap exchanges two dynamic matrices of different shapes") = [] {
            mi::DenseMatrix<double> a(2, 3, 1.0);
            mi::DenseMatrix<double> b(3, 2, 7.0);
            swap(a, b);
            expect(a.rows() == 3_u);
            expect(a.columns() == 2_u);
            expect(b.rows() == 2_u);
            expect(b.columns() == 3_u);
            expect(a(0, 0) == 7.0_d);
            expect(b(0, 0) == 1.0_d);
        };

        // Compile-time contract (not an executed test): a mixed static/dynamic
        // shape must fail to compile via the static_assert in DenseMatrix.
        //   mi::DenseMatrix<double, 3, mi::dynamic> bad; // <- ill-formed
    };
}
