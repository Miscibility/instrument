// test_matrix_copy.cpp -- unit tests for DenseMatrix::copy and
// LUFactorization::copy (in-place, no-allocation value copy into an
// already-sized destination). boost/ut. See specs/matrix-copy.md.
//
// Every test is expected to FAIL until tdd-3-implement fills in the stubs:
// both copy() methods currently throw std::runtime_error.

#include "instrument/matrix.hpp"

#include <boost/ut.hpp>
#include <cmath>
#include <cstddef>
#include <stdexcept>

namespace mi = miscibility::instrument;

namespace {

template<class T> bool close(T a, T b, T tol = T(1e-10))
{
    return std::abs(a - b) <= tol * (T(1) + std::abs(a) + std::abs(b));
}

} // namespace

int main()
{
    using namespace boost::ut;

    suite<"MatrixCopy"> matrix_copy_suite = [] {
        // ---- DenseMatrix::copy ------------------------------------------------

        test("dynamic <- dynamic, same shape: values copied, shape unchanged") = [] {
            mi::DenseMatrix<double> src{{1, 2, 3}, {4, 5, 6}}; // 2x3
            mi::DenseMatrix<double> dst(2, 3);
            dst.copy(src);
            expect(dst.rows() == 2_u);
            expect(dst.columns() == 3_u);
            for (std::size_t i = 0; i < 2; ++i) {
                for (std::size_t j = 0; j < 3; ++j) {
                    expect(dst(i, j) == src(i, j)) << "entry (" << i << "," << j << ")";
                }
            }
        };

        test("cross-extent dynamic <- static: values copied") = [] {
            mi::DenseMatrix<double, 2, 2> s{{1, 2}, {3, 4}};
            mi::DenseMatrix<double> d(2, 2);
            d.copy(s);
            expect(d(0, 0) == 1.0_d);
            expect(d(0, 1) == 2.0_d);
            expect(d(1, 0) == 3.0_d);
            expect(d(1, 1) == 4.0_d);
        };

        test("cross-extent static <- dynamic: values copied") = [] {
            mi::DenseMatrix<double> d{{5, 6}, {7, 8}};
            mi::DenseMatrix<double, 2, 2> s;
            s.copy(d);
            expect(s(0, 0) == 5.0_d);
            expect(s(0, 1) == 6.0_d);
            expect(s(1, 0) == 7.0_d);
            expect(s(1, 1) == 8.0_d);
        };

        test("column-major layout preserved element-for-element") = [] {
            mi::DenseMatrix<double> src{{1, 2, 3}, {4, 5, 6}}; // 2x3
            mi::DenseMatrix<double> dst(2, 3);
            dst.copy(src);
            for (std::size_t k = 0; k < src.size(); ++k) {
                expect(dst.data()[k] == src.data()[k]) << "column-major slot" << k;
            }
        };

        test("returns *this for chaining") = [] {
            mi::DenseMatrix<double> src{{1, 2}, {3, 4}};
            mi::DenseMatrix<double> dst(2, 2);
            dst.copy(src).scale(2.0);
            expect(dst(0, 0) == 2.0_d);
            expect(dst(1, 1) == 8.0_d);
        };

        test("copy is an independent value copy, not an alias") = [] {
            mi::DenseMatrix<double> src{{1, 2}, {3, 4}};
            mi::DenseMatrix<double> dst(2, 2);
            dst.copy(src);
            src(0, 0) = 99.0;
            expect(dst(0, 0) == 1.0_d) << "mutating source must not change destination";
        };

        test("no allocation: copy reuses the destination buffer (dynamic)") = [] {
            mi::DenseMatrix<double> src(2, 3, 4.0);
            mi::DenseMatrix<double> dst(2, 3);
            const double* before = dst.data();
            dst.copy(src);
            expect(dst.data() == before) << "copy must not reallocate";
        };

        test("shape mismatch throws invalid_argument and leaves destination untouched") = [] {
            // 2x3 and 3x2 have equal size() == 6 but different shapes.
            mi::DenseMatrix<double> src(2, 3, 5.0);
            mi::DenseMatrix<double> dst(3, 2, 1.0);
            expect(throws<std::invalid_argument>([&] { dst.copy(src); }));
            expect(dst.rows() == 3_u);
            expect(dst.columns() == 2_u);
            expect(dst(0, 0) == 1.0_d);
        };

        // ---- LUFactorization::copy --------------------------------------------

        test("happy path: copy refills a factorization; solves match the source") = [] {
            mi::DenseMatrix<double> a{{2, 1}, {1, 3}}; // luA
            mi::DenseMatrix<double> b{{4, 1}, {1, 2}}; // luB (distinct, same order)
            mi::LUFactorization<double> luA{a};
            mi::LUFactorization<double> luB{b};
            luA.copy(luB);
            for (const auto& rhs : {mi::Vector<double>{3, 5}, mi::Vector<double>{1, 1}, mi::Vector<double>{2, 7}}) {
                auto xa = luA.solve(rhs);
                auto xb = luB.solve(rhs);
                expect(close(xa[0], xb[0]));
                expect(close(xa[1], xb[1]));
            }
            expect(luA.order() == 2_u);
        };

        test("singular flag is carried by copy") = [] {
            mi::DenseMatrix<double> a{{2, 1}, {1, 3}}; // non-singular
            mi::DenseMatrix<double> b{{1, 2}, {2, 4}}; // singular, same order
            mi::LUFactorization<double> luA{a};
            mi::LUFactorization<double> luB{b};
            expect(!luA.singular());
            luA.copy(luB);
            expect(luA.singular());
            expect(throws<std::runtime_error>([&] { (void)luA.solve(mi::Vector<double>{1, 1}); }));
        };

        test("no allocation / state: order unchanged and solves stay correct after copy") = [] {
            mi::DenseMatrix<double> a{{2, 1}, {1, 3}};
            mi::DenseMatrix<double> b{{5, 2}, {2, 5}};
            mi::LUFactorization<double> luA{a};
            mi::LUFactorization<double> luB{b};
            const auto order_before = luA.order();
            luA.copy(luB);
            expect(luA.order() == order_before) << "order must be unchanged (no resize)";
            // [5 2; 2 5] x = [7; 7] -> x = {1, 1}.
            auto x = luA.solve(mi::Vector<double>{7, 7});
            expect(close(x[0], 1.0));
            expect(close(x[1], 1.0));
        };

        test("self-copy leaves solves unchanged") = [] {
            mi::DenseMatrix<double> a{{2, 1}, {1, 3}};
            mi::LUFactorization<double> lu{a};
            lu.copy(lu);
            auto x = lu.solve(mi::Vector<double>{3, 5});
            expect(close(x[0], 0.8));
            expect(close(x[1], 1.4));
        };

        test("order mismatch throws invalid_argument") = [] {
            mi::DenseMatrix<double> a{{2, 1}, {1, 3}};                  // order 2
            mi::DenseMatrix<double> b{{1, 0, 0}, {0, 1, 0}, {0, 0, 1}}; // order 3
            mi::LUFactorization<double> luA{a};
            mi::LUFactorization<double> luB{b};
            expect(throws<std::invalid_argument>([&] { luA.copy(luB); }));
        };
    };

    return 0;
}
