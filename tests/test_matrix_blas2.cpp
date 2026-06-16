// test_matrix_blas2.cpp -- unit tests for the DenseMatrix matrix-vector product
// (CBLAS gemv): multiply, multiply_into, operator*. boost/ut.

#include "instrument/matrix.hpp"

#include <boost/ut.hpp>
#include <cmath>

namespace mi = miscibility::instrument;

namespace {

template<class T> bool close(T a, T b, T tol = T(1e-12))
{
    return std::abs(a - b) <= tol * (T(1) + std::abs(a) + std::abs(b));
}

} // namespace

int main()
{
    using namespace boost::ut;

    suite<"DenseMatrixMatVec"> matvec_suite = [] {
        test("multiply: A*x matches a hand loop and operator*") = [] {
            mi::DenseMatrix<double> a{{1, 2}, {3, 4}};
            mi::Vector<double> x{5, 6};
            auto y = a.multiply(x);
            expect(y.size() == 2_u);
            expect(close(y[0], 17.0)); // 1*5 + 2*6
            expect(close(y[1], 39.0)); // 3*5 + 4*6
            auto y2 = a * x;
            expect(close(y2[0], 17.0));
            expect(close(y2[1], 39.0));
        };

        test("transposed: A^T*x") = [] {
            mi::DenseMatrix<double> a{{1, 2}, {3, 4}};
            mi::Vector<double> x{5, 6};
            auto y = a.multiply(x, mi::Transpose::Transposed);
            expect(y.size() == 2_u);
            expect(close(y[0], 23.0)); // 1*5 + 3*6
            expect(close(y[1], 34.0)); // 2*5 + 4*6
        };

        test("multiply_into with alpha/beta is the scale-add variant") = [] {
            mi::DenseMatrix<double> a{{1, 2}, {3, 4}};
            mi::Vector<double> x{5, 6};
            mi::Vector<double> y{1, 1};
            a.multiply_into(x, y, 2.0, 3.0); // 2*A*x + 3*y
            expect(close(y[0], 37.0));       // 37
            expect(close(y[1], 81.0));       // 81
        };

        test("non-square: 2x3 times length-3 gives length 2; transposed gives length 3") = [] {
            mi::DenseMatrix<double> a{{1, 2, 3}, {4, 5, 6}};
            mi::Vector<double> x{1, 1, 1};
            auto y = a.multiply(x);
            expect(y.size() == 2_u);
            expect(close(y[0], 6.0));  // 1+2+3
            expect(close(y[1], 15.0)); // 4+5+6

            mi::Vector<double> xt{1, 1};
            auto yt = a.multiply(xt, mi::Transpose::Transposed);
            expect(yt.size() == 3_u);
            expect(close(yt[0], 5.0)); // 1+4
            expect(close(yt[1], 7.0)); // 2+5
            expect(close(yt[2], 9.0)); // 3+6
        };

        test("length mismatch throws invalid_argument") = [] {
            mi::DenseMatrix<double> a{{1, 2}, {3, 4}};
            mi::Vector<double> bad{1, 2, 3};
            expect(throws<std::invalid_argument>([&] { (void)a.multiply(bad); }));
            mi::Vector<double> x{5, 6};
            mi::Vector<double> wrong_y{0, 0, 0};
            expect(throws<std::invalid_argument>([&] { a.multiply_into(x, wrong_y); }));
        };

        test("float and double both work") = [] {
            mi::DenseMatrix<float> af{{1, 2}, {3, 4}};
            mi::Vector<float> xf{5, 6};
            auto yf = af.multiply(xf);
            expect(close(yf[0], 17.0F));
            expect(close(yf[1], 39.0F));

            mi::DenseMatrix<double> ad{{1, 2}, {3, 4}};
            mi::Vector<double> xd{5, 6};
            auto yd = ad.multiply(xd);
            expect(close(yd[0], 17.0));
            expect(close(yd[1], 39.0));
        };
    };
}
