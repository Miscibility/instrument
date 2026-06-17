// test_zero_matrix.cpp -- unit tests for the MatrixOperator concept and the
// ZeroMatrix type (the sizes-only all-zeros matrix). boost/ut.
//
// ZeroMatrix's members currently throw std::runtime_error{"not implemented"},
// so every runtime test below fails until tdd-3-implement fills them in. The
// compile-time static_asserts, by contrast, pin down the *shape* of the API and
// must hold for the suite to compile at all.

#include "instrument/matrix.hpp"
#include "instrument/vector.hpp"
#include "instrument/zero_matrix.hpp"

#include <boost/ut.hpp>
#include <stdexcept>

namespace mi = miscibility::instrument;

// -- concept shape (compile-time) -------------------------------------------

// DenseMatrix and ZeroMatrix both model the MatrixOperator seam ...
static_assert(mi::MatrixOperator<mi::DenseMatrix<double>, double>);
static_assert(mi::MatrixOperator<mi::ZeroMatrix<double>, double>);
static_assert(mi::MatrixOperator<mi::ZeroMatrix<float>, float>);
// ... but a plain Vector (no rows()/columns()/multiply_into) does not.
static_assert(!mi::MatrixOperator<mi::Vector<double>, double>);

namespace {

template<class T> bool close(T a, T b, T tol = T(1e-12))
{
    return std::abs(a - b) <= tol * (T(1) + std::abs(a) + std::abs(b));
}

} // namespace

int main()
{
    using namespace boost::ut;

    suite<"ZeroMatrix"> zero_suite = [] {
        test("multiply returns an all-zero vector of length rows()") = [] {
            mi::ZeroMatrix<double> z(3, 4);
            mi::Vector<double> x{1, 2, 3, 4};
            auto y = z.multiply(x);
            expect(y.size() == 3_u);
            expect(close(y[0], 0.0));
            expect(close(y[1], 0.0));
            expect(close(y[2], 0.0));
        };

        test("operator* matches multiply") = [] {
            mi::ZeroMatrix<double> z(2, 2);
            mi::Vector<double> x{5, 6};
            auto y = z * x;
            expect(y.size() == 2_u);
            expect(close(y[0], 0.0));
            expect(close(y[1], 0.0));
        };

        test("multiply_into with beta == 0 zero-fills y regardless of x/alpha") = [] {
            mi::ZeroMatrix<double> z(3, 3);
            mi::Vector<double> x{1, 1, 1};
            mi::Vector<double> y{9, 9, 9};
            z.multiply_into(x, y, 7.0, 0.0);
            expect(close(y[0], 0.0));
            expect(close(y[1], 0.0));
            expect(close(y[2], 0.0));
        };

        test("multiply_into with beta == 2 scales y only (y <- beta*y)") = [] {
            mi::ZeroMatrix<double> z(3, 3);
            mi::Vector<double> x{1, 1, 1};
            mi::Vector<double> y{1, 2, 3};
            z.multiply_into(x, y, 5.0, 2.0); // alpha is irrelevant
            expect(close(y[0], 2.0));
            expect(close(y[1], 4.0));
            expect(close(y[2], 6.0));
        };

        test("default alpha/beta zero-fills y") = [] {
            mi::ZeroMatrix<double> z(2, 3);
            mi::Vector<double> x{1, 2, 3};
            mi::Vector<double> y{4, 5};
            z.multiply_into(x, y);
            expect(close(y[0], 0.0));
            expect(close(y[1], 0.0));
        };

        test("transposed: shape flips, result is a length-columns() zero vector") = [] {
            mi::ZeroMatrix<double> z(3, 4);
            mi::Vector<double> x{1, 2, 3}; // length rows()
            auto y = z.multiply(x, mi::Transpose::Transposed);
            expect(y.size() == 4_u);
            expect(close(y[0], 0.0));
            expect(close(y[1], 0.0));
            expect(close(y[2], 0.0));
            expect(close(y[3], 0.0));
        };

        test("transposed multiply_into honors beta") = [] {
            mi::ZeroMatrix<double> z(3, 4);
            mi::Vector<double> x{1, 2, 3};    // length rows()
            mi::Vector<double> y{1, 1, 1, 1}; // length columns()
            z.multiply_into(x, y, 1.0, 3.0, mi::Transpose::Transposed);
            expect(close(y[0], 3.0));
            expect(close(y[1], 3.0));
            expect(close(y[2], 3.0));
            expect(close(y[3], 3.0));
        };

        test("wrong-length operands throw invalid_argument (None)") = [] {
            mi::ZeroMatrix<double> z(3, 4);
            mi::Vector<double> bad_x{1, 2, 3}; // should be length 4
            expect(throws<std::invalid_argument>([&] { (void)z.multiply(bad_x); }));

            mi::Vector<double> x{1, 2, 3, 4};
            mi::Vector<double> wrong_y{0, 0}; // should be length 3
            expect(throws<std::invalid_argument>([&] { z.multiply_into(x, wrong_y); }));
        };

        test("wrong-length operands throw invalid_argument (Transposed)") = [] {
            mi::ZeroMatrix<double> z(3, 4);
            mi::Vector<double> bad_x{1, 2, 3, 4}; // should be length 3 when transposed
            expect(throws<std::invalid_argument>([&] { (void)z.multiply(bad_x, mi::Transpose::Transposed); }));

            mi::Vector<double> x{1, 2, 3};
            mi::Vector<double> wrong_y{0, 0, 0}; // should be length 4 when transposed
            expect(throws<std::invalid_argument>(
                [&] { z.multiply_into(x, wrong_y, 1.0, 0.0, mi::Transpose::Transposed); }));
        };

        test("rectangular shape reports rows/columns/size/empty") = [] {
            mi::ZeroMatrix<double> z(3, 4);
            expect(z.rows() == 3_u);
            expect(z.columns() == 4_u);
            expect(z.size() == 12_u);
            expect(!z.empty());
        };

        test("zero-dimension shape is empty but keeps its dimensions") = [] {
            mi::ZeroMatrix<double> z(0, 5);
            expect(z.rows() == 0_u);
            expect(z.columns() == 5_u);
            expect(z.size() == 0_u);
            expect(z.empty());
        };

        test("float instantiation behaves the same") = [] {
            mi::ZeroMatrix<float> z(2, 2);
            mi::Vector<float> x{1.0F, 2.0F};
            auto y = z.multiply(x);
            expect(y.size() == 2_u);
            expect(close(y[0], 0.0F));
            expect(close(y[1], 0.0F));
        };
    };
}
