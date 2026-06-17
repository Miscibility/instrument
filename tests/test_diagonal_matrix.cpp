// test_diagonal_matrix.cpp -- unit tests for DiagonalMatrix: construction, the
// Highway matrix-vector product, diagonal-times-dense row scaling, the member
// solve, and the cached singular() flag. boost/ut.
//
// Every member currently throws std::runtime_error{"not implemented"}, so each
// runtime test below fails until tdd-3-implement fills them in. The compile-time
// static_assert pins the MatrixOperator shape and must hold for the suite to
// compile.

#include "instrument/diagonal_matrix.hpp"
#include "instrument/matrix.hpp"
#include "instrument/vector.hpp"

#include <boost/ut.hpp>
#include <stdexcept>

namespace mi = miscibility::instrument;

// A DiagonalMatrix models the matrix-vector seam.
static_assert(mi::MatrixOperator<mi::DiagonalMatrix<double>, double>);
static_assert(mi::MatrixOperator<mi::DiagonalMatrix<float>, float>);

namespace {

template<class T> bool close(T a, T b, T tol = T(1e-12))
{
    return std::abs(a - b) <= tol * (T(1) + std::abs(a) + std::abs(b));
}

} // namespace

int main()
{
    using namespace boost::ut;

    suite<"DiagonalMatrix"> diag_suite = [] {
        // -- construction / access -------------------------------------------

        test("construct from initializer list: shape and element read-back") = [] {
            mi::DiagonalMatrix<double> d{2, 3, 4};
            expect(d.rows() == 3_u);
            expect(d.columns() == 3_u);
            expect(d.size() == 3_u);
            expect(!d.empty());
            expect(close(d[0], 2.0));
            expect(close(d[1], 3.0));
            expect(close(d[2], 4.0));
            expect(close(d.at(0), 2.0));
            expect(close(d.at(2), 4.0));
        };

        test("at is bounds-checked") = [] {
            mi::DiagonalMatrix<double> d{2, 3, 4};
            expect(throws<std::out_of_range>([&] { (void)d.at(3); }));
        };

        test("static shape works inline") = [] {
            mi::DiagonalMatrix<double, 3> d{1, 2, 3};
            expect(d.rows() == 3_u);
            expect(close(d[0], 1.0));
            expect(close(d[2], 3.0));
        };

        test("static shape with wrong-length initializer list throws invalid_argument") = [] {
            expect(throws<std::invalid_argument>([] { mi::DiagonalMatrix<double, 3> d{1, 2}; }));
        };

        test("construct from a Vector adopts the diagonal") = [] {
            mi::Vector<double> v{5, 6, 7};
            mi::DiagonalMatrix<double> d{v};
            expect(d.size() == 3_u);
            expect(close(d[1], 6.0));
        };

        test("diagonal() exposes the backing vector") = [] {
            mi::DiagonalMatrix<double> d{2, 3, 4};
            const auto& v = d.diagonal();
            expect(v.size() == 3_u);
            expect(close(v[2], 4.0));
        };

        // -- matrix-vector product -------------------------------------------

        test("matvec: diag{2,3,4} * {1,1,1} == {2,3,4}") = [] {
            mi::DiagonalMatrix<double> d{2, 3, 4};
            mi::Vector<double> x{1, 1, 1};
            auto y = d.multiply(x);
            expect(y.size() == 3_u);
            expect(close(y[0], 2.0));
            expect(close(y[1], 3.0));
            expect(close(y[2], 4.0));
            auto y2 = d * x;
            expect(close(y2[0], 2.0));
            expect(close(y2[1], 3.0));
            expect(close(y2[2], 4.0));
        };

        test("matvec with a non-unit x matches a hand loop") = [] {
            mi::DiagonalMatrix<double> d{2, 3, 4};
            mi::Vector<double> x{5, 6, 7};
            auto y = d.multiply(x);
            expect(close(y[0], 10.0)); // 2*5
            expect(close(y[1], 18.0)); // 3*6
            expect(close(y[2], 28.0)); // 4*7
        };

        test("matvec gemv: multiply_into computes alpha*d_i*x_i + beta*y_i") = [] {
            mi::DiagonalMatrix<double> d{2, 3, 4};
            mi::Vector<double> x{1, 1, 1};
            mi::Vector<double> y{1, 1, 1};
            d.multiply_into(x, y, 2.0, 3.0); // 2*d_i*1 + 3*1
            expect(close(y[0], 7.0));        // 2*2 + 3
            expect(close(y[1], 9.0));        // 2*3 + 3
            expect(close(y[2], 11.0));       // 2*4 + 3
        };

        test("matvec transposed equals untransposed (diagonal is its own transpose)") = [] {
            mi::DiagonalMatrix<double> d{2, 3, 4};
            mi::Vector<double> x{5, 6, 7};
            auto y_none = d.multiply(x, mi::Transpose::None);
            auto y_trans = d.multiply(x, mi::Transpose::Transposed);
            expect(close(y_trans[0], y_none[0]));
            expect(close(y_trans[1], y_none[1]));
            expect(close(y_trans[2], y_none[2]));
        };

        test("matvec wrong-length operands throw invalid_argument") = [] {
            mi::DiagonalMatrix<double> d{2, 3, 4};
            mi::Vector<double> bad_x{1, 2};
            expect(throws<std::invalid_argument>([&] { (void)d.multiply(bad_x); }));
            mi::Vector<double> x{1, 2, 3};
            mi::Vector<double> bad_y{0, 0};
            expect(throws<std::invalid_argument>([&] { d.multiply_into(x, bad_y); }));
        };

        // -- matrix-matrix product (row scaling) -----------------------------

        test("matmat: diag{2,3} * {{1,2},{3,4}} scales rows -> {{2,4},{9,12}}") = [] {
            mi::DiagonalMatrix<double> d{2, 3};
            mi::DenseMatrix<double> b{{1, 2}, {3, 4}};
            auto c = d.multiply(b);
            expect(c.rows() == 2_u);
            expect(c.columns() == 2_u);
            expect(close(c(0, 0), 2.0));  // 2*1
            expect(close(c(0, 1), 4.0));  // 2*2
            expect(close(c(1, 0), 9.0));  // 3*3
            expect(close(c(1, 1), 12.0)); // 3*4

            // matches multiplying by the equivalent dense diagonal
            mi::DenseMatrix<double> dense_d{{2, 0}, {0, 3}};
            auto expected = dense_d.multiply(b);
            expect(close(c(0, 0), expected(0, 0)));
            expect(close(c(1, 1), expected(1, 1)));

            auto c2 = d * b;
            expect(close(c2(1, 0), 9.0));
        };

        test("matmat with opB == Transposed scales rows of B^T") = [] {
            mi::DiagonalMatrix<double> d{2, 3};
            mi::DenseMatrix<double> b{{1, 2}, {3, 4}};
            auto c = d.multiply(b, mi::Transpose::Transposed);
            // B^T == {{1,3},{2,4}}; row-scale by {2,3} -> {{2,6},{6,12}}
            expect(close(c(0, 0), 2.0));
            expect(close(c(0, 1), 6.0));
            expect(close(c(1, 0), 6.0));
            expect(close(c(1, 1), 12.0));
        };

        test("matmat gemv-style: C <- alpha*D*B + beta*C accumulates") = [] {
            mi::DiagonalMatrix<double> d{2, 3};
            mi::DenseMatrix<double> b{{1, 2}, {3, 4}};
            mi::DenseMatrix<double> c{{1, 1}, {1, 1}};
            d.multiply_into(b, c, 2.0, 3.0); // 2*D*B + 3*C
            expect(close(c(0, 0), 7.0));     // 2*(2*1) + 3*1 = 7
            expect(close(c(0, 1), 11.0));    // 2*(2*2) + 3*1 = 11
            expect(close(c(1, 0), 21.0));    // 2*(3*3) + 3*1 = 21
            expect(close(c(1, 1), 27.0));    // 2*(3*4) + 3*1 = 27
        };

        test("matmat shape mismatch throws invalid_argument") = [] {
            mi::DiagonalMatrix<double> d{2, 3};      // 2x2
            mi::DenseMatrix<double> b{{1, 2, 3}};    // 1x3, only 1 row
            expect(throws<std::invalid_argument>([&] { (void)d.multiply(b); }));
        };

        // -- solve ------------------------------------------------------------

        test("solve: diag{2,4,5} solving b={2,4,5} -> x={1,1,1}") = [] {
            mi::DiagonalMatrix<double> d{2, 4, 5};
            mi::Vector<double> b{2, 4, 5};
            auto x = d.solve(b);
            expect(x.size() == 3_u);
            expect(close(x[0], 1.0));
            expect(close(x[1], 1.0));
            expect(close(x[2], 1.0));
        };

        test("solve: non-unit case equals b_i / d_i") = [] {
            mi::DiagonalMatrix<double> d{2, 4, 5};
            mi::Vector<double> b{6, 2, 20};
            mi::Vector<double> x(3);
            d.solve_into(b, x);
            expect(close(x[0], 3.0));
            expect(close(x[1], 0.5));
            expect(close(x[2], 4.0));
        };

        // -- singularity ------------------------------------------------------

        test("singular() is false for an all-nonzero diagonal") = [] {
            mi::DiagonalMatrix<double> d{1, 2, 3};
            expect(!d.singular());
        };

        test("singular() is true when any diagonal entry is zero") = [] {
            mi::DiagonalMatrix<double> d{1, 0, 3};
            expect(d.singular());
        };

        test("solve on a singular diagonal throws runtime_error") = [] {
            mi::DiagonalMatrix<double> d{1, 0, 3}; // arrange (must succeed once implemented)
            mi::Vector<double> b{1, 2, 3};
            expect(throws<std::runtime_error>([&] { (void)d.solve(b); }));
        };

        // -- mutation ---------------------------------------------------------

        test("set updates the entry and the cached singular flag both ways") = [] {
            mi::DiagonalMatrix<double> d{1, 2, 3};
            expect(!d.singular());
            d.set(1, 0.0);
            expect(d.singular());
            expect(close(d[1], 0.0));
            d.set(1, 5.0);
            expect(!d.singular());
            expect(close(d[1], 5.0));
        };

        test("set with two zeros then clearing one keeps singular until both cleared") = [] {
            mi::DiagonalMatrix<double> d{0, 2, 0};
            expect(d.singular());
            d.set(0, 1.0);
            expect(d.singular()); // entry 2 still zero
            d.set(2, 1.0);
            expect(!d.singular());
        };

        test("set out of range throws out_of_range") = [] {
            mi::DiagonalMatrix<double> d{1, 2, 3};
            expect(throws<std::out_of_range>([&] { d.set(3, 9.0); }));
        };

        // -- float ------------------------------------------------------------

        test("float instantiation") = [] {
            mi::DiagonalMatrix<float> d{2.0F, 4.0F};
            mi::Vector<float> x{3.0F, 5.0F};
            auto y = d.multiply(x);
            expect(close(y[0], 6.0F, 1e-6F));
            expect(close(y[1], 20.0F, 1e-6F));
        };
    };
}
