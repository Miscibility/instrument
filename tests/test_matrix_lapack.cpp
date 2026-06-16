// test_matrix_lapack.cpp -- unit tests for LUFactorization: copy/move construction
// from a DenseMatrix and solving linear systems (LAPACK getrf/getrs). boost/ut.


#include <boost/ut.hpp>

#include <cmath>
#include <utility>

#include "instrument/matrix.hpp"

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

    suite<"DenseMatrixFactorization"> fact_suite = [] {
        test("construct and solve a 2x2 system; source A preserved by the copy") = [] {
            mi::DenseMatrix<double> a{{2, 1}, {1, 3}};
            mi::Vector<double> b{3, 5};
            mi::LUFactorization<double> lu{a};
            auto x = lu.solve(b);
            expect(close(x[0], 0.8));
            expect(close(x[1], 1.4));
            // A is intact (copy construction): verify A*x == b via the original matrix.
            auto ax = a.multiply(x);
            expect(close(ax[0], 3.0));
            expect(close(ax[1], 5.0));
            expect(a(0, 0) == 2.0_d); // entries unchanged
        };

        test("reuse one factorization across two right-hand sides") = [] {
            mi::DenseMatrix<double> a{{2, 1}, {1, 3}};
            mi::LUFactorization<double> lu{a};
            auto x1 = lu.solve(mi::Vector<double>{3, 5});
            auto x2 = lu.solve(mi::Vector<double>{5, 5});
            expect(close(x1[0], 0.8));
            expect(close(x1[1], 1.4));
            // Independent check for the second system: [2 1;1 3] x = [5;5] -> x = {2, 1}.
            expect(close(x2[0], 2.0));
            expect(close(x2[1], 1.0));
        };

        test("solve_into fills a pre-allocated x; wrong sizes throw") = [] {
            mi::DenseMatrix<double> a{{2, 1}, {1, 3}};
            mi::LUFactorization<double> lu{a};
            mi::Vector<double> x(2);
            lu.solve_into(mi::Vector<double>{3, 5}, x);
            expect(close(x[0], 0.8));
            expect(close(x[1], 1.4));

            mi::Vector<double> wrong_b{1, 2, 3};
            expect(throws<std::invalid_argument>([&] { (void)lu.solve(wrong_b); }));
            mi::Vector<double> wrong_x(3);
            expect(throws<std::invalid_argument>(
                [&] { lu.solve_into(mi::Vector<double>{3, 5}, wrong_x); }));
        };

        test("move construction guts a dynamic source to empty 0x0") = [] {
            mi::DenseMatrix<double> a{{2, 1}, {1, 3}};
            mi::LUFactorization<double> lu{std::move(a)};
            expect(a.rows() == 0_u);     // NOLINT(bugprone-use-after-move) -- intentional
            expect(a.columns() == 0_u);
            auto x = lu.solve(mi::Vector<double>{3, 5});
            expect(close(x[0], 0.8));
            expect(close(x[1], 1.4));
        };

        test("a static source routes through the copy overload and is left intact") = [] {
            mi::DenseMatrix<double, 2, 2> a{{2, 1}, {1, 3}};
            // NOLINTNEXTLINE(performance-move-const-arg) -- the move is the point: it must route to copy.
            mi::LUFactorization<double> lu{std::move(a)};
            // A static matrix has no movable buffer, so it is copied and stays valid.
            expect(a(0, 0) == 2.0_d);
            expect(a(1, 1) == 3.0_d);
            auto x = lu.solve(mi::Vector<double>{3, 5});
            expect(close(x[0], 0.8));
            expect(close(x[1], 1.4));
        };

        test("A.lu() copies (A intact), std::move(A).lu() steals (A emptied); same solution") = [] {
            mi::DenseMatrix<double> a{{2, 1}, {1, 3}};
            mi::Vector<double> b{3, 5};
            auto x1 = a.lu().solve(b); // lvalue -> copy, A preserved
            expect(a.rows() == 2_u);
            auto x2 = std::move(a).lu().solve(b); // rvalue -> move, A emptied
            expect(a.rows() == 0_u); // NOLINT(bugprone-use-after-move)
            expect(close(x1[0], x2[0]));
            expect(close(x1[1], x2[1]));
            expect(close(x1[0], 0.8));
        };

        test("singular matrix: singular() true and solve throws runtime_error") = [] {
            mi::DenseMatrix<double> a{{1, 2}, {2, 4}};
            mi::LUFactorization<double> lu{a};
            expect(lu.singular());
            expect(throws<std::runtime_error>([&] { (void)lu.solve(mi::Vector<double>{1, 1}); }));
        };

        test("non-square source throws invalid_argument at construction") = [] {
            mi::DenseMatrix<double> a{{1, 2, 3}, {4, 5, 6}};
            expect(throws<std::invalid_argument>([&] { mi::LUFactorization<double> lu{a}; }));
        };

        test("float and double both solve the happy path") = [] {
            mi::DenseMatrix<float> af{{2, 1}, {1, 3}};
            auto xf = mi::LUFactorization<float>{af}.solve(mi::Vector<float>{3, 5});
            expect(close(xf[0], 0.8F, 1e-8F));
            expect(close(xf[1], 1.4F, 1e-8F));

            mi::DenseMatrix<double> ad{{2, 1}, {1, 3}};
            auto xd = mi::LUFactorization<double>{ad}.solve(mi::Vector<double>{3, 5});
            expect(close(xd[0], 0.8));
            expect(close(xd[1], 1.4));
        };
    };
}
