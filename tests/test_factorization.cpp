// test_factorization.cpp -- unit tests for miscibility::instrument Factorization (dense LAPACKE solver).

#include "instrument/context.hpp"
#include "instrument/dense_matrix.hpp"
#include "instrument/factorization.hpp"
#include "instrument/vector.hpp"

#include <boost/ut.hpp>
#include <cmath>
#include <ginkgo/ginkgo.hpp>
#include <memory>

namespace mi = miscibility::instrument;

namespace {

bool accepts_linop(const std::shared_ptr<const gko::LinOp>& op) { return op != nullptr; }

// ||A*x - b||_2, the solve residual.
double residual_norm(mi::Context& ctx, mi::DenseMatrix<double>& a, const mi::Vector<double>& x,
                     const mi::Vector<double>& b)
{
    mi::Vector<double> r{ctx, "r", b.size()};
    a.apply(x, r);
    r.sub_scaled(1.0, b);
    return r.norm2();
}

} // namespace

int main()
{
    using namespace boost::ut;

    suite<"Factorization LU"> lu_suite = [] {
        test("LU solve satisfies A*x = b") = [] {
            mi::Context ctx;
            mi::DenseMatrix<double> a{ctx, "A", {{2.0, 1.0}, {1.0, 3.0}}};
            mi::Factorization<double> fact{ctx, "LU", a};
            mi::Vector<double> b{ctx, "b", {3.0, 5.0}};
            mi::Vector<double> x{ctx, "x", 2};
            fact.solve(b, x);
            expect(residual_norm(ctx, a, x, b) < 1e-10);
        };

        test("a single factorization solves multiple right-hand sides") = [] {
            mi::Context ctx;
            mi::DenseMatrix<double> a{ctx, "A", {{2.0, 1.0}, {1.0, 3.0}}};
            mi::Factorization<double> fact{ctx, "LU", a};

            mi::Vector<double> b1{ctx, "b1", {3.0, 5.0}};
            mi::Vector<double> x1{ctx, "x1", 2};
            fact.solve(b1, x1);
            expect(residual_norm(ctx, a, x1, b1) < 1e-10);

            mi::Vector<double> b2{ctx, "b2", {1.0, 1.0}};
            mi::Vector<double> x2{ctx, "x2", 2};
            fact.solve(b2, x2);
            expect(residual_norm(ctx, a, x2, b2) < 1e-10);
        };

        test("multi-column apply solves every column") = [] {
            mi::Context ctx;
            mi::DenseMatrix<double> a{ctx, "A", {{2.0, 1.0}, {1.0, 3.0}}};
            mi::Factorization<double> fact{ctx, "LU", a};
            mi::DenseMatrix<double> rhs{ctx, "B", {{3.0, 1.0}, {5.0, 1.0}}};
            mi::DenseMatrix<double> solution{ctx, "X", gko::dim<2>{2, 2}};
            fact.linop()->apply(rhs.linop(), solution.linop());

            mi::DenseMatrix<double> reconstructed{ctx, "R", gko::dim<2>{2, 2}};
            a.apply(solution, reconstructed);
            for (gko::size_type i = 0; i < 2; ++i) {
                for (gko::size_type j = 0; j < 2; ++j) {
                    expect(std::abs(reconstructed.at(i, j) - rhs.at(i, j)) < 1e-10);
                }
            }
        };

        test("singular matrix throws") = [] {
            mi::Context ctx;
            mi::DenseMatrix<double> a{ctx, "A", {{1.0, 2.0}, {2.0, 4.0}}};
            expect(throws<mi::FactorizationError>([&] { mi::Factorization<double> fact{ctx, "LU", a}; }));
        };

        test("DenseMatrix apply reproduces b from the LU solution") = [] {
            mi::Context ctx;
            mi::DenseMatrix<double> a{ctx, "A", {{2.0, 1.0}, {1.0, 3.0}}};
            mi::Factorization<double> fact{ctx, "LU", a};
            mi::Vector<double> b{ctx, "b", {3.0, 5.0}};
            mi::Vector<double> x{ctx, "x", 2};
            fact.solve(b, x);
            mi::Vector<double> reconstructed{ctx, "Ax", 2};
            a.apply(x, reconstructed);
            expect(std::abs(reconstructed.at(0) - 3.0) < 1e-10);
            expect(std::abs(reconstructed.at(1) - 5.0) < 1e-10);
        };
    };

    suite<"Factorization Cholesky"> cholesky_suite = [] {
        test("SPD matrix solves correctly") = [] {
            mi::Context ctx;
            mi::DenseMatrix<double> a{ctx, "A", {{4.0, 2.0}, {2.0, 3.0}}};
            mi::Factorization<double> fact{ctx, "chol", a, mi::FactorKind::Cholesky};
            mi::Vector<double> b{ctx, "b", {6.0, 5.0}};
            mi::Vector<double> x{ctx, "x", 2};
            fact.solve(b, x);
            expect(residual_norm(ctx, a, x, b) < 1e-10);
        };

        test("non-SPD matrix throws") = [] {
            mi::Context ctx;
            mi::DenseMatrix<double> a{ctx, "A", {{1.0, 2.0}, {2.0, 1.0}}};
            expect(throws<mi::FactorizationError>(
                [&] { mi::Factorization<double> fact{ctx, "chol", a, mi::FactorKind::Cholesky}; }));
        };
    };

    suite<"Factorization LDLT"> ldlt_suite = [] {
        test("symmetric indefinite matrix solves correctly") = [] {
            mi::Context ctx;
            mi::DenseMatrix<double> a{ctx, "A", {{1.0, 2.0}, {2.0, 1.0}}};
            mi::Factorization<double> fact{ctx, "ldlt", a, mi::FactorKind::LDLT};
            mi::Vector<double> b{ctx, "b", {5.0, 4.0}};
            mi::Vector<double> x{ctx, "x", 2};
            fact.solve(b, x);
            expect(residual_norm(ctx, a, x, b) < 1e-10);
        };
    };

    suite<"Factorization QR"> qr_suite = [] {
        test("square system solves correctly") = [] {
            mi::Context ctx;
            mi::DenseMatrix<double> a{ctx, "A", {{2.0, 1.0}, {1.0, 3.0}}};
            mi::Factorization<double> fact{ctx, "qr", a, mi::FactorKind::QR};
            mi::Vector<double> b{ctx, "b", {3.0, 5.0}};
            mi::Vector<double> x{ctx, "x", 2};
            fact.solve(b, x);
            expect(residual_norm(ctx, a, x, b) < 1e-10);
        };

        test("overdetermined system gives the least-squares solution") = [] {
            mi::Context ctx;
            // columns span {(1,0,1),(0,1,1)}; b = 1*c0 + 2*c1 lies in the range,
            // so the least-squares solution is exactly (1, 2).
            mi::DenseMatrix<double> a{ctx, "A", {{1.0, 0.0}, {0.0, 1.0}, {1.0, 1.0}}};
            mi::Factorization<double> fact{ctx, "qr", a, mi::FactorKind::QR};
            mi::Vector<double> b{ctx, "b", {1.0, 2.0, 3.0}};
            mi::Vector<double> x{ctx, "x", 2};
            fact.solve(b, x);
            expect(std::abs(x.at(0) - 1.0) < 1e-10);
            expect(std::abs(x.at(1) - 2.0) < 1e-10);
        };
    };

    suite<"Factorization handle"> handle_suite = [] {
        test("reports the factorization kind") = [] {
            mi::Context ctx;
            mi::DenseMatrix<double> a{ctx, "A", {{4.0, 2.0}, {2.0, 3.0}}};
            mi::Factorization<double> fact{ctx, "chol", a, mi::FactorKind::Cholesky};
            expect(fact.kind() == mi::FactorKind::Cholesky);
        };

        test("implicitly converts to shared_ptr<const gko::LinOp>") = [] {
            mi::Context ctx;
            mi::DenseMatrix<double> a{ctx, "A", {{2.0, 1.0}, {1.0, 3.0}}};
            mi::Factorization<double> fact{ctx, "LU", a};
            expect(accepts_linop(fact));
        };
    };

    return 0;
}
