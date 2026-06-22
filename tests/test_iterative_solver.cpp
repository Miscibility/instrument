// test_iterative_solver.cpp -- unit tests for miscibility::instrument IterativeSolver.

#include "instrument/context.hpp"
#include "instrument/iterative_solver.hpp"
#include "instrument/sparse_matrix.hpp"
#include "instrument/sparsity_pattern.hpp"
#include "instrument/timing.hpp"
#include "instrument/vector.hpp"

#include <boost/ut.hpp>
#include <cmath>
#include <ginkgo/ginkgo.hpp>

namespace mi = miscibility::instrument;

namespace {

// SPD tridiagonal matrix: 2 on the diagonal, -1 off-diagonal.
mi::SparseMatrix<double> make_spd(mi::Context& ctx, int n)
{
    mi::SparsityPattern<double> pattern{gko::dim<2>{static_cast<gko::size_type>(n), static_cast<gko::size_type>(n)}};
    for (int i = 0; i < n; ++i) {
        pattern.add_value(i, i, 2.0);
        if (i > 0) {
            pattern.add_value(i, i - 1, -1.0);
        }
        if (i < n - 1) {
            pattern.add_value(i, i + 1, -1.0);
        }
    }
    return mi::SparseMatrix<double>{ctx, "A", pattern};
}

// Nonsymmetric matrix.
mi::SparseMatrix<double> make_nonsymmetric(mi::Context& ctx, int n)
{
    mi::SparsityPattern<double> pattern{gko::dim<2>{static_cast<gko::size_type>(n), static_cast<gko::size_type>(n)}};
    for (int i = 0; i < n; ++i) {
        pattern.add_value(i, i, 3.0);
        if (i < n - 1) {
            pattern.add_value(i, i + 1, -1.0);
        }
        if (i > 0) {
            pattern.add_value(i, i - 1, -2.0);
        }
    }
    return mi::SparseMatrix<double>{ctx, "A", pattern};
}

double residual_norm(mi::Context& ctx, mi::SparseMatrix<double>& a, const mi::Vector<double>& x,
                     const mi::Vector<double>& b)
{
    mi::Vector<double> r{ctx, "r", b.size()};
    a.linop()->apply(x.linop(), r.linop());
    r.sub_scaled(1.0, b);
    return r.norm2();
}

} // namespace

int main()
{
    using namespace boost::ut;

    suite<"IterativeSolver"> solver_suite = [] {
        test("CG converges on an SPD system") = [] {
            mi::Context ctx;
            auto a = make_spd(ctx, 10);
            mi::IterativeSolver<double> solver{
                ctx, "cg", {.type = mi::SolverType::Cg, .stop = {.rel_reduction = 1e-10}}};
            mi::Vector<double> b{ctx, "b", 10, 1.0};
            mi::Vector<double> x{ctx, "x", 10};
            auto stats = solver.solve(a, b, x);
            expect(stats.converged);
            expect(stats.iterations > 0_u);
            expect(residual_norm(ctx, a, x, b) < 1e-8);
        };

        test("GMRES converges on a nonsymmetric system") = [] {
            mi::Context ctx;
            auto a = make_nonsymmetric(ctx, 10);
            mi::IterativeSolver<double> solver{
                ctx, "gmres", {.type = mi::SolverType::Gmres, .stop = {.rel_reduction = 1e-10}}};
            mi::Vector<double> b{ctx, "b", 10, 1.0};
            mi::Vector<double> x{ctx, "x", 10};
            auto stats = solver.solve(a, b, x);
            expect(stats.converged);
            expect(residual_norm(ctx, a, x, b) < 1e-8);
        };

        test("an exhausted iteration budget reports non-convergence") = [] {
            mi::Context ctx;
            auto a = make_spd(ctx, 50);
            mi::IterativeSolver<double> solver{ctx, "cg", {.type = mi::SolverType::Cg, .stop = {.max_iters = 2}}};
            mi::Vector<double> b{ctx, "b", 50, 1.0};
            mi::Vector<double> x{ctx, "x", 50};
            auto stats = solver.solve(a, b, x);
            expect(not stats.converged);
            expect(stats.iterations == 2_u);
        };

        test("a Jacobi preconditioner converges in no more iterations") = [] {
            mi::Context ctx;
            auto a = make_spd(ctx, 30);
            mi::Vector<double> b{ctx, "b", 30, 1.0};

            mi::IterativeSolver<double> plain{
                ctx, "cg", {.type = mi::SolverType::Cg, .stop = {.rel_reduction = 1e-10}}};
            mi::Vector<double> x_plain{ctx, "x1", 30};
            auto plain_stats = plain.solve(a, b, x_plain);

            mi::IterativeSolver<double> prec{ctx,
                                             "pcg",
                                             {.type = mi::SolverType::Cg,
                                              .stop = {.rel_reduction = 1e-10},
                                              .preconditioner = mi::PrecondOptions{.type = mi::PrecondType::Jacobi}}};
            mi::Vector<double> x_prec{ctx, "x2", 30};
            auto prec_stats = prec.solve(a, b, x_prec);

            expect(prec_stats.converged);
            expect(prec_stats.iterations <= plain_stats.iterations);
        };

        test("reusing an unchanged matrix regenerates only on the first solve") = [] {
            mi::Context ctx;
            auto a = make_spd(ctx, 10);
            mi::IterativeSolver<double> solver{
                ctx, "cg", {.type = mi::SolverType::Cg, .stop = {.rel_reduction = 1e-10}}};
            mi::Vector<double> b{ctx, "b", 10, 1.0};
            mi::Vector<double> x1{ctx, "x1", 10};
            mi::Vector<double> x2{ctx, "x2", 10};
            auto first = solver.solve(a, b, x1);
            auto second = solver.solve(a, b, x2);
            expect(first.regenerated);
            expect(not second.regenerated);
        };

        test("an in-place value update forces regeneration") = [] {
            mi::Context ctx;
            auto a = make_spd(ctx, 10);
            mi::IterativeSolver<double> solver{
                ctx, "cg", {.type = mi::SolverType::Cg, .stop = {.rel_reduction = 1e-10}}};
            mi::Vector<double> b{ctx, "b", 10, 1.0};
            mi::Vector<double> x{ctx, "x", 10};
            solver.solve(a, b, x);

            mi::SparsityPattern<double> updated{gko::dim<2>{10, 10}};
            for (int i = 0; i < 10; ++i) {
                updated.add_value(i, i, 4.0);
                if (i > 0) {
                    updated.add_value(i, i - 1, -1.0);
                }
                if (i < 9) {
                    updated.add_value(i, i + 1, -1.0);
                }
            }
            a.update_values(updated);
            auto after = solver.solve(a, b, x);
            expect(after.regenerated);
        };

        test("prepare makes the following solve report no regeneration") = [] {
            mi::Context ctx;
            auto a = make_spd(ctx, 10);
            mi::IterativeSolver<double> solver{
                ctx, "cg", {.type = mi::SolverType::Cg, .stop = {.rel_reduction = 1e-10}}};
            solver.prepare(a);
            mi::Vector<double> b{ctx, "b", 10, 1.0};
            mi::Vector<double> x{ctx, "x", 10};
            auto stats = solver.solve(a, b, x);
            expect(not stats.regenerated);
        };

        test("force_regenerate makes the next solve regenerate") = [] {
            mi::Context ctx;
            auto a = make_spd(ctx, 10);
            mi::IterativeSolver<double> solver{
                ctx, "cg", {.type = mi::SolverType::Cg, .stop = {.rel_reduction = 1e-10}}};
            mi::Vector<double> b{ctx, "b", 10, 1.0};
            mi::Vector<double> x{ctx, "x", 10};
            solver.solve(a, b, x);
            solver.force_regenerate();
            auto stats = solver.solve(a, b, x);
            expect(stats.regenerated);
        };

        test("zero_initial_guess ignores a garbage x on entry") = [] {
            mi::Context ctx;
            auto a = make_spd(ctx, 10);
            mi::IterativeSolver<double> solver{
                ctx, "cg", {.type = mi::SolverType::Cg, .stop = {.rel_reduction = 1e-10}, .zero_initial_guess = true}};
            mi::Vector<double> b{ctx, "b", 10, 1.0};
            mi::Vector<double> x{ctx, "x", 10, 1e6}; // garbage initial guess
            auto stats = solver.solve(a, b, x);
            expect(stats.converged);
            expect(residual_norm(ctx, a, x, b) < 1e-8);
        };

        test("statistics report a consistent relative residual") = [] {
            mi::Context ctx;
            auto a = make_spd(ctx, 10);
            mi::IterativeSolver<double> solver{
                ctx, "cg", {.type = mi::SolverType::Cg, .stop = {.rel_reduction = 1e-10}}};
            mi::Vector<double> b{ctx, "b", 10, 1.0};
            mi::Vector<double> x{ctx, "x", 10};
            auto stats = solver.solve(a, b, x);
            const double expected = stats.residual_norm / b.norm2();
            expect(std::abs(stats.relative_residual - expected) < 1e-12);
        };

        test("an absolute tolerance stops at the requested residual") = [] {
            mi::Context ctx;
            auto a = make_spd(ctx, 10);
            mi::IterativeSolver<double> solver{ctx, "cg", {.type = mi::SolverType::Cg, .stop = {.abs_tol = 1e-6}}};
            mi::Vector<double> b{ctx, "b", 10, 1.0};
            mi::Vector<double> x{ctx, "x", 10};
            auto stats = solver.solve(a, b, x);
            expect(stats.converged);
            expect(residual_norm(ctx, a, x, b) < 1e-4);
        };

        test("empty stopping criteria is rejected") = [] {
            mi::Context ctx;
            expect(throws<std::invalid_argument>(
                [&] { mi::IterativeSolver<double> solver{ctx, "cg", {.type = mi::SolverType::Cg}}; }));
        };

        test("the solve is recorded as a named timing region") = [] {
            mi::reset();
            mi::Context ctx;
            auto a = make_spd(ctx, 10);
            mi::IterativeSolver<double> solver{
                ctx, "cg_solve", {.type = mi::SolverType::Cg, .stop = {.rel_reduction = 1e-10}}};
            mi::Vector<double> b{ctx, "b", 10, 1.0};
            mi::Vector<double> x{ctx, "x", 10};
            solver.solve(a, b, x);
            auto region = mi::query("cg_solve");
            expect(region.has_value());
            expect(region->calls >= 1_u);
        };
    };

    return 0;
}
