// test_preconditioner.cpp -- unit tests for miscibility::instrument Preconditioner.

#include "instrument/context.hpp"
#include "instrument/preconditioner.hpp"
#include "instrument/sparse_matrix.hpp"
#include "instrument/sparsity_pattern.hpp"
#include "instrument/vector.hpp"

#include <boost/ut.hpp>
#include <cmath>
#include <ginkgo/ginkgo.hpp>
#include <memory>

namespace mi = miscibility::instrument;

namespace {

bool accepts_linop(const std::shared_ptr<const gko::LinOp>& op) { return op != nullptr; }

mi::SparseMatrix<double> make_diagonal(mi::Context& ctx, std::initializer_list<double> diag)
{
    const auto n = static_cast<int>(diag.size());
    mi::SparsityPattern<double> pattern{gko::dim<2>{static_cast<gko::size_type>(n), static_cast<gko::size_type>(n)}};
    int i = 0;
    for (double d : diag) {
        pattern.add_value(i, i, d);
        ++i;
    }
    return mi::SparseMatrix<double>{ctx, "A", pattern};
}

} // namespace

int main()
{
    using namespace boost::ut;

    suite<"Preconditioner"> precond_suite = [] {
        test("Jacobi on a diagonal matrix divides componentwise") = [] {
            mi::Context ctx;
            auto a = make_diagonal(ctx, {2.0, 4.0, 5.0});
            mi::Preconditioner<double> m{ctx, "M", a, {.type = mi::PrecondType::Jacobi}};
            mi::Vector<double> r{ctx, "r", {2.0, 8.0, 5.0}};
            mi::Vector<double> z{ctx, "z", 3};
            m.apply(r, z);
            expect(std::abs(z.at(0) - 1.0) < 1e-12);
            expect(std::abs(z.at(1) - 2.0) < 1e-12);
            expect(std::abs(z.at(2) - 1.0) < 1e-12);
        };

        test("None acts as the identity") = [] {
            mi::Context ctx;
            auto a = make_diagonal(ctx, {2.0, 4.0});
            mi::Preconditioner<double> m{ctx, "M", a, {.type = mi::PrecondType::None}};
            mi::Vector<double> r{ctx, "r", {7.0, 9.0}};
            mi::Vector<double> z{ctx, "z", 2};
            m.apply(r, z);
            expect(z.at(0) == 7.0_d);
            expect(z.at(1) == 9.0_d);
        };

        test("block Jacobi inverts each diagonal block") = [] {
            mi::Context ctx;
            // block-diagonal: [[2,1],[1,2]] and [3]
            mi::SparsityPattern<double> pattern{gko::dim<2>{3, 3}};
            pattern.add_value(0, 0, 2.0);
            pattern.add_value(0, 1, 1.0);
            pattern.add_value(1, 0, 1.0);
            pattern.add_value(1, 1, 2.0);
            pattern.add_value(2, 2, 3.0);
            mi::SparseMatrix<double> a{ctx, "A", pattern};
            mi::Preconditioner<double> m{ctx, "M", a, {.type = mi::PrecondType::BlockJacobi, .max_block_size = 2}};
            mi::Vector<double> r{ctx, "r", {1.0, 1.0, 3.0}};
            mi::Vector<double> z{ctx, "z", 3};
            m.apply(r, z);
            // inverse of [[2,1],[1,2]] is (1/3)[[2,-1],[-1,2]]; applied to {1,1} -> {1/3,1/3}
            expect(std::abs(z.at(0) - (1.0 / 3.0)) < 1e-12);
            expect(std::abs(z.at(1) - (1.0 / 3.0)) < 1e-12);
            expect(std::abs(z.at(2) - 1.0) < 1e-12);
        };

        test("ILU builds and applies to give finite, correctly-sized output") = [] {
            mi::Context ctx;
            mi::SparsityPattern<double> pattern{gko::dim<2>{3, 3}};
            pattern.add_value(0, 0, 4.0);
            pattern.add_value(1, 0, 1.0);
            pattern.add_value(1, 1, 4.0);
            pattern.add_value(2, 1, 1.0);
            pattern.add_value(2, 2, 4.0);
            mi::SparseMatrix<double> a{ctx, "A", pattern};
            mi::Preconditioner<double> m{ctx, "M", a, {.type = mi::PrecondType::Ilu}};
            mi::Vector<double> r{ctx, "r", {1.0, 2.0, 3.0}};
            mi::Vector<double> z{ctx, "z", 3};
            m.apply(r, z);
            expect(std::isfinite(z.at(0)));
            expect(std::isfinite(z.at(1)));
            expect(std::isfinite(z.at(2)));
        };

        test("single-precision storage with refinement matches the double Jacobi result") = [] {
            mi::Context ctx;
            auto a = make_diagonal(ctx, {2.0, 4.0});
            mi::Preconditioner<double> m{
                ctx, "M", a, {.type = mi::PrecondType::Jacobi, .storage = mi::Precision::Single, .refine = true}};
            mi::Vector<double> r{ctx, "r", {2.0, 8.0}};
            mi::Vector<double> z{ctx, "z", 2};
            m.apply(r, z);
            expect(std::abs(z.at(0) - 1.0) < 1e-5);
            expect(std::abs(z.at(1) - 2.0) < 1e-5);
        };

        test("half-precision storage builds and applies") = [] {
            mi::Context ctx;
            auto a = make_diagonal(ctx, {2.0, 4.0});
            mi::Preconditioner<double> m{
                ctx, "M", a, {.type = mi::PrecondType::Jacobi, .storage = mi::Precision::Half}};
            mi::Vector<double> r{ctx, "r", {2.0, 8.0}};
            mi::Vector<double> z{ctx, "z", 2};
            m.apply(r, z);
            expect(std::isfinite(z.at(0)));
            expect(std::isfinite(z.at(1)));
        };

        test("bfloat16 storage builds and applies") = [] {
            mi::Context ctx;
            auto a = make_diagonal(ctx, {2.0, 4.0});
            mi::Preconditioner<double> m{
                ctx, "M", a, {.type = mi::PrecondType::Jacobi, .storage = mi::Precision::BFloat16}};
            mi::Vector<double> r{ctx, "r", {2.0, 8.0}};
            mi::Vector<double> z{ctx, "z", 2};
            m.apply(r, z);
            expect(std::isfinite(z.at(0)));
            expect(std::isfinite(z.at(1)));
        };

        test("adopts a custom preconditioner LinOp") = [] {
            mi::Context ctx;
            auto exec = ctx.executor();
            // a hand-written diagonal scaler: z = diag(2,2) * r
            auto scaler = gko::share(gko::matrix::Diagonal<double>::create(exec, 2));
            scaler->get_values()[0] = 2.0;
            scaler->get_values()[1] = 2.0;
            mi::Preconditioner<double> m{ctx, "M", scaler};
            mi::Vector<double> r{ctx, "r", {1.0, 2.0}};
            mi::Vector<double> z{ctx, "z", 2};
            m.apply(r, z);
            expect(z.at(0) == 2.0_d);
            expect(z.at(1) == 4.0_d);
        };

        test("implicitly converts to shared_ptr<const gko::LinOp>") = [] {
            mi::Context ctx;
            auto a = make_diagonal(ctx, {2.0, 4.0});
            mi::Preconditioner<double> m{ctx, "M", a};
            expect(accepts_linop(m));
        };
    };

    return 0;
}
