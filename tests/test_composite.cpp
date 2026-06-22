// test_composite.cpp -- unit tests for miscibility::instrument composite operators and the operator DSL.

#include "instrument/composite.hpp"
#include "instrument/context.hpp"
#include "instrument/dense_matrix.hpp"
#include "instrument/sparse_matrix.hpp"
#include "instrument/sparsity_pattern.hpp"
#include "instrument/vector.hpp"

#include <boost/ut.hpp>
#include <cmath>
#include <ginkgo/ginkgo.hpp>
#include <memory>
#include <vector>

namespace mi = miscibility::instrument;

namespace {

bool accepts_linop(const std::shared_ptr<const gko::LinOp>& op) { return op != nullptr; }

mi::SparseMatrix<double> diagonal_sparse(mi::Context& ctx, const char* name, std::initializer_list<double> diag)
{
    const auto n = static_cast<int>(diag.size());
    mi::SparsityPattern<double> pattern{gko::dim<2>{static_cast<gko::size_type>(n), static_cast<gko::size_type>(n)}};
    int i = 0;
    for (double d : diag) {
        pattern.add_value(i, i, d);
        ++i;
    }
    return mi::SparseMatrix<double>{ctx, name, pattern};
}

} // namespace

int main()
{
    using namespace boost::ut;

    suite<"BlockMatrix"> block_suite = [] {
        test("a 2x2 block matrix applies block-wise") = [] {
            mi::Context ctx;
            // A=2I, B=I, C=I, D=3I (each 2x2) -> full 4x4 operator
            auto a = diagonal_sparse(ctx, "A", {2.0, 2.0});
            auto b = diagonal_sparse(ctx, "B", {1.0, 1.0});
            auto c = diagonal_sparse(ctx, "C", {1.0, 1.0});
            auto d = diagonal_sparse(ctx, "D", {3.0, 3.0});
            mi::BlockMatrix block{ctx, "M", {{a, b}, {c, d}}};
            expect(block.block_size() == gko::dim<2>{2, 2});

            mi::Vector<double> x{ctx, "x", {1.0, 1.0, 1.0, 1.0}};
            mi::Vector<double> y{ctx, "y", 4};
            block.linop()->apply(x.linop(), y.linop());
            // top rows: A*x_top + B*x_bot = 2*1 + 1*1 = 3
            expect(y.at(0) == 3.0_d);
            expect(y.at(1) == 3.0_d);
            // bottom rows: C*x_top + D*x_bot = 1*1 + 3*1 = 4
            expect(y.at(2) == 4.0_d);
            expect(y.at(3) == 4.0_d);
        };

        test("null blocks are structural zeros") = [] {
            mi::Context ctx;
            auto a = diagonal_sparse(ctx, "A", {2.0, 2.0});
            auto d = diagonal_sparse(ctx, "D", {3.0, 3.0});
            mi::BlockMatrix block{ctx, "M", {{a, nullptr}, {nullptr, d}}};
            expect(block.block_at(0, 1) == nullptr);
            expect(block.block_at(1, 0) == nullptr);
            expect(block.block_at(0, 0) != nullptr);

            mi::Vector<double> x{ctx, "x", {1.0, 1.0, 1.0, 1.0}};
            mi::Vector<double> y{ctx, "y", 4};
            block.linop()->apply(x.linop(), y.linop());
            expect(y.at(0) == 2.0_d);
            expect(y.at(2) == 3.0_d);
        };

        test("heterogeneous block types apply together") = [] {
            mi::Context ctx;
            auto exec = ctx.executor();
            auto sparse = diagonal_sparse(ctx, "A", {2.0, 2.0});
            mi::DenseMatrix<double> dense{ctx, "D", {{3.0, 0.0}, {0.0, 3.0}}};
            mi::BlockMatrix block{ctx, "M", {{sparse, nullptr}, {nullptr, dense}}};

            mi::Vector<double> x{ctx, "x", {1.0, 1.0, 1.0, 1.0}};
            mi::Vector<double> y{ctx, "y", 4};
            block.linop()->apply(x.linop(), y.linop());
            expect(y.at(0) == 2.0_d);
            expect(y.at(2) == 3.0_d);
        };

        test("inconsistent block sizes throw at construction") = [] {
            mi::Context ctx;
            auto a = diagonal_sparse(ctx, "A", {2.0, 2.0}); // 2x2
            auto small = diagonal_sparse(ctx, "S", {1.0});  // 1x1
            expect(throws<gko::Error>([&] { mi::BlockMatrix block{ctx, "M", {{a, small}}}; }));
        };
    };

    suite<"composite builders"> builder_suite = [] {
        test("combination computes the scaled sum") = [] {
            mi::Context ctx;
            auto a = diagonal_sparse(ctx, "A", {1.0, 1.0, 1.0});
            auto b = diagonal_sparse(ctx, "B", {2.0, 2.0, 2.0});
            auto sum = mi::combination<double>(ctx, "2A+3B", {{2.0, a}, {3.0, b}});

            mi::Vector<double> x{ctx, "x", {1.0, 1.0, 1.0}};
            mi::Vector<double> y{ctx, "y", 3};
            sum.linop()->apply(x.linop(), y.linop());
            // 2*(A x) + 3*(B x) = 2*1 + 3*2 = 8
            expect(y.at(0) == 8.0_d);
        };

        test("composition applies right-to-left") = [] {
            mi::Context ctx;
            auto a = diagonal_sparse(ctx, "A", {2.0, 2.0});
            auto b = diagonal_sparse(ctx, "B", {3.0, 3.0});
            auto product = mi::composition<double>(ctx, "AB", {a, b});

            mi::Vector<double> x{ctx, "x", {1.0, 1.0}};
            mi::Vector<double> y{ctx, "y", 2};
            product.linop()->apply(x.linop(), y.linop());
            // A(B x) = 2*(3*1) = 6
            expect(y.at(0) == 6.0_d);
        };
    };

    suite<"operator DSL"> dsl_suite = [] {
        test("operator+ sums the applies") = [] {
            mi::Context ctx;
            auto a = diagonal_sparse(ctx, "A", {2.0, 2.0});
            auto b = diagonal_sparse(ctx, "B", {3.0, 3.0});
            auto sum = a + b;
            mi::Vector<double> x{ctx, "x", {1.0, 1.0}};
            mi::Vector<double> y{ctx, "y", 2};
            sum.linop()->apply(x.linop(), y.linop());
            expect(y.at(0) == 5.0_d); // 2 + 3
        };

        test("operator- subtracts the applies") = [] {
            mi::Context ctx;
            auto a = diagonal_sparse(ctx, "A", {5.0, 5.0});
            auto b = diagonal_sparse(ctx, "B", {3.0, 3.0});
            auto diff = a - b;
            mi::Vector<double> x{ctx, "x", {1.0, 1.0}};
            mi::Vector<double> y{ctx, "y", 2};
            diff.linop()->apply(x.linop(), y.linop());
            expect(y.at(0) == 2.0_d); // 5 - 3
        };

        test("operator* composes the applies") = [] {
            mi::Context ctx;
            auto a = diagonal_sparse(ctx, "A", {2.0, 2.0});
            auto b = diagonal_sparse(ctx, "B", {3.0, 3.0});
            auto product = a * b;
            mi::Vector<double> x{ctx, "x", {1.0, 1.0}};
            mi::Vector<double> y{ctx, "y", 2};
            product.linop()->apply(x.linop(), y.linop());
            expect(y.at(0) == 6.0_d); // 2*(3*1)
        };

        test("scalar multiplication scales the apply") = [] {
            mi::Context ctx;
            auto a = diagonal_sparse(ctx, "A", {3.0, 3.0});
            auto scaled = 2.0 * a;
            mi::Vector<double> x{ctx, "x", {1.0, 1.0}};
            mi::Vector<double> y{ctx, "y", 2};
            scaled.linop()->apply(x.linop(), y.linop());
            expect(y.at(0) == 6.0_d); // 2*(3*1)
        };

        test("composites nest and convert to a LinOp") = [] {
            mi::Context ctx;
            auto a = diagonal_sparse(ctx, "A", {2.0, 2.0});
            auto b = diagonal_sparse(ctx, "B", {3.0, 3.0});
            auto c = diagonal_sparse(ctx, "C", {1.0, 1.0});
            auto nested = (a + b) * c; // (A+B) composed with C
            expect(accepts_linop(nested));
            mi::Vector<double> x{ctx, "x", {1.0, 1.0}};
            mi::Vector<double> y{ctx, "y", 2};
            nested.linop()->apply(x.linop(), y.linop());
            // (A+B)(C x) = (2+3)*(1*1) = 5
            expect(y.at(0) == 5.0_d);
        };
    };

    return 0;
}
