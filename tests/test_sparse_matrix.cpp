// test_sparse_matrix.cpp -- unit tests for miscibility::instrument SparseMatrix.

#include "instrument/context.hpp"
#include "instrument/sparse_matrix.hpp"
#include "instrument/sparsity_pattern.hpp"
#include "instrument/vector.hpp"

#include <array>
#include <boost/ut.hpp>
#include <ginkgo/ginkgo.hpp>
#include <memory>

namespace mi = miscibility::instrument;

namespace {

bool accepts_linop(const std::shared_ptr<const gko::LinOp>& op) { return op != nullptr; }

// A 3x3 pattern:
//   [2 0 1]
//   [0 3 0]
//   [4 0 5]
mi::SparsityPattern<double> make_pattern()
{
    mi::SparsityPattern<double> pattern{gko::dim<2>{3, 3}};
    pattern.add_value(0, 0, 2.0);
    pattern.add_value(0, 2, 1.0);
    pattern.add_value(1, 1, 3.0);
    pattern.add_value(2, 0, 4.0);
    pattern.add_value(2, 2, 5.0);
    return pattern;
}

} // namespace

int main()
{
    using namespace boost::ut;

    suite<"SparseMatrix build"> build_suite = [] {
        test("CSR SpMV matches the dense reference") = [] {
            mi::Context ctx;
            mi::SparseMatrix<double> a{ctx, "A", make_pattern()};
            mi::Vector<double> x{ctx, "x", {1.0, 1.0, 1.0}};
            mi::Vector<double> y{ctx, "y", 3};
            a.linop()->apply(x.linop(), y.linop());
            // A*[1,1,1] = [3, 3, 9]
            expect(y.at(0) == 3.0_d);
            expect(y.at(1) == 3.0_d);
            expect(y.at(2) == 9.0_d);
        };

        test("Coo and Ell give the same apply result as CSR") = [] {
            mi::Context ctx;
            mi::SparseMatrix<double> csr{ctx, "csr", make_pattern(), {.format = mi::SparseFormat::Csr}};
            mi::SparseMatrix<double> coo{ctx, "coo", make_pattern(), {.format = mi::SparseFormat::Coo}};
            mi::SparseMatrix<double> ell{ctx, "ell", make_pattern(), {.format = mi::SparseFormat::Ell}};
            mi::Vector<double> x{ctx, "x", {1.0, 2.0, 3.0}};
            mi::Vector<double> y_csr{ctx, "y1", 3};
            mi::Vector<double> y_coo{ctx, "y2", 3};
            mi::Vector<double> y_ell{ctx, "y3", 3};
            csr.linop()->apply(x.linop(), y_csr.linop());
            coo.linop()->apply(x.linop(), y_coo.linop());
            ell.linop()->apply(x.linop(), y_ell.linop());
            for (gko::size_type i = 0; i < 3; ++i) {
                expect(y_coo.at(i) == y_csr.at(i));
                expect(y_ell.at(i) == y_csr.at(i));
            }
        };

        test("num_nonzeros matches the pattern") = [] {
            mi::Context ctx;
            mi::SparseMatrix<double> a{ctx, "A", make_pattern()};
            expect(a.num_nonzeros() == 5_u);
        };

        test("format reports the constructed format") = [] {
            mi::Context ctx;
            mi::SparseMatrix<double> a{ctx, "A", make_pattern(), {.format = mi::SparseFormat::Coo}};
            expect(a.format() == mi::SparseFormat::Coo);
        };

        test("revision starts at zero") = [] {
            mi::Context ctx;
            mi::SparseMatrix<double> a{ctx, "A", make_pattern()};
            expect(a.revision() == 0_u);
        };
    };

    suite<"SparseMatrix update"> update_suite = [] {
        test("update_values from a same-structure pattern changes apply and bumps revision") = [] {
            mi::Context ctx;
            mi::SparseMatrix<double> a{ctx, "A", make_pattern()};
            const auto revision_before = a.revision();

            mi::SparsityPattern<double> doubled{gko::dim<2>{3, 3}};
            doubled.add_value(0, 0, 4.0);
            doubled.add_value(0, 2, 2.0);
            doubled.add_value(1, 1, 6.0);
            doubled.add_value(2, 0, 8.0);
            doubled.add_value(2, 2, 10.0);
            a.update_values(doubled);

            expect(a.revision() > revision_before);
            mi::Vector<double> x{ctx, "x", {1.0, 1.0, 1.0}};
            mi::Vector<double> y{ctx, "y", 3};
            a.linop()->apply(x.linop(), y.linop());
            expect(y.at(0) == 6.0_d);  // 4 + 2
            expect(y.at(2) == 18.0_d); // 8 + 10
        };

        test("update_values from a different-structure pattern throws and keeps revision") = [] {
            mi::Context ctx;
            mi::SparseMatrix<double> a{ctx, "A", make_pattern()};
            const auto revision_before = a.revision();
            mi::SparsityPattern<double> fewer{gko::dim<2>{3, 3}};
            fewer.add_value(0, 0, 1.0);
            expect(throws<std::invalid_argument>([&] { a.update_values(fewer); }));
            expect(a.revision() == revision_before);
        };

        test("update_values from a span refills the value array") = [] {
            mi::Context ctx;
            mi::SparseMatrix<double> a{ctx, "A", make_pattern()};
            // CSR stores values row-major by sorted column: 2,1,3,4,5
            std::array<double, 5> new_values{20.0, 10.0, 30.0, 40.0, 50.0};
            a.update_values(std::span<const double>{new_values});
            mi::Vector<double> x{ctx, "x", {1.0, 1.0, 1.0}};
            mi::Vector<double> y{ctx, "y", 3};
            a.linop()->apply(x.linop(), y.linop());
            expect(y.at(0) == 30.0_d); // 20 + 10
            expect(y.at(1) == 30.0_d);
            expect(y.at(2) == 90.0_d); // 40 + 50
        };

        test("update_values span of the wrong length throws") = [] {
            mi::Context ctx;
            mi::SparseMatrix<double> a{ctx, "A", make_pattern()};
            std::array<double, 3> wrong{1.0, 2.0, 3.0};
            expect(throws<std::invalid_argument>([&] { a.update_values(std::span<const double>{wrong}); }));
        };
    };

    suite<"SparseMatrix format interop"> interop_suite = [] {
        test("convert_to preserves the apply result") = [] {
            mi::Context ctx;
            mi::SparseMatrix<double> csr{ctx, "A", make_pattern()};
            auto coo = csr.convert_to(ctx, "A_coo", mi::SparseFormat::Coo);
            expect(coo.format() == mi::SparseFormat::Coo);

            mi::Vector<double> x{ctx, "x", {1.0, 2.0, 3.0}};
            mi::Vector<double> y_csr{ctx, "y1", 3};
            mi::Vector<double> y_coo{ctx, "y2", 3};
            csr.linop()->apply(x.linop(), y_csr.linop());
            coo.linop()->apply(x.linop(), y_coo.linop());
            for (gko::size_type i = 0; i < 3; ++i) {
                expect(y_coo.at(i) == y_csr.at(i));
            }
        };

        test("CSR accessors expose canonical arrays and the value write path changes apply") = [] {
            mi::Context ctx;
            mi::SparseMatrix<double> a{ctx, "A", make_pattern()};
            // row_ptrs has length rows+1, last entry == nnz.
            expect(a.row_ptrs()[0] == 0_i);
            expect(a.row_ptrs()[3] == 5_i);
            // first stored entry is (0,0) = 2.
            expect(a.col_idxs()[0] == 0_i);
            expect(a.values()[0] == 2.0_d);

            a.values()[0] = 20.0;
            mi::Vector<double> x{ctx, "x", {1.0, 0.0, 0.0}};
            mi::Vector<double> y{ctx, "y", 3};
            a.linop()->apply(x.linop(), y.linop());
            expect(y.at(0) == 20.0_d);
        };

        test("implicitly converts to shared_ptr<const gko::LinOp>") = [] {
            mi::Context ctx;
            mi::SparseMatrix<double> a{ctx, "A", make_pattern()};
            expect(accepts_linop(a));
        };
    };

    return 0;
}
