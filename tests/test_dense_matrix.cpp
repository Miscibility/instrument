// test_dense_matrix.cpp -- unit tests for miscibility::instrument DenseMatrix.

#include "instrument/context.hpp"
#include "instrument/dense_matrix.hpp"
#include "instrument/vector.hpp"

#include <boost/ut.hpp>
#include <ginkgo/ginkgo.hpp>
#include <memory>
#include <vector>

namespace mi = miscibility::instrument;

namespace {

bool accepts_linop(const std::shared_ptr<const gko::LinOp>& op) { return op != nullptr; }

} // namespace

int main()
{
    using namespace boost::ut;

    suite<"DenseMatrix construction"> construction_suite = [] {
        test("nested-init constructor fills row-major and reports its shape") = [] {
            mi::Context ctx;
            mi::DenseMatrix<double> a{ctx, "A", {{1.0, 2.0}, {3.0, 4.0}}};
            expect(a.shape() == gko::dim<2>{2, 2});
            expect(a.rows() == 2_u);
            expect(a.cols() == 2_u);
            expect(a.at(0, 0) == 1.0_d);
            expect(a.at(0, 1) == 2.0_d);
            expect(a.at(1, 0) == 3.0_d);
            expect(a.at(1, 1) == 4.0_d);
        };

        test("ragged nested-init constructor throws std::invalid_argument") = [] {
            mi::Context ctx;
            expect(throws<std::invalid_argument>([&] { mi::DenseMatrix<double> a{ctx, "A", {{1.0, 2.0}, {3.0}}}; }));
        };

        test("shape constructor zero-fills") = [] {
            mi::Context ctx;
            mi::DenseMatrix<double> a{ctx, "A", gko::dim<2>{2, 3}};
            expect(a.shape() == gko::dim<2>{2, 3});
            for (mi::DenseMatrix<double>::size_type i=0; i<a.rows();++i){
                for (mi::DenseMatrix<double>::size_type j=0; j<a.cols();++j){
                    expect(a.at(i,j) == 0.0_d);
                }
            }
        };

        test("matrix_data constructor reads the entries") = [] {
            mi::Context ctx;
            gko::matrix_data<double, int> data{gko::dim<2>{2, 2}, {{.row=0, .col=0, .val=5.0}, {.row=1, .col=1, .val=6.0}}};
            mi::DenseMatrix<double> a{ctx, "A", data};
            expect(a.at(0, 0) == 5.0_d);
            expect(a.at(1, 1) == 6.0_d);
            expect(a.at(0, 1) == 0.0_d);
        };
    };

    suite<"DenseMatrix element access"> access_suite = [] {
        test("fill and at write round-trip") = [] {
            mi::Context ctx;
            mi::DenseMatrix<double> a{ctx, "A", gko::dim<2>{2, 2}};
            a.fill(3.0);
            expect(a.at(0, 1) == 3.0_d);
            a.at(0, 1) = 9.0;
            expect(a.at(0, 1) == 9.0_d);
        };

        test("row-major layout: data()[i*stride()+j] equals at(i,j)") = [] {
            mi::Context ctx;
            mi::DenseMatrix<double> a{ctx, "A", {{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}}};
            for (gko::size_type i = 0; i < a.rows(); ++i) {
                for (gko::size_type j = 0; j < a.cols(); ++j) {
                    expect(a.data()[(i * a.stride()) + j] == a.at(i, j));
                }
            }
        };
    };

    suite<"DenseMatrix products"> product_suite = [] {
        test("apply(x,y) computes the matrix-vector product") = [] {
            mi::Context ctx;
            mi::DenseMatrix<double> a{ctx, "A", {{1.0, 2.0}, {3.0, 4.0}}};
            mi::Vector<double> x{ctx, "x", {1.0, 1.0}};
            mi::Vector<double> y{ctx, "y", 2};
            a.apply(x, y);
            expect(y.at(0) == 3.0_d);
            expect(y.at(1) == 7.0_d);
        };

        test("advanced apply computes alpha*A*x + beta*y") = [] {
            mi::Context ctx;
            mi::DenseMatrix<double> a{ctx, "A", {{1.0, 2.0}, {3.0, 4.0}}};
            mi::Vector<double> x{ctx, "x", {1.0, 1.0}};
            mi::Vector<double> y{ctx, "y", {3.0, 7.0}};
            // residual: A*x - b, with b held in y, gives 0
            a.apply(1.0, x, -1.0, y);
            expect(y.at(0) == 0.0_d);
            expect(y.at(1) == 0.0_d);
        };

        test("matrix-matrix apply computes the product") = [] {
            mi::Context ctx;
            mi::DenseMatrix<double> a{ctx, "A", {{1.0, 2.0}, {3.0, 4.0}}};
            mi::DenseMatrix<double> x{ctx, "X", {{1.0, 0.0}, {0.0, 1.0}}};
            mi::DenseMatrix<double> y{ctx, "Y", gko::dim<2>{2, 2}};
            a.apply(x, y);
            expect(y.at(0, 0) == 1.0_d);
            expect(y.at(0, 1) == 2.0_d);
            expect(y.at(1, 0) == 3.0_d);
            expect(y.at(1, 1) == 4.0_d);
        };
    };

    suite<"DenseMatrix interop"> interop_suite = [] {
        test("copy_values extracts row-major values regardless of stride") = [] {
            mi::Context ctx;
            mi::DenseMatrix<double> a{ctx, "A", {{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}}};
            std::vector<double> out(a.rows() * a.cols());
            a.copy_values(out.data());
            gko::size_type k = 0;
            for (gko::size_type i = 0; i < a.rows(); ++i) {
                for (gko::size_type j = 0; j < a.cols(); ++j) {
                    expect(out[k++] == a.at(i, j));
                }
            }
        };

        test("to_matrix_data round-trips through a rebuilt matrix") = [] {
            mi::Context ctx;
            mi::DenseMatrix<double> a{ctx, "A", {{1.0, 2.0}, {3.0, 4.0}}};
            auto data = a.to_matrix_data();
            mi::DenseMatrix<double> b{ctx, "B", data};
            expect(b.at(0, 0) == 1.0_d);
            expect(b.at(0, 1) == 2.0_d);
            expect(b.at(1, 0) == 3.0_d);
            expect(b.at(1, 1) == 4.0_d);
        };

        test("scale and add_scaled mutate the entries") = [] {
            mi::Context ctx;
            mi::DenseMatrix<double> a{ctx, "A", {{1.0, 2.0}, {3.0, 4.0}}};
            mi::DenseMatrix<double> b{ctx, "B", {{1.0, 1.0}, {1.0, 1.0}}};
            a.scale(2.0);
            expect(a.at(0, 0) == 2.0_d);
            expect(a.at(1, 1) == 8.0_d);
            a.add_scaled(1.0, b);
            expect(a.at(0, 0) == 3.0_d);
            expect(a.at(1, 1) == 9.0_d);
        };

        test("implicitly converts to shared_ptr<const gko::LinOp>") = [] {
            mi::Context ctx;
            mi::DenseMatrix<double> a{ctx, "A", gko::dim<2>{2, 2}};
            expect(accepts_linop(a));
        };
    };

    return 0;
}
