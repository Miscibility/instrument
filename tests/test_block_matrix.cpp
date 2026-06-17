// test_block_matrix.cpp -- unit tests for BlockMatrix and BlockVector: the grid
// of type-erased sub-matrices and the block matrix-vector product y = A*x.
// boost/ut.
//
// Every member of BlockMatrix/BlockVector currently throws
// std::runtime_error{"not implemented"}, so every test below fails until
// tdd-3-implement fills the bodies in.

#include "instrument/block_matrix.hpp"
#include "instrument/csr_matrix.hpp"
#include "instrument/diagonal_matrix.hpp"
#include "instrument/matrix.hpp"
#include "instrument/sparsity_pattern.hpp"
#include "instrument/vector.hpp"
#include "instrument/zero_matrix.hpp"

#include <boost/ut.hpp>
#include <stdexcept>
#include <typeinfo>

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

    suite<"BlockMatrix"> block_suite = [] {
        // The 2x2 heterogeneous reference matrix used by several tests:
        //   A00 = [[1,2],[3,4]] (dense)   A01 = 0 (2x2)
        //   A10 = diag(5,6)               A11 = [[7,8],[9,10]] (dense)
        // Assembled 4x4:
        //   1 2 0 0
        //   3 4 0 0
        //   5 0 7 8
        //   0 6 9 10
        auto make_hetero = [] {
            mi::BlockMatrix<double> a(2, 2);
            a.set_block(0, 0, mi::DenseMatrix<double>{{1, 2}, {3, 4}});
            a.set_block(0, 1, mi::ZeroMatrix<double>(2, 2));
            a.set_block(1, 0, mi::DiagonalMatrix<double>{5, 6});
            a.set_block(1, 1, mi::DenseMatrix<double>{{7, 8}, {9, 10}});
            return a;
        };

        test("heterogeneous 2x2 block product equals the assembled dense product") = [&] {
            auto a = make_hetero();
            mi::BlockVector<double> x{mi::Vector<double>{1, 2}, mi::Vector<double>{3, 4}};
            // Reference (assembled A * [1,2,3,4]):
            //   row0: 1+4=5, row1: 3+8=11, row2: 5+21+32=58, row3: 12+27+40=79
            auto y = a.multiply(x);
            expect(y.block_count() == 2_u);
            expect(close(y.block(0)[0], 5.0));
            expect(close(y.block(0)[1], 11.0));
            expect(close(y.block(1)[0], 58.0));
            expect(close(y.block(1)[1], 79.0));
        };

        test("operator* matches multiply") = [&] {
            auto a = make_hetero();
            mi::BlockVector<double> x{mi::Vector<double>{1, 2}, mi::Vector<double>{3, 4}};
            auto y = a * x;
            expect(y.block_count() == 2_u);
            expect(close(y.block(0)[0], 5.0));
            expect(close(y.block(1)[1], 79.0));
        };

        test("ZeroMatrix block contributes nothing") = [] {
            // 1x2 grid: A00 dense, A01 zero. Result is A00*x0 only.
            mi::BlockMatrix<double> a(1, 2);
            a.set_block(0, 0, mi::DenseMatrix<double>{{1, 2}, {3, 4}});
            a.set_block(0, 1, mi::ZeroMatrix<double>(2, 2));
            mi::BlockVector<double> x{mi::Vector<double>{1, 1}, mi::Vector<double>{9, 9}};
            auto y = a.multiply(x);
            // A00*[1,1] = [3,7]; the zero block adds nothing despite x1 = [9,9].
            expect(y.block_count() == 1_u);
            expect(close(y.block(0)[0], 3.0));
            expect(close(y.block(0)[1], 7.0));
        };

        test("accumulation across block-columns sums both contributions") = [] {
            // 1x2 grid: A00 = [[1,2],[3,4]], A01 = [[5,6],[7,8]].
            mi::BlockMatrix<double> a(1, 2);
            a.set_block(0, 0, mi::DenseMatrix<double>{{1, 2}, {3, 4}});
            a.set_block(0, 1, mi::DenseMatrix<double>{{5, 6}, {7, 8}});
            mi::BlockVector<double> x{mi::Vector<double>{1, 1}, mi::Vector<double>{1, 1}};
            auto y = a.multiply(x);
            // A00*[1,1] = [3,7]; A01*[1,1] = [11,15]; sum = [14,22].
            expect(close(y.block(0)[0], 14.0));
            expect(close(y.block(0)[1], 22.0));
        };

        test("gemv: alpha/beta applied once per destination block, not per source block") = [] {
            // 1x2 grid (two source blocks into one destination block).
            mi::BlockMatrix<double> a(1, 2);
            a.set_block(0, 0, mi::DenseMatrix<double>{{1, 2}, {3, 4}});
            a.set_block(0, 1, mi::DenseMatrix<double>{{5, 6}, {7, 8}});
            mi::BlockVector<double> x{mi::Vector<double>{1, 1}, mi::Vector<double>{1, 1}};
            mi::BlockVector<double> y{mi::Vector<double>{10, 20}};
            a.multiply_into(x, y, 2.0, 3.0);
            // 2*(A00*x0 + A01*x1) + 3*y = 2*[14,22] + 3*[10,20] = [28,44]+[30,60] = [58,104].
            // (If beta were applied per source block it would add 3*y twice -> wrong.)
            expect(close(y.block(0)[0], 58.0));
            expect(close(y.block(0)[1], 104.0));
        };

        test("transposed product matches the assembled dense transpose product") = [&] {
            auto a = make_hetero();
            // x has block-row heights [2,2] (length rows() = 4).
            mi::BlockVector<double> x{mi::Vector<double>{1, 2}, mi::Vector<double>{3, 4}};
            auto y = a.multiply(x, mi::Transpose::Transposed);
            // A^T * [1,2,3,4]: cols of A dotted with x ->
            //   [1+6+15, 2+8+24, 21+36, 24+40] = [22,34,57,64]. Result blocks [2,2].
            expect(y.block_count() == 2_u);
            expect(close(y.block(0)[0], 22.0));
            expect(close(y.block(0)[1], 34.0));
            expect(close(y.block(1)[0], 57.0));
            expect(close(y.block(1)[1], 64.0));
        };

        test("block_as recovers the underlying concrete matrix") = [&] {
            auto a = make_hetero();
            auto& dense = a.block_as<mi::DenseMatrix<double>>(0, 0);
            expect(close(dense(0, 0), 1.0));
            expect(close(dense(0, 1), 2.0));
            expect(close(dense(1, 0), 3.0));
            expect(close(dense(1, 1), 4.0));
        };

        test("block_as with the wrong type throws bad_cast") = [&] {
            auto a = make_hetero();
            expect(throws<std::bad_cast>([&] { (void)a.block_as<mi::DiagonalMatrix<double>>(0, 0); }));
        };

        test("erased block exposes correct rows()/columns()") = [] {
            mi::BlockMatrix<double> a(1, 1);
            a.set_block(0, 0, mi::DenseMatrix<double>(2, 3)); // 2x3
            const auto& blk = a.block(0, 0);
            expect(blk.rows() == 2_u);
            expect(blk.columns() == 3_u);
        };

        test("total rows()/columns() sum the block dimensions") = [&] {
            auto a = make_hetero();
            expect(a.rows() == 4_u);
            expect(a.columns() == 4_u);
        };

        test("unset cell used in a product throws logic_error") = [] {
            mi::BlockMatrix<double> a(1, 1); // cell (0,0) never set
            mi::BlockVector<double> x{mi::Vector<double>{1, 2}};
            expect(throws<std::logic_error>([&] { (void)a.multiply(x); }));
        };

        test("inconsistent block-row heights throw invalid_argument") = [] {
            // Row 0 mixes a 2-row and a 3-row block -> inconsistent height.
            mi::BlockMatrix<double> a(1, 2);
            a.set_block(0, 0, mi::DenseMatrix<double>(2, 2));
            a.set_block(0, 1, mi::DenseMatrix<double>(3, 2));
            mi::BlockVector<double> x{mi::Vector<double>{1, 2}, mi::Vector<double>{3, 4}};
            expect(throws<std::invalid_argument>([&] { (void)a.multiply(x); }));
        };

        test("inconsistent block-column widths throw invalid_argument") = [] {
            // Column 0 mixes a 2-col and a 3-col block -> inconsistent width.
            mi::BlockMatrix<double> a(2, 1);
            a.set_block(0, 0, mi::DenseMatrix<double>(2, 2));
            a.set_block(1, 0, mi::DenseMatrix<double>(2, 3));
            mi::BlockVector<double> x{mi::Vector<double>{1, 2}};
            expect(throws<std::invalid_argument>([&] { (void)a.multiply(x); }));
        };

        test("BlockVector with the wrong block count throws invalid_argument") = [&] {
            auto a = make_hetero(); // expects 2 operand blocks
            mi::BlockVector<double> x{mi::Vector<double>{1, 2}}; // only 1 block
            expect(throws<std::invalid_argument>([&] { (void)a.multiply(x); }));
        };

        test("BlockVector with a wrong sub-vector size throws invalid_argument") = [&] {
            auto a = make_hetero(); // expects operand blocks of length 2,2
            mi::BlockVector<double> x{mi::Vector<double>{1, 2, 3}, mi::Vector<double>{3, 4}};
            expect(throws<std::invalid_argument>([&] { (void)a.multiply(x); }));
        };

        test("non-square 2x3 block grid works for None") = [] {
            // Each block 1x1 (diagonal of order 1 stands in as a scalar block).
            //   A = [[1,2,3],[4,5,6]] assembled.
            mi::BlockMatrix<double> a(2, 3);
            a.set_block(0, 0, mi::DenseMatrix<double>{{1}});
            a.set_block(0, 1, mi::DenseMatrix<double>{{2}});
            a.set_block(0, 2, mi::DenseMatrix<double>{{3}});
            a.set_block(1, 0, mi::DenseMatrix<double>{{4}});
            a.set_block(1, 1, mi::DenseMatrix<double>{{5}});
            a.set_block(1, 2, mi::DenseMatrix<double>{{6}});
            expect(a.rows() == 2_u);
            expect(a.columns() == 3_u);
            mi::BlockVector<double> x{mi::Vector<double>{1}, mi::Vector<double>{1}, mi::Vector<double>{1}};
            auto y = a.multiply(x);
            // row0: 1+2+3=6, row1: 4+5+6=15
            expect(y.block_count() == 2_u);
            expect(close(y.block(0)[0], 6.0));
            expect(close(y.block(1)[0], 15.0));
        };

        test("non-square 2x3 block grid works for Transposed") = [] {
            mi::BlockMatrix<double> a(2, 3);
            a.set_block(0, 0, mi::DenseMatrix<double>{{1}});
            a.set_block(0, 1, mi::DenseMatrix<double>{{2}});
            a.set_block(0, 2, mi::DenseMatrix<double>{{3}});
            a.set_block(1, 0, mi::DenseMatrix<double>{{4}});
            a.set_block(1, 1, mi::DenseMatrix<double>{{5}});
            a.set_block(1, 2, mi::DenseMatrix<double>{{6}});
            // A^T is 3x2; operand has block-row heights [1,1] (length rows()=2).
            mi::BlockVector<double> x{mi::Vector<double>{1}, mi::Vector<double>{1}};
            auto y = a.multiply(x, mi::Transpose::Transposed);
            // A^T * [1,1] = [1+4, 2+5, 3+6] = [5,7,9]; result has 3 blocks.
            expect(y.block_count() == 3_u);
            expect(close(y.block(0)[0], 5.0));
            expect(close(y.block(1)[0], 7.0));
            expect(close(y.block(2)[0], 9.0));
        };

        test("a CsrMatrix block participates correctly") = [] {
            // 2x2 grid: A00 = CSR diag(1,2), A01/A10 = zero, A11 = [[1,1],[1,1]].
            mi::SparsityPattern<double> pattern(2, 2, {{.row=0, .col=0, .value=1.0}, {.row=1, .col=1, .value=2.0}});
            mi::BlockMatrix<double> a(2, 2);
            a.set_block(0, 0, mi::CsrMatrix<double>(pattern));
            a.set_block(0, 1, mi::ZeroMatrix<double>(2, 2));
            a.set_block(1, 0, mi::ZeroMatrix<double>(2, 2));
            a.set_block(1, 1, mi::DenseMatrix<double>{{1, 1}, {1, 1}});
            mi::BlockVector<double> x{mi::Vector<double>{3, 4}, mi::Vector<double>{5, 6}};
            auto y = a.multiply(x);
            // y0 = CSR*[3,4] = [3,8]; y1 = [[1,1],[1,1]]*[5,6] = [11,11].
            expect(close(y.block(0)[0], 3.0));
            expect(close(y.block(0)[1], 8.0));
            expect(close(y.block(1)[0], 11.0));
            expect(close(y.block(1)[1], 11.0));
        };
    };

    suite<"BlockMatrix float"> block_float_suite = [] {
        test("float instantiation: heterogeneous 2x2 product") = [] {
            mi::BlockMatrix<float> a(2, 2);
            a.set_block(0, 0, mi::DenseMatrix<float>{{1, 2}, {3, 4}});
            a.set_block(0, 1, mi::ZeroMatrix<float>(2, 2));
            a.set_block(1, 0, mi::DiagonalMatrix<float>{5, 6});
            a.set_block(1, 1, mi::DenseMatrix<float>{{7, 8}, {9, 10}});
            mi::BlockVector<float> x{mi::Vector<float>{1, 2}, mi::Vector<float>{3, 4}};
            auto y = a.multiply(x);
            expect(close(y.block(0)[0], 5.0F, 1e-4F));
            expect(close(y.block(0)[1], 11.0F, 1e-4F));
            expect(close(y.block(1)[0], 58.0F, 1e-4F));
            expect(close(y.block(1)[1], 79.0F, 1e-4F));
        };
    };

    suite<"BlockVector"> block_vector_suite = [] {
        test("block_count, block access, and size") = [] {
            mi::BlockVector<double> v{mi::Vector<double>{1, 2}, mi::Vector<double>{3, 4, 5}};
            expect(v.block_count() == 2_u);
            expect(v.size() == 5_u); // 2 + 3
            expect(close(v.block(0)[0], 1.0));
            expect(close(v.block(1)[2], 5.0));
        };

        test("out-of-range block access throws out_of_range") = [] {
            mi::BlockVector<double> v{mi::Vector<double>{1, 2}};
            expect(throws<std::out_of_range>([&] { (void)v.block(5); }));
        };

        test("push_block appends a sub-vector") = [] {
            mi::BlockVector<double> v;
            v.push_block(mi::Vector<double>{1, 2});
            v.push_block(mi::Vector<double>{3});
            expect(v.block_count() == 2_u);
            expect(v.size() == 3_u);
            expect(close(v.block(1)[0], 3.0));
        };
    };
}
