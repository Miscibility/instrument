// test_block_vector_blas1.cpp -- unit tests for the BLAS-1 surface on BlockVector:
// dot, euclidean_norm, absolute_sum, add_scaled, scale, copy, fill, and the
// convenience operators (*=, /=, +=, -=). Pure block-wise orchestration over the
// per-block Vector kernels. boost/ut.
//
// Every new BlockVector BLAS-1 member is currently a stub: the throwing members
// (dot, add_scaled, copy) throw std::runtime_error{"not implemented"}; the
// noexcept members (euclidean_norm, absolute_sum, scale, fill) return a wrong
// sentinel or do nothing. So every test below fails until tdd-3-implement fills
// the bodies in.

#include "instrument/block_matrix.hpp"
#include "instrument/vector.hpp"

#include <boost/ut.hpp>
#include <cmath>
#include <stdexcept>

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

    suite<"BlockVector BLAS-1"> block_vector_blas1_suite = [] {
        // dot --------------------------------------------------------------
        test("dot sums the per-block inner products") = [] {
            mi::BlockVector<double> a{mi::Vector<double>{1, 2}, mi::Vector<double>{3}};
            mi::BlockVector<double> b{mi::Vector<double>{4, 5}, mi::Vector<double>{6}};
            // 1*4 + 2*5 + 3*6 = 4 + 10 + 18 = 32.
            expect(close(a.dot(b), 32.0));
        };

        // euclidean_norm ---------------------------------------------------
        test("euclidean_norm is the norm of the concatenation") = [] {
            mi::BlockVector<double> v{mi::Vector<double>{3, 4}, mi::Vector<double>{12}};
            // sqrt(9 + 16 + 144) = sqrt(169) = 13.
            expect(close(v.euclidean_norm(), 13.0));
        };

        test("euclidean_norm of an empty block vector is zero") = [] {
            mi::BlockVector<double> v; // zero blocks
            expect(close(v.euclidean_norm(), 0.0));
        };

        // absolute_sum -----------------------------------------------------
        test("absolute_sum sums the per-block absolute sums") = [] {
            mi::BlockVector<double> v{mi::Vector<double>{-1, 2}, mi::Vector<double>{-3}};
            // |−1| + |2| + |−3| = 6.
            expect(close(v.absolute_sum(), 6.0));
        };

        // scale ------------------------------------------------------------
        test("scale multiplies every block element in place and returns *this") = [] {
            mi::BlockVector<double> v{mi::Vector<double>{1, 2}, mi::Vector<double>{3}};
            auto& ref = v.scale(2.0);
            expect(&ref == &v);
            expect(close(v.block(0)[0], 2.0));
            expect(close(v.block(0)[1], 4.0));
            expect(close(v.block(1)[0], 6.0));
        };

        // add_scaled -------------------------------------------------------
        test("add_scaled performs block-wise axpy and returns *this") = [] {
            mi::BlockVector<double> y{mi::Vector<double>{1, 1}, mi::Vector<double>{1}};
            mi::BlockVector<double> x{mi::Vector<double>{2, 4}, mi::Vector<double>{6}};
            auto& ref = y.add_scaled(0.5, x);
            expect(&ref == &y);
            // y + 0.5*x = {1+1, 1+2} , {1+3} = {2,3},{4}.
            expect(close(y.block(0)[0], 2.0));
            expect(close(y.block(0)[1], 3.0));
            expect(close(y.block(1)[0], 4.0));
        };

        // copy -------------------------------------------------------------
        test("copy overwrites values, returns *this, and allocates nothing") = [] {
            mi::BlockVector<double> dst{mi::Vector<double>{0, 0}, mi::Vector<double>{0}};
            mi::BlockVector<double> src{mi::Vector<double>{1, 2}, mi::Vector<double>{3}};
            auto& ref = dst.copy(src);
            expect(&ref == &dst);
            expect(dst.block_count() == 2_u);
            expect(dst.block(0).size() == 2_u); // sizes unchanged: no reallocation
            expect(dst.block(1).size() == 1_u);
            expect(close(dst.block(0)[0], 1.0));
            expect(close(dst.block(0)[1], 2.0));
            expect(close(dst.block(1)[0], 3.0));
        };

        // fill -------------------------------------------------------------
        test("fill sets every element of every block") = [] {
            mi::BlockVector<double> v{mi::Vector<double>{1, 2}, mi::Vector<double>{3}};
            v.fill(7.0);
            expect(close(v.block(0)[0], 7.0));
            expect(close(v.block(0)[1], 7.0));
            expect(close(v.block(1)[0], 7.0));
        };

        // operators --------------------------------------------------------
        test("operator*= agrees with scale") = [] {
            mi::BlockVector<double> v{mi::Vector<double>{1, 2}, mi::Vector<double>{3}};
            v *= 2.0;
            expect(close(v.block(0)[0], 2.0));
            expect(close(v.block(0)[1], 4.0));
            expect(close(v.block(1)[0], 6.0));
        };

        test("operator/= agrees with scale by the reciprocal") = [] {
            mi::BlockVector<double> v{mi::Vector<double>{2, 4}, mi::Vector<double>{6}};
            v /= 2.0;
            expect(close(v.block(0)[0], 1.0));
            expect(close(v.block(0)[1], 2.0));
            expect(close(v.block(1)[0], 3.0));
        };

        test("operator+= agrees with add_scaled(1, x)") = [] {
            mi::BlockVector<double> y{mi::Vector<double>{1, 1}, mi::Vector<double>{1}};
            mi::BlockVector<double> x{mi::Vector<double>{2, 4}, mi::Vector<double>{6}};
            y += x;
            expect(close(y.block(0)[0], 3.0));
            expect(close(y.block(0)[1], 5.0));
            expect(close(y.block(1)[0], 7.0));
        };

        test("operator-= agrees with add_scaled(-1, x)") = [] {
            mi::BlockVector<double> y{mi::Vector<double>{1, 1}, mi::Vector<double>{1}};
            mi::BlockVector<double> x{mi::Vector<double>{2, 4}, mi::Vector<double>{6}};
            y -= x;
            expect(close(y.block(0)[0], -1.0));
            expect(close(y.block(0)[1], -3.0));
            expect(close(y.block(1)[0], -5.0));
        };

        // conformance: block-count mismatch --------------------------------
        test("dot with a block-count mismatch throws invalid_argument") = [] {
            mi::BlockVector<double> a{mi::Vector<double>{1, 2}, mi::Vector<double>{3}};
            mi::BlockVector<double> b{mi::Vector<double>{1}, mi::Vector<double>{2}, mi::Vector<double>{3}};
            expect(throws<std::invalid_argument>([&] { (void)a.dot(b); }));
        };

        test("add_scaled with a block-count mismatch throws invalid_argument") = [] {
            mi::BlockVector<double> a{mi::Vector<double>{1, 2}, mi::Vector<double>{3}};
            mi::BlockVector<double> b{mi::Vector<double>{1}, mi::Vector<double>{2}, mi::Vector<double>{3}};
            expect(throws<std::invalid_argument>([&] { (void)a.add_scaled(1.0, b); }));
        };

        test("copy with a block-count mismatch throws invalid_argument") = [] {
            mi::BlockVector<double> a{mi::Vector<double>{1, 2}, mi::Vector<double>{3}};
            mi::BlockVector<double> b{mi::Vector<double>{1}, mi::Vector<double>{2}, mi::Vector<double>{3}};
            expect(throws<std::invalid_argument>([&] { (void)a.copy(b); }));
        };

        // conformance: per-block size mismatch -----------------------------
        test("dot with a per-block size mismatch throws invalid_argument") = [] {
            mi::BlockVector<double> a{mi::Vector<double>{1, 2}, mi::Vector<double>{3}};
            mi::BlockVector<double> b{mi::Vector<double>{1, 2, 3}, mi::Vector<double>{4}};
            expect(throws<std::invalid_argument>([&] { (void)a.dot(b); }));
        };

        test("add_scaled with a per-block size mismatch throws invalid_argument") = [] {
            mi::BlockVector<double> a{mi::Vector<double>{1, 2}, mi::Vector<double>{3}};
            mi::BlockVector<double> b{mi::Vector<double>{1, 2, 3}, mi::Vector<double>{4}};
            expect(throws<std::invalid_argument>([&] { (void)a.add_scaled(1.0, b); }));
        };

        test("copy with a per-block size mismatch throws invalid_argument") = [] {
            mi::BlockVector<double> a{mi::Vector<double>{1, 2}, mi::Vector<double>{3}};
            mi::BlockVector<double> b{mi::Vector<double>{1, 2, 3}, mi::Vector<double>{4}};
            expect(throws<std::invalid_argument>([&] { (void)a.copy(b); }));
        };

        // equivalence to a flat Vector -------------------------------------
        test("dot, euclidean_norm, absolute_sum agree with the flat concatenation") = [] {
            mi::BlockVector<double> a{mi::Vector<double>{1, -2, 3}, mi::Vector<double>{4, 5}};
            mi::BlockVector<double> b{mi::Vector<double>{6, 7, -8}, mi::Vector<double>{9, 10}};
            // Flat concatenations of the blocks.
            mi::Vector<double> fa{1, -2, 3, 4, 5};
            mi::Vector<double> fb{6, 7, -8, 9, 10};
            expect(close(a.dot(b), fa.dot(fb)));
            expect(close(a.euclidean_norm(), fa.euclidean_norm()));
            expect(close(a.absolute_sum(), fa.absolute_sum()));
        };
    };

    suite<"BlockVector BLAS-1 float"> block_vector_blas1_float_suite = [] {
        test("float instantiation: dot and euclidean_norm") = [] {
            mi::BlockVector<float> a{mi::Vector<float>{1, 2}, mi::Vector<float>{3}};
            mi::BlockVector<float> b{mi::Vector<float>{4, 5}, mi::Vector<float>{6}};
            expect(close(a.dot(b), 32.0F, 1e-4F));
            mi::BlockVector<float> v{mi::Vector<float>{3, 4}, mi::Vector<float>{12}};
            expect(close(v.euclidean_norm(), 13.0F, 1e-4F));
        };
    };
}
