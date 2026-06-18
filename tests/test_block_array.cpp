// test_block_array.cpp -- unit tests for BlockArray: construction/access plus the
// block-wise BLAS-1-style surface (sum, absolute_sum, add_scaled, scale, copy,
// fill, and the *=, /=, +=, -= operators). Pure block-wise orchestration over the
// per-block Array kernels. boost/ut. See specs/block-array.md.
//
// Construction/access is real; every operation is currently a stub. The throwing
// members (add_scaled, copy) run check_conformable first, so the conformability
// tests pin the throwing contract and pass now; every other operation throws
// std::runtime_error{"not implemented"} (the noexcept members turn that into
// std::terminate). So the happy-path cases below FAIL until tdd-3-implement fills
// in the bodies.

#include "instrument/array.hpp"
#include "instrument/block_array.hpp"

#include <boost/ut.hpp>
#include <cmath>
#include <stdexcept>
#include <utility>

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

    suite<"BlockArray"> block_array_suite = [] {
        // -- construction / access ---------------------------------------------

        test("braced construction: block_count, size, contents, checked access") = [] {
            mi::BlockArray<double> a{mi::Array<double>{1, 2}, mi::Array<double>{3, 4, 5}};
            expect(a.block_count() == 2_u);
            expect(a.size() == 5_u) << "total length is the sum of the block sizes";
            expect(close(a.block(0)[0], 1.0));
            expect(close(a.block(0)[1], 2.0));
            expect(close(a.block(1)[0], 3.0));
            expect(close(a.block(1)[2], 5.0));
            expect(throws<std::out_of_range>([&] { (void)a.block(2); }));
        };

        test("push_block grows block_count and size") = [] {
            mi::BlockArray<double> a;
            expect(a.block_count() == 0_u);
            expect(a.size() == 0_u);
            a.push_block(mi::Array<double>{1, 2});
            a.push_block(mi::Array<double>{3});
            expect(a.block_count() == 2_u);
            expect(a.size() == 3_u);
        };

        // -- sum ---------------------------------------------------------------

        test("sum sums the per-block sums") = [] {
            mi::BlockArray<double> a{mi::Array<double>{1, 2}, mi::Array<double>{3, 4}};
            expect(close(a.sum(), 10.0));
        };

        test("sum of an empty block array is zero") = [] {
            mi::BlockArray<double> v; // zero blocks
            expect(close(v.sum(), 0.0));
        };

        test("sum with cancellation across blocks matches the flattened sum") = [] {
            mi::BlockArray<double> v{mi::Array<double>{1, -1}, mi::Array<double>{2, -2}};
            expect(close(v.sum(), 0.0));
            // absolute_sum on the same input would be 6 -- distinguishes sum from asum.
            expect(close(v.absolute_sum(), 6.0));
        };

        // -- absolute_sum ------------------------------------------------------

        test("absolute_sum sums the per-block absolute sums") = [] {
            mi::BlockArray<double> v{mi::Array<double>{-1, 2}, mi::Array<double>{-3}};
            expect(close(v.absolute_sum(), 6.0));
        };

        test("absolute_sum of an empty block array is zero") = [] {
            mi::BlockArray<double> v;
            expect(close(v.absolute_sum(), 0.0));
        };

        // -- scale -------------------------------------------------------------

        test("scale multiplies every block element in place and returns *this") = [] {
            mi::BlockArray<double> v{mi::Array<double>{1, 2}, mi::Array<double>{3}};
            auto& ref = v.scale(2.0);
            expect(&ref == &v);
            expect(close(v.block(0)[0], 2.0));
            expect(close(v.block(0)[1], 4.0));
            expect(close(v.block(1)[0], 6.0));
        };

        // -- add_scaled --------------------------------------------------------

        test("add_scaled performs block-wise axpy and returns *this") = [] {
            mi::BlockArray<double> y{mi::Array<double>{1, 1}, mi::Array<double>{1}};
            mi::BlockArray<double> x{mi::Array<double>{2, 4}, mi::Array<double>{6}};
            auto& ref = y.add_scaled(0.5, x);
            expect(&ref == &y);
            // y + 0.5*x = {1+1, 1+2}, {1+3} = {2,3},{4}.
            expect(close(y.block(0)[0], 2.0));
            expect(close(y.block(0)[1], 3.0));
            expect(close(y.block(1)[0], 4.0));
        };

        // -- copy --------------------------------------------------------------

        test("copy overwrites values, returns *this, and allocates nothing") = [] {
            mi::BlockArray<double> dst{mi::Array<double>{0, 0}, mi::Array<double>{0}};
            mi::BlockArray<double> src{mi::Array<double>{1, 2}, mi::Array<double>{3}};
            const double* before0 = dst.block(0).data();
            auto& ref = dst.copy(src);
            expect(&ref == &dst);
            expect(dst.block_count() == 2_u);
            expect(dst.block(0).size() == 2_u); // sizes unchanged: no reallocation
            expect(dst.block(1).size() == 1_u);
            expect(dst.block(0).data() == before0) << "copy must not reallocate";
            expect(close(dst.block(0)[0], 1.0));
            expect(close(dst.block(0)[1], 2.0));
            expect(close(dst.block(1)[0], 3.0));
        };

        // -- fill --------------------------------------------------------------

        test("fill sets every element of every block; sum == value * size; pads stay zero") = [] {
            mi::BlockArray<double> v{mi::Array<double>{1, 2}, mi::Array<double>{3}};
            v.fill(7.0);
            expect(close(v.block(0)[0], 7.0));
            expect(close(v.block(0)[1], 7.0));
            expect(close(v.block(1)[0], 7.0));
            // A reduction proves the pads stayed zero (else they would inflate the sum).
            expect(close(v.sum(), 7.0 * static_cast<double>(v.size())));
        };

        // -- operators ---------------------------------------------------------

        test("operator*= agrees with scale") = [] {
            mi::BlockArray<double> v{mi::Array<double>{1, 2}, mi::Array<double>{3}};
            v *= 2.0;
            expect(close(v.block(0)[0], 2.0));
            expect(close(v.block(0)[1], 4.0));
            expect(close(v.block(1)[0], 6.0));
        };

        test("operator/= agrees with scale by the reciprocal") = [] {
            mi::BlockArray<double> v{mi::Array<double>{2, 4}, mi::Array<double>{6}};
            v /= 2.0;
            expect(close(v.block(0)[0], 1.0));
            expect(close(v.block(0)[1], 2.0));
            expect(close(v.block(1)[0], 3.0));
        };

        test("operator+= agrees with add_scaled(1, x)") = [] {
            mi::BlockArray<double> y{mi::Array<double>{1, 1}, mi::Array<double>{1}};
            mi::BlockArray<double> x{mi::Array<double>{2, 4}, mi::Array<double>{6}};
            y += x;
            expect(close(y.block(0)[0], 3.0));
            expect(close(y.block(0)[1], 5.0));
            expect(close(y.block(1)[0], 7.0));
        };

        test("operator-= agrees with add_scaled(-1, x)") = [] {
            mi::BlockArray<double> y{mi::Array<double>{1, 1}, mi::Array<double>{1}};
            mi::BlockArray<double> x{mi::Array<double>{2, 4}, mi::Array<double>{6}};
            y -= x;
            expect(close(y.block(0)[0], -1.0));
            expect(close(y.block(0)[1], -3.0));
            expect(close(y.block(1)[0], -5.0));
        };

        // -- conformance: block-count mismatch ---------------------------------

        test("add_scaled with a block-count mismatch throws invalid_argument") = [] {
            mi::BlockArray<double> a{mi::Array<double>{1, 2}, mi::Array<double>{3}};
            mi::BlockArray<double> b{mi::Array<double>{1}, mi::Array<double>{2}, mi::Array<double>{3}};
            expect(throws<std::invalid_argument>([&] { (void)a.add_scaled(1.0, b); }));
        };

        test("copy with a block-count mismatch throws invalid_argument") = [] {
            mi::BlockArray<double> a{mi::Array<double>{1, 2}, mi::Array<double>{3}};
            mi::BlockArray<double> b{mi::Array<double>{1}, mi::Array<double>{2}, mi::Array<double>{3}};
            expect(throws<std::invalid_argument>([&] { (void)a.copy(b); }));
        };

        test("operator+= with a block-count mismatch throws invalid_argument") = [] {
            mi::BlockArray<double> a{mi::Array<double>{1, 2}, mi::Array<double>{3}};
            mi::BlockArray<double> b{mi::Array<double>{1}, mi::Array<double>{2}, mi::Array<double>{3}};
            expect(throws<std::invalid_argument>([&] { a += b; }));
        };

        // -- conformance: per-block size mismatch ------------------------------

        test("add_scaled with a per-block size mismatch throws invalid_argument") = [] {
            mi::BlockArray<double> a{mi::Array<double>{1, 2}, mi::Array<double>{3}};
            mi::BlockArray<double> b{mi::Array<double>{1, 2, 3}, mi::Array<double>{4}};
            expect(throws<std::invalid_argument>([&] { (void)a.add_scaled(1.0, b); }));
        };

        test("copy with a per-block size mismatch throws invalid_argument") = [] {
            mi::BlockArray<double> a{mi::Array<double>{1, 2}, mi::Array<double>{3}};
            mi::BlockArray<double> b{mi::Array<double>{1, 2, 3}, mi::Array<double>{4}};
            expect(throws<std::invalid_argument>([&] { (void)a.copy(b); }));
        };

        // -- noexcept reductions -----------------------------------------------

        test("sum and absolute_sum are noexcept") = [] {
            static_assert(noexcept(std::declval<mi::BlockArray<double>>().sum()));
            static_assert(noexcept(std::declval<mi::BlockArray<double>>().absolute_sum()));
        };
    };

    suite<"BlockArray float"> block_array_float_suite = [] {
        test("float instantiation: sum and absolute_sum") = [] {
            mi::BlockArray<float> a{mi::Array<float>{1, 2}, mi::Array<float>{3}};
            expect(close(a.sum(), 6.0F, 1e-4F));
            mi::BlockArray<float> v{mi::Array<float>{-3, 4}, mi::Array<float>{-12}};
            expect(close(v.absolute_sum(), 19.0F, 1e-4F));
        };
    };
}
