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

    suite<"BlockArray transforms"> block_array_transforms_suite = [] {
        // -- reductions: argmax / max_magnitude --------------------------------

        test("max_magnitude is the largest |element| across blocks") = [] {
            mi::BlockArray<double> v{mi::Array<double>{1, -4, 3}, mi::Array<double>{-2, 9}};
            expect(close(v.max_magnitude(), 9.0));
        };

        test("max_magnitude of an empty block array is zero") = [] {
            mi::BlockArray<double> v;
            expect(close(v.max_magnitude(), 0.0));
        };

        test("index_of_max_magnitude reports the global (concatenated) index") = [] {
            // Concatenation: [1, -4, 3, -2, 9] -> argmax |.| at global index 4.
            mi::BlockArray<double> v{mi::Array<double>{1, -4, 3}, mi::Array<double>{-2, 9}};
            expect(v.index_of_max_magnitude() == 4_u);
        };

        test("index_of_max_magnitude breaks ties at the smallest global index") = [] {
            // |4| in block 0 at global 1; |-4| in block 1 at global 3. First wins.
            mi::BlockArray<double> v{mi::Array<double>{1, 4}, mi::Array<double>{-4, 2}};
            expect(v.index_of_max_magnitude() == 1_u);
        };

        test("index_of_max_magnitude of an empty block array == size()") = [] {
            mi::BlockArray<double> v;
            expect(v.index_of_max_magnitude() == v.size());
            expect(v.index_of_max_magnitude() == 0_u);
        };

        // -- unary componentwise transforms ------------------------------------

        test("abs applies block-wise and returns *this") = [] {
            mi::BlockArray<double> v{mi::Array<double>{-1, 2}, mi::Array<double>{-3}};
            auto& ref = v.abs();
            expect(&ref == &v);
            expect(close(v.block(0)[0], 1.0));
            expect(close(v.block(0)[1], 2.0));
            expect(close(v.block(1)[0], 3.0));
        };

        test("exp applies block-wise and the pads stay zero (sum stays finite/correct)") = [] {
            mi::BlockArray<double> v{mi::Array<double>{0.0, 1.0}, mi::Array<double>{2.0}};
            v.exp();
            expect(close(v.block(0)[0], std::exp(0.0)));
            expect(close(v.block(0)[1], std::exp(1.0)));
            expect(close(v.block(1)[0], std::exp(2.0)));
            // If a pad had been left at exp(0)=1, the sum would exceed the logical total.
            expect(close(v.sum(), std::exp(0.0) + std::exp(1.0) + std::exp(2.0)));
        };

        test("chaining: abs().sqrt() composes block-wise") = [] {
            mi::BlockArray<double> v{mi::Array<double>{-4, 9}, mi::Array<double>{-16}};
            v.abs().sqrt();
            expect(close(v.block(0)[0], 2.0));
            expect(close(v.block(0)[1], 3.0));
            expect(close(v.block(1)[0], 4.0));
        };

        test("apply with a custom functor (x + 1) runs block-wise") = [] {
            mi::BlockArray<double> v{mi::Array<double>{1, 2}, mi::Array<double>{3}};
            v.apply([](auto d, auto x) { return mi::detail::hn::Add(x, mi::detail::hn::Set(d, 1.0)); });
            expect(close(v.block(0)[0], 2.0));
            expect(close(v.block(0)[1], 3.0));
            expect(close(v.block(1)[0], 4.0));
        };

        // -- binary elementwise ops --------------------------------------------

        test("elementwise_product multiplies block-wise") = [] {
            mi::BlockArray<double> a{mi::Array<double>{1, 2}, mi::Array<double>{3}};
            mi::BlockArray<double> b{mi::Array<double>{4, 5}, mi::Array<double>{6}};
            a.elementwise_product(b);
            expect(close(a.block(0)[0], 4.0));
            expect(close(a.block(0)[1], 10.0));
            expect(close(a.block(1)[0], 18.0));
        };

        test("elementwise_quotient divides block-wise; pad NaN does not leak") = [] {
            mi::BlockArray<double> a{mi::Array<double>{8, 9}, mi::Array<double>{6}};
            mi::BlockArray<double> b{mi::Array<double>{2, 3}, mi::Array<double>{6}};
            a.elementwise_quotient(b);
            expect(close(a.block(0)[0], 4.0));
            expect(close(a.block(0)[1], 3.0));
            expect(close(a.block(1)[0], 1.0));
            // The transient 0/0 = NaN in each block's pad must have been scrubbed.
            expect(std::isfinite(a.absolute_sum())) << "no NaN leaked from a pad";
        };

        test("elementwise_product/quotient throw invalid_argument on a mismatch") = [] {
            mi::BlockArray<double> a{mi::Array<double>{1, 2}, mi::Array<double>{3}};
            mi::BlockArray<double> b{mi::Array<double>{1, 2, 3}, mi::Array<double>{4}};
            expect(throws<std::invalid_argument>([&] { a.elementwise_product(b); }));
            expect(throws<std::invalid_argument>([&] { a.elementwise_quotient(b); }));
        };

        // -- swap --------------------------------------------------------------

        test("swap (ADL) exchanges contents") = [] {
            mi::BlockArray<double> a{mi::Array<double>{1, 2}};
            mi::BlockArray<double> b{mi::Array<double>{7, 8}, mi::Array<double>{9}};
            swap(a, b);
            expect(a.block_count() == 2_u);
            expect(b.block_count() == 1_u);
            expect(close(a.block(0)[0], 7.0));
            expect(close(b.block(0)[1], 2.0));
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
