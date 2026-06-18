// test_array_copy.cpp -- unit tests for Array::copy (in-place, no-allocation
// value copy) and the same-size dynamic copy-assign no-allocation optimization
// (boost/ut). See specs/array.md.
//
// This is the renamed test_vector_copy.cpp (Vector -> Array); the copy surface is
// unchanged by the rename, so these cases pass against the implementation carried
// over into array.hpp.

#include "instrument/array.hpp"

#include <boost/ut.hpp>
#include <stdexcept>

namespace mi = miscibility::instrument;

namespace {

// Assert every pad slot [size(), capacity()) is zero.
template<class V> void expect_pad_zero(const V& v)
{
    using namespace boost::ut;
    using T = typename V::value_type;
    for (std::size_t i = v.size(); i < v.capacity(); ++i) {
        expect(v.data()[i] == T{}) << "pad slot" << i << "must be zero";
    }
}

} // namespace

int main()
{
    using namespace boost::ut;

    suite<"ArrayCopy"> copy_suite = [] {
        test("static <- static, same type: values copied, pad zero") = [] {
            mi::Array<double, 5> v1{1, 2, 3, 4, 5};
            mi::Array<double, 5> v2;
            v2.copy(v1);
            for (std::size_t i = 0; i < 5; ++i) {
                expect(v2[i] == v1[i]);
            }
            expect_pad_zero(v2);
        };

        test("cross-extent dynamic <- static: values copied, size unchanged, pad zero") = [] {
            mi::Array<double, 5> v1{1, 2, 3, 4, 5};
            mi::Array<double> v3(5);
            v3.copy(v1);
            expect(v3.size() == 5_u);
            for (std::size_t i = 0; i < 5; ++i) {
                expect(v3[i] == v1[i]);
            }
            expect_pad_zero(v3);
        };

        test("cross-extent static <- dynamic: values copied, pad zero") = [] {
            mi::Array<double> d(5, 7.0);
            mi::Array<double, 5> s;
            s.copy(d);
            for (std::size_t i = 0; i < 5; ++i) {
                expect(s[i] == 7.0_d);
            }
            expect_pad_zero(s);
        };

        test("dynamic <- dynamic, same size: values copied, pad zero") = [] {
            mi::Array<double> src(5);
            for (std::size_t i = 0; i < 5; ++i) {
                src[i] = static_cast<double>(i) - 2.0;
            }
            mi::Array<double> dst(5);
            dst.copy(src);
            for (std::size_t i = 0; i < 5; ++i) {
                expect(dst[i] == src[i]);
            }
            expect_pad_zero(dst);
        };

        test("returns *this for chaining") = [] {
            mi::Array<double, 5> v1{1, 2, 3, 4, 5};
            mi::Array<double, 5> v2;
            v2.copy(v1).scale(2.0);
            for (std::size_t i = 0; i < 5; ++i) {
                expect(v2[i] == 2.0 * v1[i]);
            }
        };

        test("copy makes an independent value copy, not an alias") = [] {
            mi::Array<double, 5> v1{1, 2, 3, 4, 5};
            mi::Array<double, 5> v2;
            v2.copy(v1);
            v1[0] = 99.0;
            expect(v2[0] == 1.0_d) << "mutating source must not change destination";
        };

        test("self-copy leaves contents and size unchanged") = [] {
            mi::Array<double> v(5);
            for (std::size_t i = 0; i < 5; ++i) {
                v[i] = static_cast<double>(i) + 1.0;
            }
            v.copy(v);
            expect(v.size() == 5_u);
            for (std::size_t i = 0; i < 5; ++i) {
                expect(v[i] == static_cast<double>(i) + 1.0);
            }
            expect_pad_zero(v);
        };

        test("pad is restored to zero with no explicit zero_pad by the caller") = [] {
            mi::Array<double, 5> src{1, 2, 3, 4, 5};
            mi::Array<double, 5> dst;
            // Dirty the destination's pad through the full padded view.
            auto full = dst.padded_span();
            for (std::size_t i = dst.size(); i < dst.capacity(); ++i) {
                full[i] = 99.0;
            }
            dst.copy(src);
            for (std::size_t i = 0; i < 5; ++i) {
                expect(dst[i] == src[i]);
            }
            expect_pad_zero(dst);
        };

        test("size mismatch throws invalid_argument and leaves destination untouched") = [] {
            mi::Array<double> src(5, 3.0);
            mi::Array<double> dst(4, 1.0);
            expect(throws<std::invalid_argument>([&] { dst.copy(src); }));
            // destination untouched
            expect(dst.size() == 4_u);
            for (std::size_t i = 0; i < 4; ++i) {
                expect(dst[i] == 1.0_d);
            }
        };

        test("no allocation: copy reuses the destination buffer (dynamic <- dynamic)") = [] {
            mi::Array<double> src(5, 4.0);
            mi::Array<double> dst(5);
            const double* before = dst.data();
            dst.copy(src);
            expect(dst.data() == before) << "copy must not reallocate";
        };

        // The optimized same-size copy-assign must reuse the buffer (new behavior) AND remain a
        // correct, independent value copy that is self-assignment-safe (regression guard).
        test("no allocation: same-size dynamic copy-assign reuses the buffer and stays correct") = [] {
            mi::Array<double> a(5, 1.0);
            mi::Array<double> b(5, 2.0);
            const double* before = a.data();
            a = b;
            expect(a.data() == before) << "same-size copy-assign must not reallocate";
            for (std::size_t i = 0; i < 5; ++i) {
                expect(a[i] == 2.0_d);
            }
            expect_pad_zero(a);
            // independent buffers
            a[0] = 7.0;
            expect(b[0] == 2.0_d) << "buffers must stay independent after copy-assign";
            // self copy-assign must be safe
            a = a;
            expect(a.size() == 5_u);
            expect(a[0] == 7.0_d);
        };
    };

    return 0;
}
