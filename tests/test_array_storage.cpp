// test_array_storage.cpp -- unit tests for the aligned, padded Array
// container's storage & layout (boost/ut). See specs/array.md.
//
// This is the renamed test_vector_storage.cpp (Vector -> Array); the storage
// surface is unchanged by the rename, so these cases pass against the ported
// implementation in array.hpp.

#include "instrument/array.hpp"

#include <array>
#include <boost/ut.hpp>
#include <cstdint>
#include <stdexcept>

namespace mi = miscibility::instrument;

namespace {

// Number of SIMD lanes for the build's static-dispatch target.
template<class T> std::size_t lane_count() { return mi::detail::hn::Lanes(mi::detail::hn::ScalableTag<T>{}); }

// The full storage invariant: aligned buffer, capacity a whole multiple of the
// lane count, capacity >= size, and every pad slot held at zero.
template<class V> void check_aligned_and_padded(const V& v)
{
    using namespace boost::ut;
    using T = typename V::value_type;
    expect((reinterpret_cast<std::uintptr_t>(v.data()) % mi::detail::alignment) == 0_u);
    const std::size_t lanes = lane_count<T>();
    expect((v.capacity() % lanes) == 0_u) << "single-loop, no remainder";
    expect(v.capacity() >= v.size());
    for (std::size_t i = v.size(); i < v.capacity(); ++i) {
        expect(v.data()[i] == T{}) << "pad slot" << i << "must be zero";
    }
}

} // namespace

int main()
{
    using namespace boost::ut;

    suite<"ArrayStorage"> storage_suite = [] {
        test("buffer is over-aligned to detail::alignment") = [] {
            mi::Array<double, 5> a{1, -4, 3, -2, 9};
            mi::Array<float, 17> b{};
            mi::Array<double> c(1000);
            expect((reinterpret_cast<std::uintptr_t>(a.data()) % mi::detail::alignment) == 0_u);
            expect((reinterpret_cast<std::uintptr_t>(b.data()) % mi::detail::alignment) == 0_u);
            expect((reinterpret_cast<std::uintptr_t>(c.data()) % mi::detail::alignment) == 0_u);
        };

        test("capacity is a whole multiple of the lane count and >= size") = [] {
            mi::Array<double, 5> a{1, -4, 3, -2, 9};
            mi::Array<float, 17> b{};
            mi::Array<double> c(1000);
            expect((a.capacity() % lane_count<double>()) == 0_u);
            expect((b.capacity() % lane_count<float>()) == 0_u);
            expect((c.capacity() % lane_count<double>()) == 0_u);
            expect(a.capacity() >= a.size());
            expect(b.capacity() >= b.size());
            expect(c.capacity() >= c.size());
        };

        test("pad is zero immediately after construction") = [] {
            mi::Array<double, 5> a{1, -4, 3, -2, 9};
            mi::Array<float, 17> b{};
            for (int i = 0; i < 17; ++i) {
                b[static_cast<std::size_t>(i)] = static_cast<float>((i * 7 % 11) - 5);
            }
            mi::Array<double> c(1000);
            check_aligned_and_padded(a);
            check_aligned_and_padded(b);
            check_aligned_and_padded(c);
        };

        test("size and capacity report logical vs padded length") = [] {
            mi::Array<double, 5> a{1, -4, 3, -2, 9};
            expect(a.size() == 5_u);
            expect(a.capacity() >= 5_u);
            mi::Array<double> c(1000);
            expect(c.size() == 1000_u);
            expect(c.capacity() >= 1000_u);
        };

        test("fill sets logical elements and keeps the pad at zero") = [] {
            mi::Array<double> v(7);
            v.fill(3.5);
            for (double i : v) {
                expect(i == 3.5_d);
            }
            check_aligned_and_padded(v);
        };

        test("zero_pad restores the invariant after the pad is dirtied") = [] {
            mi::Array<double, 5> v{1, 2, 3, 4, 5};
            // Dirty the pad through the full padded view.
            auto full = v.padded_span();
            for (std::size_t i = v.size(); i < v.capacity(); ++i) {
                full[i] = 99.0;
            }
            v.zero_pad();
            // Logical elements untouched...
            const std::array<double, 5> expected{1, 2, 3, 4, 5};
            for (std::size_t i = 0; i < v.size(); ++i) {
                expect(v[i] == expected[i]);
            }
            // ...and the pad is zero again.
            check_aligned_and_padded(v);
        };

        test("as_span covers the logical range, padded_span the full buffer") = [] {
            mi::Array<double, 5> v{1, 2, 3, 4, 5};
            expect(v.as_span().size() == v.size());
            expect(v.padded_span().size() == v.capacity());
            expect(v.padded_span().size() >= v.as_span().size());
            expect(v.as_span().data() == v.data());
            expect(v.padded_span().data() == v.data());
        };

        test("at returns in-range elements and throws out_of_range otherwise") = [] {
            mi::Array<double, 5> v{1, -4, 3, -2, 9};
            expect(v.at(0) == 1.0_d);
            expect(v.at(4) == 9.0_d);
            expect(throws<std::out_of_range>([&] { (void)v.at(5); }));
            expect(throws<std::out_of_range>([&] { (void)v.at(100); }));
        };

        test("static initializer_list with the wrong length throws invalid_argument") = [] {
            expect(throws<std::invalid_argument>([] { mi::Array<double, 5> v{1, 2, 3}; }));
            expect(throws<std::invalid_argument>([] { mi::Array<double, 5> v{1, 2, 3, 4, 5, 6}; }));
        };

        test("static initializer_list with the correct length copies values") = [] {
            mi::Array<double, 4> v{10, 20, 30, 40};
            expect(v[0] == 10.0_d);
            expect(v[1] == 20.0_d);
            expect(v[2] == 30.0_d);
            expect(v[3] == 40.0_d);
            check_aligned_and_padded(v);
        };

        test("dynamic Array(n, value) fills every element") = [] {
            mi::Array<double> v(8, 2.0);
            expect(v.size() == 8_u);
            for (double i : v) {
                expect(i == 2.0_d);
            }
            check_aligned_and_padded(v);
        };

        test("Rule of 5: copy constructor makes an independent deep copy") = [] {
            mi::Array<double> a(8, 2.0);
            mi::Array<double> b = a; // copy ctor
            b[0] = 99.0;
            expect(a[0] == 2.0_d) << "buffers must be independent";
            expect(b[0] == 99.0_d);
            expect(a.size() == b.size());
        };

        test("Rule of 5: move, copy-assign, move-assign and self-assign") = [] {
            mi::Array<double> a(8, 2.0);
            mi::Array<double> b = a;
            b[0] = 99.0;
            mi::Array<double> c = std::move(b); // move ctor
            expect(c[0] == 99.0_d);
            a = c; // copy assign
            expect(a[0] == 99.0_d);
            a = std::move(c); // move assign
            expect(a[0] == 99.0_d);
            a = a; // self copy-assign must be safe
            expect(a.size() == 8_u);
            expect(a[0] == 99.0_d);
        };

        test("swap (ADL) exchanges contents of dynamic arrays") = [] {
            mi::Array<double> a(4, 1.0);
            mi::Array<double> b(4, 7.0);
            swap(a, b);
            for (std::size_t i = 0; i < 4; ++i) {
                expect(a[i] == 7.0_d);
                expect(b[i] == 1.0_d);
            }
        };

        test("swap (ADL) exchanges contents of static arrays") = [] {
            mi::Array<double, 3> a{1, 2, 3};
            mi::Array<double, 3> b{4, 5, 6};
            swap(a, b);
            expect(a[0] == 4.0_d);
            expect(a[1] == 5.0_d);
            expect(a[2] == 6.0_d);
            expect(b[0] == 1.0_d);
            expect(b[1] == 2.0_d);
            expect(b[2] == 3.0_d);
        };

        test("empty dynamic Array is valid and padded") = [] {
            mi::Array<double> v{};
            expect(v.size() == 0_u);
            expect(v.empty());
            expect((v.capacity() % lane_count<double>()) == 0_u);
        };
    };

    return 0;
}
