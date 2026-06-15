// test_vector_storage.cpp -- unit tests for the aligned, padded Vector
// container's storage & layout (boost/ut). See specs/vector-storage.md.
//
// Every test is expected to FAIL until tdd-3-implement fills in the stubs:
// each Vector constructor currently throws std::runtime_error, so any test
// that builds a Vector fails before it can assert real behaviour.

#include <boost/ut.hpp>

#include <cstdint>
#include <stdexcept>
#include <vector>

#include "instrument/vector.hpp"

namespace mi = miscibility::instrument;

namespace {

// Number of SIMD lanes for the build's static-dispatch target.
template<class T> std::size_t lane_count()
{
    return mi::detail::hn::Lanes(mi::detail::hn::ScalableTag<T>{});
}

// The full storage invariant: aligned buffer, capacity a whole multiple of the
// lane count, capacity >= size, and every pad slot held at zero.
template<class V> void check_aligned_and_padded(const V& v)
{
    using namespace boost::ut;
    using T = typename V::value_type;
    expect((reinterpret_cast<std::uintptr_t>(v.data()) % mi::detail::alignment) == 0_u);
    const std::size_t lanes = lane_count<T>();
    expect((v.capacity() % lanes) == 0u) << "single-loop, no remainder";
    expect(v.capacity() >= v.size());
    for (std::size_t i = v.size(); i < v.capacity(); ++i) {
        expect(v.data()[i] == T{}) << "pad slot" << i << "must be zero";
    }
}

} // namespace

int main()
{
    using namespace boost::ut;

    suite<"VectorStorage"> storage_suite = [] {
        test("buffer is over-aligned to detail::alignment") = [] {
            mi::Vector<double, 5> a{1, -4, 3, -2, 9};
            mi::Vector<float, 17> b{};
            mi::Vector<double> c(1000);
            expect((reinterpret_cast<std::uintptr_t>(a.data()) % mi::detail::alignment) == 0_u);
            expect((reinterpret_cast<std::uintptr_t>(b.data()) % mi::detail::alignment) == 0_u);
            expect((reinterpret_cast<std::uintptr_t>(c.data()) % mi::detail::alignment) == 0_u);
        };

        test("capacity is a whole multiple of the lane count and >= size") = [] {
            mi::Vector<double, 5> a{1, -4, 3, -2, 9};
            mi::Vector<float, 17> b{};
            mi::Vector<double> c(1000);
            expect((a.capacity() % lane_count<double>()) == 0_u);
            expect((b.capacity() % lane_count<float>()) == 0_u);
            expect((c.capacity() % lane_count<double>()) == 0_u);
            expect(a.capacity() >= a.size());
            expect(b.capacity() >= b.size());
            expect(c.capacity() >= c.size());
        };

        test("pad is zero immediately after construction") = [] {
            mi::Vector<double, 5> a{1, -4, 3, -2, 9};
            mi::Vector<float, 17> b{};
            for (int i = 0; i < 17; ++i) {
                b[static_cast<std::size_t>(i)] = static_cast<float>((i * 7 % 11) - 5);
            }
            mi::Vector<double> c(1000);
            check_aligned_and_padded(a);
            check_aligned_and_padded(b);
            check_aligned_and_padded(c);
        };

        test("size and capacity report logical vs padded length") = [] {
            mi::Vector<double, 5> a{1, -4, 3, -2, 9};
            expect(a.size() == 5_u);
            expect(a.capacity() >= 5_u);
            mi::Vector<double> c(1000);
            expect(c.size() == 1000_u);
            expect(c.capacity() >= 1000_u);
        };

        test("fill sets logical elements and keeps the pad at zero") = [] {
            mi::Vector<double> v(7);
            v.fill(3.5);
            for (double i : v) {
                expect(i == 3.5_d);
            }
            check_aligned_and_padded(v);
        };

        test("zero_pad restores the invariant after the pad is dirtied") = [] {
            mi::Vector<double, 5> v{1, 2, 3, 4, 5};
            // Dirty the pad through the full padded view.
            auto full = v.padded_span();
            for (std::size_t i = v.size(); i < v.capacity(); ++i) {
                full[i] = 99.0;
            }
            v.zero_pad();
            // Logical elements untouched...
            const std::array<double,5> expected{1, 2, 3, 4, 5};
            for (std::size_t i = 0; i < v.size(); ++i) {
                expect(v[i] == expected[i]);
            }
            // ...and the pad is zero again.
            check_aligned_and_padded(v);
        };

        test("as_span covers the logical range, padded_span the full buffer") = [] {
            mi::Vector<double, 5> v{1, 2, 3, 4, 5};
            expect(v.as_span().size() == v.size());
            expect(v.padded_span().size() == v.capacity());
            expect(v.padded_span().size() >= v.as_span().size());
            expect(v.as_span().data() == v.data());
            expect(v.padded_span().data() == v.data());
        };

        test("at returns in-range elements and throws out_of_range otherwise") = [] {
            mi::Vector<double, 5> v{1, -4, 3, -2, 9};
            expect(v.at(0) == 1.0_d);
            expect(v.at(4) == 9.0_d);
            expect(throws<std::out_of_range>([&] { (void)v.at(5); }));
            expect(throws<std::out_of_range>([&] { (void)v.at(100); }));
        };

        test("static initializer_list with the wrong length throws invalid_argument") = [] {
            expect(throws<std::invalid_argument>([] { mi::Vector<double, 5> v{1, 2, 3}; }));
            expect(throws<std::invalid_argument>([] { mi::Vector<double, 5> v{1, 2, 3, 4, 5, 6}; }));
        };

        test("static initializer_list with the correct length copies values") = [] {
            mi::Vector<double, 4> v{10, 20, 30, 40};
            expect(v[0] == 10.0_d);
            expect(v[1] == 20.0_d);
            expect(v[2] == 30.0_d);
            expect(v[3] == 40.0_d);
            check_aligned_and_padded(v);
        };

        test("dynamic Vector(n, value) fills every element") = [] {
            mi::Vector<double> v(8, 2.0);
            expect(v.size() == 8_u);
            for (double i : v) {
                expect(i == 2.0_d);
            }
            check_aligned_and_padded(v);
        };

        test("Rule of 5: copy constructor makes an independent deep copy") = [] {
            mi::Vector<double> a(8, 2.0);
            mi::Vector<double> b = a; // copy ctor
            b[0] = 99.0;
            expect(a[0] == 2.0_d) << "buffers must be independent";
            expect(b[0] == 99.0_d);
            expect(a.size() == b.size());
        };

        test("Rule of 5: move, copy-assign, move-assign and self-assign") = [] {
            mi::Vector<double> a(8, 2.0);
            mi::Vector<double> b = a;
            b[0] = 99.0;
            mi::Vector<double> c = std::move(b); // move ctor
            expect(c[0] == 99.0_d);
            a = c; // copy assign
            expect(a[0] == 99.0_d);
            a = std::move(c); // move assign
            expect(a[0] == 99.0_d);
            a = a; // self copy-assign must be safe
            expect(a.size() == 8_u);
            expect(a[0] == 99.0_d);
        };

        test("swap (ADL) exchanges contents of dynamic vectors") = [] {
            mi::Vector<double> a(4, 1.0);
            mi::Vector<double> b(4, 7.0);
            swap(a, b);
            for (std::size_t i = 0; i < 4; ++i) {
                expect(a[i] == 7.0_d);
                expect(b[i] == 1.0_d);
            }
        };

        test("swap (ADL) exchanges contents of static vectors") = [] {
            mi::Vector<double, 3> a{1, 2, 3};
            mi::Vector<double, 3> b{4, 5, 6};
            swap(a, b);
            expect(a[0] == 4.0_d);
            expect(a[1] == 5.0_d);
            expect(a[2] == 6.0_d);
            expect(b[0] == 1.0_d);
            expect(b[1] == 2.0_d);
            expect(b[2] == 3.0_d);
        };

        test("empty dynamic Vector is valid and padded") = [] {
            mi::Vector<double> v{};
            expect(v.size() == 0_u);
            expect(v.empty());
            expect((v.capacity() % lane_count<double>()) == 0_u);
        };
    };

    return 0;
}
