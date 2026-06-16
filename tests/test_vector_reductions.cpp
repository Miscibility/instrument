// test_vector_reductions.cpp -- unit tests for the Vector BLAS-1 kernels
// (reductions & vector ops): scale, add_scaled, dot, euclidean_norm,
// absolute_sum, index_of_max_magnitude, max_magnitude and the operators.
// boost/ut. See specs/vector-reductions.md.
//
// Every test is expected to FAIL until tdd-3-implement fills in the stubs:
// each kernel/method currently throws std::runtime_error{"not implemented"}.
// (The noexcept kernels/methods turn that throw into std::terminate, so the
// binary aborts when a reduction is first exercised -- that is the expected
// "not implemented" signal; tdd-3 replaces the bodies and the suite goes green.)

#include "instrument/vector.hpp"

#include <boost/ut.hpp>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace mi = miscibility::instrument;

namespace {

// Relative-tolerance comparator, mirroring tmp/test_vector.cpp's close(). The
// default tolerance is tight: every reduction here is either bit-exact or, for
// the 1000-element double sums, off by only the reassociation rounding of a
// SIMD vs. sequential accumulation (~2e-15 relative). 1e-12 keeps ~500x headroom.
template<class T> bool close(T a, T b, T tol = T(1e-12))
{
    return std::abs(a - b) <= tol * (T(1) + std::abs(a) + std::abs(b));
}

template<class T> std::size_t lane_count() { return mi::detail::hn::Lanes(mi::detail::hn::ScalableTag<T>{}); }

// The storage invariant: aligned buffer, capacity a whole multiple of the lane
// count, capacity >= size, every pad slot zero.
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

    suite<"VectorBLAS1"> blas1_suite = [] {
        // -- dot ---------------------------------------------------------------

        test("dot matches the scalar reference loop (static, non-dividing extent)") = [] {
            mi::Vector<double, 5> x{1, -4, 3, -2, 9};
            mi::Vector<double, 5> y{-1, 2, -3, 4, -5};
            std::vector<double> rx(x.begin(), x.end()), ry(y.begin(), y.end());
            double ref = 0;
            for (std::size_t i = 0; i < rx.size(); ++i) {
                ref += rx[i] * ry[i];
            }
            expect(close(x.dot(y), ref));
        };

        test("dot matches the scalar reference loop (dynamic, 1000 sin/cos)") = [] {
            mi::Vector<double> x(1000), y(1000);
            for (std::size_t i = 0; i < 1000; ++i) {
                x[i] = std::sin(0.1 * static_cast<double>(i));
                y[i] = std::cos(0.07 * static_cast<double>(i));
            }
            double ref = 0;
            for (std::size_t i = 0; i < 1000; ++i) {
                ref += x[i] * y[i];
            }
            expect(close(x.dot(y), ref));
        };

        test("dot matches the scalar reference loop (float, extent 17)") = [] {
            mi::Vector<float, 17> x{}, y{};
            for (int i = 0; i < 17; ++i) {
                x[static_cast<std::size_t>(i)] = static_cast<float>((i * 7 % 11) - 5);
                y[static_cast<std::size_t>(i)] = static_cast<float>((i * 3 % 13) - 6);
            }
            float ref = 0;
            for (std::size_t i = 0; i < 17; ++i) {
                ref += x[i] * y[i];
            }
            expect(close(x.dot(y), ref, 1e-8F));
        };

        // -- euclidean_norm ----------------------------------------------------

        test("euclidean_norm matches sqrt(Sum v_i^2)") = [] {
            mi::Vector<double, 5> v{1, -4, 3, -2, 9};
            double ref = 0;
            for (double e : v) {
                ref += e * e;
            }
            ref = std::sqrt(ref);
            expect(close(v.euclidean_norm(), ref));
        };

        test("euclidean_norm of a dynamic vector matches the reference") = [] {
            mi::Vector<double> v(1000);
            double ref = 0;
            for (std::size_t i = 0; i < 1000; ++i) {
                v[i] = std::sin(0.1 * static_cast<double>(i));
                ref += v[i] * v[i];
            }
            ref = std::sqrt(ref);
            expect(close(v.euclidean_norm(), ref));
        };

        // -- absolute_sum ------------------------------------------------------

        test("absolute_sum matches Sum |v_i|") = [] {
            mi::Vector<double, 5> v{1, -4, 3, -2, 9};
            double ref = 0;
            for (double e : v) {
                ref += std::abs(e);
            }
            expect(close(v.absolute_sum(), ref));
        };

        // -- index_of_max_magnitude / max_magnitude ----------------------------

        test("index_of_max_magnitude matches a scalar argmax") = [] {
            mi::Vector<double, 5> v{1, -4, 3, -2, 9};
            std::vector<double> rv(v.begin(), v.end());
            std::size_t ref = 0;
            for (std::size_t i = 1; i < rv.size(); ++i) {
                if (std::abs(rv[i]) > std::abs(rv[ref])) {
                    ref = i;
                }
            }
            expect(v.index_of_max_magnitude() == ref);
            expect(ref == 4_u) << "max magnitude is 9 at index 4";
        };

        test("index_of_max_magnitude breaks ties at the smallest index") = [] {
            // |3| == |-3|; the first wins under strict '>'.
            mi::Vector<double, 3> v{3, -3, 1};
            expect(v.index_of_max_magnitude() == 0_u);
        };

        test("max_magnitude is |element at the argmax|") = [] {
            mi::Vector<double, 5> v{1, -4, 3, -2, 9};
            expect(close(v.max_magnitude(), 9.0));
            mi::Vector<double, 3> t{3, -3, 1};
            expect(close(t.max_magnitude(), 3.0));
        };

        // -- degenerate cases --------------------------------------------------

        test("all-zero non-empty vector: argmax 0 and norm 0") = [] {
            mi::Vector<double> v(16); // zero-filled
            expect(v.index_of_max_magnitude() == 0_u);
            expect(close(v.euclidean_norm(), 0.0));
        };

        test("empty dynamic vector: index_of_max_magnitude == size()") = [] {
            mi::Vector<double> v{};
            expect(v.size() == 0_u);
            expect(v.index_of_max_magnitude() == v.size());
            expect(v.index_of_max_magnitude() == 0_u);
        };

        // -- scale -------------------------------------------------------------

        test("scale(2.5) scales every element and keeps the pad zero") = [] {
            mi::Vector<double, 5> v{1, -4, 3, -2, 9};
            std::vector<double> rv(v.begin(), v.end());
            v.scale(2.5);
            for (std::size_t i = 0; i < rv.size(); ++i) {
                expect(close(v[i], rv[i] * 2.5));
            }
            check_aligned_and_padded(v);
        };

        // -- add_scaled --------------------------------------------------------

        test("add_scaled(3, x): y <- y + 3*x and the pad stays zero") = [] {
            mi::Vector<double, 5> x{1, -4, 3, -2, 9};
            mi::Vector<double, 5> y{-1, 2, -3, 4, -5};
            std::vector<double> rx(x.begin(), x.end()), ry(y.begin(), y.end());
            y.add_scaled(3.0, x);
            for (std::size_t i = 0; i < ry.size(); ++i) {
                expect(close(y[i], ry[i] + (3.0 * rx[i])));
            }
            check_aligned_and_padded(y);
        };

        // -- size-mismatch throws ---------------------------------------------

        test("dot and add_scaled throw invalid_argument on size mismatch") = [] {
            mi::Vector<double> a(5);
            mi::Vector<double> b(6);
            expect(throws<std::invalid_argument>([&] { (void)a.dot(b); }));
            expect(throws<std::invalid_argument>([&] { a.add_scaled(1.0, b); }));
        };

        test("size mismatch throws for a static-vs-dynamic pair") = [] {
            mi::Vector<double, 5> a{1, 2, 3, 4, 5};
            mi::Vector<double> b(4);
            expect(throws<std::invalid_argument>([&] { (void)a.dot(b); }));
            expect(throws<std::invalid_argument>([&] { a.add_scaled(1.0, b); }));
        };

        // -- operators ---------------------------------------------------------

        test("operators: v *= 2; v /= 4; v += w componentwise result") = [] {
            mi::Vector<double, 4> v{1, 2, 3, 4};
            mi::Vector<double, 4> w{1, 1, 1, 1};
            v *= 2.0; // {2, 4, 6, 8}
            v /= 4.0; // {0.5, 1.0, 1.5, 2.0}
            v += w;   // {1.5, 2.0, 2.5, 3.0}
            expect(close(v[0], 1.5));
            expect(close(v[1], 2.0));
            expect(close(v[2], 2.5));
            expect(close(v[3], 3.0));
            check_aligned_and_padded(v);
        };

        test("operator-= subtracts componentwise") = [] {
            mi::Vector<double, 4> v{5, 5, 5, 5};
            mi::Vector<double, 4> w{1, 2, 3, 4};
            v -= w; // {4, 3, 2, 1}
            expect(close(v[0], 4.0));
            expect(close(v[1], 3.0));
            expect(close(v[2], 2.0));
            expect(close(v[3], 1.0));
        };

        // -- cross-extent same-size operands ----------------------------------

        test("cross-extent same-size operands: static.dot(dynamic) succeeds") = [] {
            mi::Vector<double, 5> a{1, -4, 3, -2, 9};
            mi::Vector<double> b(5);
            for (std::size_t i = 0; i < 5; ++i) {
                b[i] = static_cast<double>(i) - 2.0;
            }
            double ref = 0;
            for (std::size_t i = 0; i < 5; ++i) {
                ref += a[i] * b[i];
            }
            expect(close(a.dot(b), ref));

            // ...and add_scaled across extents likewise.
            mi::Vector<double, 5> y{0, 0, 0, 0, 0};
            std::vector<double> rb(b.begin(), b.end());
            y.add_scaled(2.0, b);
            for (std::size_t i = 0; i < 5; ++i) {
                expect(close(y[i], 2.0 * rb[i]));
            }
        };

        // -- read-only reductions do not disturb the source -------------------

        test("reductions leave the source unmodified and the pad zero") = [] {
            mi::Vector<double, 5> v{1, -4, 3, -2, 9};
            std::vector<double> before(v.begin(), v.end());

            // Exercise every read-only reduction.
            (void)v.euclidean_norm();
            (void)v.absolute_sum();
            (void)v.index_of_max_magnitude();
            (void)v.max_magnitude();
            mi::Vector<double, 5> w{-1, 2, -3, 4, -5};
            (void)v.dot(w);

            for (std::size_t i = 0; i < before.size(); ++i) {
                expect(v[i] == before[i]) << "read-only op must not mutate element" << i;
            }
            check_aligned_and_padded(v);
        };
    };

    return 0;
}
