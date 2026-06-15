// test_vector_componentwise.cpp -- unit tests for the Vector componentwise
// transforms: apply, abs, sqrt and every Highway transcendental (exp/exp2/
// expm1, log/log2/log10/log1p, sin/cos, sinh/tanh, asin/acos/asinh/acosh,
// atan/atanh) plus the binary Hadamard ops multiply and divide.
// boost/ut. See specs/vector-componentwise.md.
//
// Every test is expected to FAIL until tdd-3-implement fills in the stubs:
// detail::map and detail::zip currently throw std::runtime_error{"not
// implemented"}. Both kernels are noexcept, so that throw becomes std::terminate
// and the binary aborts the moment a transform is first exercised -- that is the
// expected "not implemented" signal. The size-mismatch test is the one exception:
// multiply/divide run the (already-implemented) check_same_size before reaching
// the stub, so that test pins the throwing contract and passes now.

#include <boost/ut.hpp>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>

#include "instrument/vector.hpp"

namespace mi = miscibility::instrument;

namespace {

// Relative-tolerance comparator, mirroring test_vector_reductions.cpp. The
// transcendentals are accurate to a few ULP, so a 1e-9 relative tolerance for
// double (1e-5 for float) leaves ample headroom while still catching a wrong
// kernel.
template<class T> bool close(T a, T b, T tol = T(1e-9))
{
    return std::abs(a - b) <= tol * (T(1) + std::abs(a) + std::abs(b));
}

template<class T> std::size_t lane_count()
{
    return mi::detail::hn::Lanes(mi::detail::hn::ScalableTag<T>{});
}

// The storage invariant: aligned buffer, capacity a whole multiple of the lane
// count, capacity >= size, every pad slot zero. The pad check is the crux of
// this spec -- it fails if a transform forgets its trailing zero_pad().
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

// Apply a unary transform to a fresh dynamic double Vector built from `input`,
// then assert it matches the scalar reference elementwise and the pad is zero.
template<class Transform, class Ref>
void check_unary_double(const std::vector<double>& input, Transform transform, Ref ref,
                        double tol = 1e-9)
{
    using namespace boost::ut;
    mi::Vector<double> v(input.size());
    std::copy(input.begin(), input.end(), v.begin());
    transform(v); // in place
    for (std::size_t i = 0; i < input.size(); ++i) {
        expect(close(v[i], ref(input[i]), tol)) << "element" << i;
    }
    check_aligned_and_padded(v);
}

} // namespace

int main()
{
    using namespace boost::ut;

    suite<"VectorComponentwise"> componentwise_suite = [] {
        // -- abs ---------------------------------------------------------------

        test("abs: x_i == |ref_x_i| for negative/positive/zero; pad zero") = [] {
            mi::Vector<double, 17> v{};
            for (int i = 0; i < 17; ++i) {
                v[static_cast<std::size_t>(i)] = static_cast<double>(i - 8); // -8..8, incl. 0
            }
            std::vector<double> ref(v.begin(), v.end());
            v.abs();
            for (std::size_t i = 0; i < ref.size(); ++i) {
                expect(close(v[i], std::abs(ref[i])));
            }
            check_aligned_and_padded(v);
        };

        // -- sqrt --------------------------------------------------------------

        test("sqrt: matches std::sqrt on non-negative input; pad zero") = [] {
            check_unary_double({0.0, 0.25, 1.0, 2.0, 9.0, 1234.5, 0.001},
                               [](auto& v) { v.sqrt(); },
                               [](double x) { return std::sqrt(x); });
        };

        // -- exp + the central pad invariant -----------------------------------

        test("exp: matches std::exp; pad is zero afterward (the zero_pad guard)") = [] {
            // exp(0) = 1 would corrupt the pad if the trailing zero_pad() were
            // omitted; check_aligned_and_padded is exactly that regression test.
            check_unary_double({-3.0, -1.0, 0.0, 0.5, 1.0, 2.5, 4.0},
                               [](auto& v) { v.exp(); },
                               [](double x) { return std::exp(x); });
        };

        // -- the remaining transcendentals -------------------------------------

        test("exp2: matches std::exp2") = [] {
            check_unary_double({-3.0, -1.0, 0.0, 0.5, 1.0, 2.5, 8.0},
                               [](auto& v) { v.exp2(); },
                               [](double x) { return std::exp2(x); });
        };

        test("expm1: matches std::expm1") = [] {
            check_unary_double({-2.0, -0.1, 0.0, 1e-6, 0.5, 1.0, 3.0},
                               [](auto& v) { v.expm1(); },
                               [](double x) { return std::expm1(x); });
        };

        test("log: matches std::log on positive input") = [] {
            check_unary_double({1e-3, 0.5, 1.0, 2.0, 10.0, 250.0, 1e4},
                               [](auto& v) { v.log(); },
                               [](double x) { return std::log(x); });
        };

        test("log2: matches std::log2 on positive input") = [] {
            check_unary_double({1e-3, 0.5, 1.0, 2.0, 8.0, 250.0, 1e4},
                               [](auto& v) { v.log2(); },
                               [](double x) { return std::log2(x); });
        };

        test("log10: matches std::log10 on positive input") = [] {
            check_unary_double({1e-3, 0.5, 1.0, 10.0, 100.0, 250.0, 1e4},
                               [](auto& v) { v.log10(); },
                               [](double x) { return std::log10(x); });
        };

        test("log1p: matches std::log1p on input > -1") = [] {
            check_unary_double({-0.9, -0.1, 0.0, 1e-6, 0.5, 2.0, 100.0},
                               [](auto& v) { v.log1p(); },
                               [](double x) { return std::log1p(x); });
        };

        test("sin: matches std::sin") = [] {
            check_unary_double({-3.0, -1.0, 0.0, 0.5, 1.5, 3.0, 6.0},
                               [](auto& v) { v.sin(); },
                               [](double x) { return std::sin(x); });
        };

        test("cos: matches std::cos") = [] {
            check_unary_double({-3.0, -1.0, 0.0, 0.5, 1.5, 3.0, 6.0},
                               [](auto& v) { v.cos(); },
                               [](double x) { return std::cos(x); });
        };

        test("sinh: matches std::sinh") = [] {
            check_unary_double({-3.0, -1.0, 0.0, 0.5, 1.5, 3.0, 5.0},
                               [](auto& v) { v.sinh(); },
                               [](double x) { return std::sinh(x); });
        };

        test("tanh: matches std::tanh") = [] {
            check_unary_double({-3.0, -1.0, 0.0, 0.5, 1.5, 3.0, 8.0},
                               [](auto& v) { v.tanh(); },
                               [](double x) { return std::tanh(x); });
        };

        test("asin: matches std::asin on [-1, 1]") = [] {
            check_unary_double({-1.0, -0.7, -0.1, 0.0, 0.25, 0.7, 1.0},
                               [](auto& v) { v.asin(); },
                               [](double x) { return std::asin(x); });
        };

        test("acos: matches std::acos on [-1, 1]") = [] {
            check_unary_double({-1.0, -0.7, -0.1, 0.0, 0.25, 0.7, 1.0},
                               [](auto& v) { v.acos(); },
                               [](double x) { return std::acos(x); });
        };

        test("asinh: matches std::asinh") = [] {
            check_unary_double({-5.0, -1.0, 0.0, 0.5, 1.5, 3.0, 10.0},
                               [](auto& v) { v.asinh(); },
                               [](double x) { return std::asinh(x); });
        };

        test("acosh: matches std::acosh on input >= 1") = [] {
            check_unary_double({1.0, 1.1, 1.5, 2.0, 5.0, 50.0, 1e3},
                               [](auto& v) { v.acosh(); },
                               [](double x) { return std::acosh(x); });
        };

        test("atan: matches std::atan") = [] {
            check_unary_double({-50.0, -1.0, -0.1, 0.0, 0.5, 3.0, 100.0},
                               [](auto& v) { v.atan(); },
                               [](double x) { return std::atan(x); });
        };

        test("atanh: matches std::atanh on (-1, 1)") = [] {
            check_unary_double({-0.95, -0.5, -0.1, 0.0, 0.25, 0.6, 0.95},
                               [](auto& v) { v.atanh(); },
                               [](double x) { return std::atanh(x); });
        };

        // -- apply with a custom functor --------------------------------------

        test("apply with a custom functor (x + 1) matches the scalar reference; pad zero") = [] {
            check_unary_double(
                {-3.0, -1.0, 0.0, 0.5, 1.0, 2.5, 4.0},
                [](auto& v) {
                    v.apply([](auto d, auto x) { return mi::detail::hn::Add(x, mi::detail::hn::Set(d, 1.0)); });
                },
                [](double x) { return x + 1.0; });
        };

        // -- elementwise_product (Hadamard) -----------------------------------

        test("elementwise_product (Hadamard): x_i == ref_x_i * ref_y_i; pad zero") = [] {
            mi::Vector<double, 17> x{}, y{};
            for (int i = 0; i < 17; ++i) {
                x[static_cast<std::size_t>(i)] = static_cast<double>(i - 8);
                y[static_cast<std::size_t>(i)] = static_cast<double>((i * 3 % 7) - 3);
            }
            std::vector<double> rx(x.begin(), x.end()), ry(y.begin(), y.end());
            x.elementwise_product(y);
            for (std::size_t i = 0; i < rx.size(); ++i) {
                expect(close(x[i], rx[i] * ry[i]));
            }
            check_aligned_and_padded(x);
        };

        // -- elementwise_quotient ---------------------------------------------

        test("elementwise_quotient: x_i == ref_x_i / ref_y_i (non-zero divisors); pad zero; norm finite") = [] {
            mi::Vector<double, 17> x{}, y{};
            for (int i = 0; i < 17; ++i) {
                x[static_cast<std::size_t>(i)] = static_cast<double>(i - 8);
                y[static_cast<std::size_t>(i)] = static_cast<double>((i % 5) + 1); // 1..5, never 0
            }
            std::vector<double> rx(x.begin(), x.end()), ry(y.begin(), y.end());
            x.elementwise_quotient(y);
            for (std::size_t i = 0; i < rx.size(); ++i) {
                expect(close(x[i], rx[i] / ry[i]));
            }
            check_aligned_and_padded(x);
            // The pad transiently computed 0/0 = NaN; zero_pad() must have scrubbed it
            // before euclidean_norm sums the squares, so the result stays finite.
            expect(std::isfinite(x.euclidean_norm())) << "no NaN leaked from 0/0 in the pad";
        };

        // -- integration: transform then reduce -------------------------------

        test("integration: x.exp() then absolute_sum() == sum of exp over logical range only") = [] {
            mi::Vector<double, 17> x{};
            for (int i = 0; i < 17; ++i) {
                x[static_cast<std::size_t>(i)] = 0.1 * static_cast<double>(i - 8);
            }
            double ref = 0;
            for (double e : x) {
                ref += std::exp(e); // logical range only
            }
            x.exp();
            // exp values are all positive, so absolute_sum == plain sum. If the pad
            // were left at exp(0)=1, the sum would be too large by (capacity-size).
            expect(close(x.absolute_sum(), ref));
            check_aligned_and_padded(x);
        };

        // -- chaining ----------------------------------------------------------

        test("chaining: x.abs().sqrt() composes and leaves a valid (aligned, zero-pad) vector") = [] {
            mi::Vector<double, 17> x{};
            for (int i = 0; i < 17; ++i) {
                x[static_cast<std::size_t>(i)] = static_cast<double>(i - 8); // negatives included
            }
            std::vector<double> ref(x.begin(), x.end());
            x.abs().sqrt();
            for (std::size_t i = 0; i < ref.size(); ++i) {
                expect(close(x[i], std::sqrt(std::abs(ref[i]))));
            }
            check_aligned_and_padded(x);
        };

        // -- size mismatch (the one test that passes against the stubs) --------

        test("elementwise_product/quotient throw invalid_argument on size mismatch (static vs dynamic)") = [] {
            mi::Vector<double, 5> a{1, 2, 3, 4, 5};
            mi::Vector<double> b(4);
            expect(throws<std::invalid_argument>([&] { a.elementwise_product(b); }));
            expect(throws<std::invalid_argument>([&] { a.elementwise_quotient(b); }));

            mi::Vector<double> c(5);
            mi::Vector<double> d(6);
            expect(throws<std::invalid_argument>([&] { c.elementwise_product(d); }));
            expect(throws<std::invalid_argument>([&] { c.elementwise_quotient(d); }));
        };

        // -- both element types and storage strategies -------------------------

        test("works for Vector<float, 17> (static, non-dividing extent)") = [] {
            mi::Vector<float, 17> x{};
            for (int i = 0; i < 17; ++i) {
                x[static_cast<std::size_t>(i)] = 0.1F * static_cast<float>(i + 1); // positive
            }
            std::vector<float> ref(x.begin(), x.end());
            x.sqrt();
            for (std::size_t i = 0; i < ref.size(); ++i) {
                expect(close(x[i], std::sqrt(ref[i]), 1e-5F));
            }
            check_aligned_and_padded(x);
        };

        test("works for Vector<double>(1000) (dynamic extent)") = [] {
            mi::Vector<double> x(1000);
            for (std::size_t i = 0; i < 1000; ++i) {
                x[i] = 0.01 * static_cast<double>(i) - 5.0;
            }
            std::vector<double> ref(x.begin(), x.end());
            x.exp();
            for (std::size_t i = 0; i < 1000; ++i) {
                expect(close(x[i], std::exp(ref[i])));
            }
            check_aligned_and_padded(x);
        };
    };

    return 0;
}
