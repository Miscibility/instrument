// test_vector.cpp -- unit tests for miscibility::instrument Vector.

#include "instrument/context.hpp"
#include "instrument/vector.hpp"

#include <algorithm>
#include <array>
#include <boost/ut.hpp>
#include <cmath>
#include <ginkgo/ginkgo.hpp>
#include <memory>
#include <numeric>

namespace mi = miscibility::instrument;

namespace {

bool accepts_linop(const std::shared_ptr<const gko::LinOp>& op) { return op != nullptr; }

} // namespace

int main()
{
    using namespace boost::ut;

    suite<"Vector construction"> construction_suite = [] {
        test("sized constructor zero-fills and reports its length") = [] {
            mi::Context ctx;
            mi::Vector<double> v{ctx, "v", 3};
            expect(v.size() == 3_u);
            expect(v.at(0) == 0.0_d);
            expect(v.at(1) == 0.0_d);
            expect(v.at(2) == 0.0_d);
        };

        test("value constructor fills every element") = [] {
            mi::Context ctx;
            mi::Vector<double> v{ctx, "v", 3, 7.0};
            expect(v.at(0) == 7.0_d);
            expect(v.at(1) == 7.0_d);
            expect(v.at(2) == 7.0_d);
        };

        test("initializer-list constructor copies the listed values") = [] {
            mi::Context ctx;
            mi::Vector<double> v{ctx, "v", {1.0, 2.0, 3.0}};
            expect(v.size() == 3_u);
            expect(v.at(0) == 1.0_d);
            expect(v.at(1) == 2.0_d);
            expect(v.at(2) == 3.0_d);
        };

        test("raw-pointer constructor copies and does not alias the source") = [] {
            mi::Context ctx;
            std::array<double, 3> source{1.0, 2.0, 3.0};
            mi::Vector<double> v{ctx, "v", source.data(), source.size()};
            source[0] = 99.0;
            expect(v.at(0) == 1.0_d) << "constructor must copy, not view";
        };
    };

    suite<"Vector container surface"> container_suite = [] {
        test("operator[] reads and writes round-trip") = [] {
            mi::Context ctx;
            mi::Vector<double> v{ctx, "v", 3};
            v[1] = 5.0;
            expect(v[1] == 5.0_d);
        };

        test("at throws std::out_of_range past the end") = [] {
            mi::Context ctx;
            mi::Vector<double> v{ctx, "v", 3};
            expect(throws<std::out_of_range>([&] { (void)v.at(3); }));
        };

        test("as_span spans exactly the logical elements") = [] {
            mi::Context ctx;
            mi::Vector<double> v{ctx, "v", 4};
            expect(v.as_span().size() == v.size());
        };

        test("std::accumulate over the range sums the elements") = [] {
            mi::Context ctx;
            mi::Vector<double> v{ctx, "v", {1.0, 2.0, 3.0, 4.0}};
            const double total = std::accumulate(v.begin(), v.end(), 0.0);
            expect(total == 10.0_d);
        };

        test("std::transform over the range mutates in place") = [] {
            mi::Context ctx;
            mi::Vector<double> v{ctx, "v", {1.0, 2.0, 3.0}};
            std::transform(v.begin(), v.end(), v.begin(), [](double x) { return x * x; });
            expect(v.at(0) == 1.0_d);
            expect(v.at(1) == 4.0_d);
            expect(v.at(2) == 9.0_d);
        };
    };

    suite<"Vector math"> math_suite = [] {
        test("fill sets every element") = [] {
            mi::Context ctx;
            mi::Vector<double> v{ctx, "v", 3};
            v.fill(2.5);
            expect(v.at(0) == 2.5_d);
            expect(v.at(1) == 2.5_d);
            expect(v.at(2) == 2.5_d);
        };

        test("scale multiplies every element") = [] {
            mi::Context ctx;
            mi::Vector<double> v{ctx, "v", {1.0, 2.0, 3.0}};
            v.scale(2.0);
            expect(v.at(0) == 2.0_d);
            expect(v.at(1) == 4.0_d);
            expect(v.at(2) == 6.0_d);
        };

        test("add_scaled computes y + alpha*x elementwise") = [] {
            mi::Context ctx;
            mi::Vector<double> y{ctx, "y", {1.0, 2.0, 3.0}};
            mi::Vector<double> x{ctx, "x", {4.0, 5.0, 6.0}};
            y.add_scaled(1.0, x);
            expect(y.at(0) == 5.0_d);
            expect(y.at(1) == 7.0_d);
            expect(y.at(2) == 9.0_d);
        };

        test("sub_scaled computes y - alpha*x elementwise") = [] {
            mi::Context ctx;
            mi::Vector<double> y{ctx, "y", {10.0, 10.0, 10.0}};
            mi::Vector<double> x{ctx, "x", {1.0, 2.0, 3.0}};
            y.sub_scaled(2.0, x);
            expect(y.at(0) == 8.0_d);
            expect(y.at(1) == 6.0_d);
            expect(y.at(2) == 4.0_d);
        };

        test("dot returns the inner product") = [] {
            mi::Context ctx;
            mi::Vector<double> a{ctx, "a", {1.0, 2.0, 3.0}};
            mi::Vector<double> b{ctx, "b", {4.0, 5.0, 6.0}};
            expect(a.dot(b) == 32.0_d);
        };

        test("norm2 returns the Euclidean norm") = [] {
            mi::Context ctx;
            mi::Vector<double> v{ctx, "v", {3.0, 4.0}};
            expect(std::abs(v.norm2() - 5.0) < 1e-12);
        };

        test("norm1 returns the sum of absolute values") = [] {
            mi::Context ctx;
            mi::Vector<double> v{ctx, "v", {1.0, -2.0, 3.0}};
            expect(v.norm1() == 6.0_d);
        };

        test("compound operators delegate to the math methods") = [] {
            mi::Context ctx;
            mi::Vector<double> v{ctx, "v", {2.0, 4.0}};
            v *= 3.0;
            expect(v.at(0) == 6.0_d);
            v /= 2.0;
            expect(v.at(0) == 3.0_d);
        };

        test("size-mismatched add_scaled throws std::invalid_argument") = [] {
            mi::Context ctx;
            mi::Vector<double> y{ctx, "y", 3};
            mi::Vector<double> x{ctx, "x", 2};
            expect(throws<std::invalid_argument>([&] { y.add_scaled(1.0, x); }));
        };
    };

    suite<"Vector interop"> interop_suite = [] {
        test("view aliases the foreign buffer in both directions") = [] {
            mi::Context ctx;
            std::array<double, 3> buffer{1.0, 2.0, 3.0};
            auto v = mi::Vector<double>::view(ctx, "v", buffer.data(), buffer.size());
            v[0] = 42.0;
            expect(buffer[0] == 42.0_d) << "writes to the Vector reach the buffer";
            buffer[1] = 17.0;
            expect(v[1] == 17.0_d) << "writes to the buffer are visible through the Vector";
        };

        test("implicitly converts to shared_ptr<const gko::LinOp>") = [] {
            mi::Context ctx;
            mi::Vector<double> v{ctx, "v", 3};
            expect(accepts_linop(v));
        };
    };

    return 0;
}
