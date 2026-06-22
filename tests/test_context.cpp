// test_context.cpp -- unit tests for miscibility::instrument Context / TimingLogger / OperatorHandle.

#include "instrument/context.hpp"
#include "instrument/timing.hpp"

#include <boost/ut.hpp>
#include <ginkgo/ginkgo.hpp>
#include <memory>
#include <string>
#include <utility>

namespace mi = miscibility::instrument;

namespace {

// OperatorHandle's constructor is protected; this trivial subclass exposes it so
// tests can wrap an arbitrary LinOp the way the real wrappers (Vector, ...) will.
struct TestHandle : mi::OperatorHandle {
    TestHandle(mi::Context& ctx, std::string name, std::shared_ptr<gko::LinOp> op) :
        mi::OperatorHandle(ctx, std::move(name), std::move(op))
    {
    }
};

bool accepts_linop(const std::shared_ptr<const gko::LinOp>& op) { return op != nullptr; }

std::shared_ptr<gko::matrix::Dense<double>> make_diagonal(const std::shared_ptr<const gko::Executor>& exec,
                                                          double value)
{
    return gko::share(gko::initialize<gko::matrix::Dense<double>>({{value, 0.0}, {0.0, value}}, exec));
}

} // namespace

int main()
{
    using namespace boost::ut;

    suite<"Context"> context_suite = [] {
        test("default-constructs with a non-null ReferenceExecutor and timing enabled") = [] {
            mi::Context ctx;
            expect(ctx.executor() != nullptr);
            expect(ctx.timing_enabled());
            expect(dynamic_cast<const gko::ReferenceExecutor*>(ctx.executor().get()) != nullptr);
        };

        test("constructed with enable_timing=false reports timing disabled") = [] {
            mi::Context ctx{gko::ReferenceExecutor::create(), false};
            expect(not ctx.timing_enabled());
        };

        test("register_name then name_of returns the registered name") = [] {
            mi::Context ctx;
            auto exec = ctx.executor();
            auto op = make_diagonal(exec, 1.0);
            ctx.register_name(op.get(), "A");
            auto found = ctx.name_of(op.get());
            expect(found.has_value());
            expect(found.value() == std::string{"A"});
        };

        test("name_of an unregistered pointer returns nullopt") = [] {
            mi::Context ctx;
            const auto& exec = ctx.executor();
            auto op = make_diagonal(exec, 1.0);
            expect(not ctx.name_of(op.get()).has_value());
        };

        test("re-registering the same pointer overwrites the name") = [] {
            mi::Context ctx;
            auto exec = ctx.executor();
            auto op = make_diagonal(exec, 1.0);
            ctx.register_name(op.get(), "first");
            ctx.register_name(op.get(), "second");
            expect(ctx.name_of(op.get()).value() == std::string{"second"});
        };
    };

    suite<"TimingLogger"> timing_logger_suite = [] {
        test("apply on a registered operator records a timing region by name") = [] {
            mi::reset();
            mi::Context ctx;
            auto exec = ctx.executor();
            auto matrix = make_diagonal(exec, 2.0);
            TestHandle handle{ctx, "A", matrix};
            auto b = gko::initialize<gko::matrix::Dense<double>>({1.0, 1.0}, exec);
            auto x = gko::matrix::Dense<double>::create(exec, gko::dim<2>{2, 1});
            handle.linop()->apply(b, x);
            auto stats = mi::query("A");
            expect(stats.has_value());
            expect(stats->calls >= 1_u);
        };

        test("apply records nothing when timing is disabled") = [] {
            mi::reset();
            mi::Context ctx{gko::ReferenceExecutor::create(), false};
            auto exec = ctx.executor();
            auto matrix = make_diagonal(exec, 2.0);
            TestHandle handle{ctx, "A", matrix};
            auto b = gko::initialize<gko::matrix::Dense<double>>({1.0, 1.0}, exec);
            auto x = gko::matrix::Dense<double>::create(exec, gko::dim<2>{2, 1});
            handle.linop()->apply(b, x);
            expect(not mi::query("A").has_value());
        };

        test("apply on an unregistered operator does not crash and records nothing") = [] {
            mi::reset();
            mi::Context ctx;
            const auto& exec = ctx.executor();
            auto matrix = make_diagonal(exec, 2.0);
            auto b = gko::initialize<gko::matrix::Dense<double>>({1.0, 1.0}, exec);
            auto x = gko::matrix::Dense<double>::create(exec, gko::dim<2>{2, 1});
            expect(nothrow([&] { matrix->apply(b, x); }));
            expect(not mi::query("unregistered").has_value());
        };
    };

    suite<"OperatorHandle"> handle_suite = [] {
        test("implicitly converts to shared_ptr<const gko::LinOp>") = [] {
            mi::Context ctx;
            auto exec = ctx.executor();
            auto matrix = make_diagonal(exec, 1.0);
            TestHandle handle{ctx, "A", matrix};
            expect(accepts_linop(handle));
        };

        test("size() forwards the wrapped operator's dimensions") = [] {
            mi::Context ctx;
            auto exec = ctx.executor();
            auto matrix = make_diagonal(exec, 1.0);
            TestHandle handle{ctx, "A", matrix};
            expect(handle.size() == gko::dim<2>{2, 2});
        };

        test("name() returns the name the handle was constructed with") = [] {
            mi::Context ctx;
            auto exec = ctx.executor();
            auto matrix = make_diagonal(exec, 1.0);
            TestHandle handle{ctx, "my_operator", matrix};
            expect(handle.name() == std::string{"my_operator"});
        };

        test("registers its name in the context on construction") = [] {
            mi::Context ctx;
            auto exec = ctx.executor();
            auto matrix = make_diagonal(exec, 1.0);
            TestHandle handle{ctx, "registered", matrix};
            expect(ctx.name_of(matrix.get()).value() == std::string{"registered"});
        };
    };

    return 0;
}
