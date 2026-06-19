#pragma once

#include "instrument/timing.hpp"

#include <ginkgo/ginkgo.hpp>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <unordered_map>

namespace miscibility::instrument {

class Context;

/// A Ginkgo logger that times every operator ``apply`` and names the region after the operand.
///
/// The logger is attached to an executor by a :cpp:`Context`. Because it reports
/// that it :cpp:`needs_propagation`, Ginkgo copies it onto every object created
/// on that executor, so it observes the applies of matrices, solvers, and
/// preconditioners alike. On each apply it asks the owning :cpp:`Context` for the
/// operand's registered name; if one exists it opens an :cpp:`instrument` timing
/// region with that name, and closes it when the apply completes. Applies of
/// unregistered operators are ignored, and because the timing regions use a
/// manual stack, the nested applies of composite operators nest correctly.
class TimingLogger : public gko::log::Logger {
public:
    /// Builds a logger that reports apply timings against ``context``'s name registry.
    ///
    /// :param context: The context whose registry maps operators to region names;
    ///     must outlive the logger.
    explicit TimingLogger(const Context* context) :
        gko::log::Logger(gko::log::Logger::linop_apply_started_mask | gko::log::Logger::linop_apply_completed_mask),
        context_{context}
    {
    }

    /// Opens a timing region named after ``system`` when an apply begins.
    ///
    /// Does nothing if the operand is not registered with the context.
    ///
    /// :param system: The operator being applied.
    /// :param right_hand_side: The apply's input operand (unused).
    /// :param solution: The apply's output operand (unused).
    void on_linop_apply_started(const gko::LinOp* system, const gko::LinOp* right_hand_side,
                                const gko::LinOp* solution) const override;

    /// Closes the timing region opened for ``system`` when its apply completes.
    ///
    /// :param system: The operator whose apply has finished.
    /// :param right_hand_side: The apply's input operand (unused).
    /// :param solution: The apply's output operand (unused).
    void on_linop_apply_completed(const gko::LinOp* system, const gko::LinOp* right_hand_side,
                                  const gko::LinOp* solution) const override;

    /// Always true, so the executor propagates this logger onto the objects it creates.
    [[nodiscard]] bool needs_propagation() const override { return true; }

private:
    const Context* context_;
};

/// Owns the Ginkgo executor and the machinery that ties operator applies to named timings.
///
/// A ``Context`` is the root every operator wrapper hangs from: it holds the
/// executor that data lives on, a registry mapping each wrapped ``gko::LinOp`` to
/// a human-readable name, and (when timing is enabled) a :cpp:`TimingLogger`
/// attached to the executor. With the logger in place, every ``apply`` on a
/// registered operator is recorded as an :cpp:`instrument` timing region named
/// after that operator, so a timing report reads in terms of the objects the
/// program built rather than anonymous Ginkgo calls.
///
/// Downstream libraries embed a ``Context`` inside their own context type. It is
/// neither copyable nor movable — it owns a logger bound to the executor — so
/// callers hold it by reference and let the wrappers register themselves through
/// their back-pointer to it.
///
/// .. code-block:: cpp
///
///     Context ctx;                       // ReferenceExecutor, timing on
///     auto exec = ctx.executor();
///     // ... build wrappers on exec; their applies now show up in query()/report()
class Context {
public:
    /// Creates a context over ``exec``, optionally attaching the timing logger.
    ///
    /// :param exec: Executor that owns the data; defaults to a fresh
    ///     ``ReferenceExecutor``.
    /// :param enable_timing: When true, a :cpp:`TimingLogger` is constructed and
    ///     added to ``exec`` so applies are timed; when false, the context is a
    ///     plain executor-plus-registry holder and records nothing.
    explicit Context(std::shared_ptr<const gko::Executor> exec = gko::ReferenceExecutor::create(),
                     bool enable_timing = true);

    Context(const Context&) = delete;
    Context& operator=(const Context&) = delete;
    Context(Context&&) = delete;
    Context& operator=(Context&&) = delete;

    /// Detaches the timing logger from the executor, if one was attached.
    ~Context();

    /// The executor that data and operators live on.
    [[nodiscard]] const std::shared_ptr<const gko::Executor>& executor() const noexcept { return executor_; }

    /// Whether this context attached a timing logger and is recording applies.
    [[nodiscard]] bool timing_enabled() const noexcept { return timing_enabled_; }

    /// Records the name to report for an operator, overwriting any previous one.
    ///
    /// The entry is keyed on the raw pointer ``apply`` reports, so lookups during
    /// logging are exact. Entries are not reference-counted; they persist for the
    /// life of the context.
    ///
    /// :param op: The operator to name; the raw ``gko::LinOp`` pointer.
    /// :param name: The human-readable name to report for it.
    void register_name(const gko::LinOp* op, std::string name);

    /// Looks up the name registered for an operator.
    ///
    /// :param op: The operator pointer to resolve.
    /// :returns: The registered name, or ``std::nullopt`` if ``op`` was never registered.
    [[nodiscard]] std::optional<std::string> name_of(const gko::LinOp* op) const;

private:
    std::shared_ptr<const gko::Executor> executor_;
    bool timing_enabled_ = false;
    mutable std::mutex registry_mutex_;
    std::unordered_map<const gko::LinOp*, std::string> registry_;
    std::shared_ptr<TimingLogger> logger_;
};

/// The value-semantic base every operator wrapper is built on.
///
/// An ``OperatorHandle`` *owns* a ``gko::LinOp`` rather than *being* one: it
/// holds a shared pointer to the wrapped operator, a back-pointer to its
/// :cpp:`Context`, and the name the operator is registered under. Derived
/// wrappers (``Vector``, ``DenseMatrix``, ``Preconditioner``, composites, ...)
/// add typed access on top of this type-erased base.
///
/// A handle converts implicitly to ``std::shared_ptr<const gko::LinOp>``, which
/// is the seam that lets a wrapper be passed straight into raw Ginkgo APIs that
/// expect a ``LinOp`` — a solver factory's ``generate``, ``BlockOperator::create``,
/// and so on:
///
/// .. code-block:: cpp
///
///     auto solver = factory->generate(my_handle);   // implicit conversion
///
/// Handles are copyable: a copy shares the same underlying operator and context
/// and keeps the same name and registration.
class OperatorHandle {
public:
    /// The wrapped operator.
    [[nodiscard]] const std::shared_ptr<gko::LinOp>& linop() const noexcept { return linop_; }

    /// Implicitly exposes the wrapped operator for raw Ginkgo APIs that take a ``LinOp``.
    operator std::shared_ptr<const gko::LinOp>() const noexcept { return linop_; }

    /// The context this handle belongs to, used by the operator DSL to recover a
    /// context from its operands.
    [[nodiscard]] Context& context() const noexcept { return *context_; }

    /// The name this operator is registered and timed under.
    [[nodiscard]] const std::string& name() const noexcept { return name_; }

    /// The dimensions of the wrapped operator.
    [[nodiscard]] gko::dim<2> size() const noexcept { return linop_->get_size(); }

protected:
    /// Wraps ``op``, ties it to ``ctx``, and registers it under ``name`` so its applies are timed.
    ///
    /// :param ctx: The context the handle binds to; must outlive the handle.
    /// :param name: The name to register and report the operator under.
    /// :param op: The Ginkgo operator to wrap.
    OperatorHandle(Context& ctx, std::string name, std::shared_ptr<gko::LinOp> op);

private:
    Context* context_;
    std::string name_;
    std::shared_ptr<gko::LinOp> linop_;
};

inline void TimingLogger::on_linop_apply_started(const gko::LinOp* system, const gko::LinOp* /*right_hand_side*/,
                                                 const gko::LinOp* /*solution*/) const
{
    if (auto name = context_->name_of(system)) {
        start(*name);
    }
}

inline void TimingLogger::on_linop_apply_completed(const gko::LinOp* system, const gko::LinOp* /*right_hand_side*/,
                                                   const gko::LinOp* /*solution*/) const
{
    if (auto name = context_->name_of(system)) {
        stop(*name);
    }
}

inline Context::Context(std::shared_ptr<const gko::Executor> exec, bool enable_timing) :
    executor_{std::move(exec)}, timing_enabled_{enable_timing}
{
    if (timing_enabled_) {
        logger_ = std::make_shared<TimingLogger>(this);
        // Attaching/detaching a logger mutates the executor's logger list, but the
        // public contract stores the executor as const; const_cast bridges the two.
        const_cast<gko::Executor&>(*executor_).add_logger(logger_);
    }
}

inline Context::~Context()
{
    if (logger_) {
        const_cast<gko::Executor&>(*executor_).remove_logger(logger_.get());
    }
}

inline void Context::register_name(const gko::LinOp* op, std::string name)
{
    const std::lock_guard<std::mutex> lock{registry_mutex_};
    registry_[op] = std::move(name);
}

inline std::optional<std::string> Context::name_of(const gko::LinOp* op) const
{
    const std::lock_guard<std::mutex> lock{registry_mutex_};
    if (auto it = registry_.find(op); it != registry_.end()) {
        return it->second;
    }
    return std::nullopt;
}

inline OperatorHandle::OperatorHandle(Context& ctx, std::string name, std::shared_ptr<gko::LinOp> op) :
    context_{&ctx}, name_{std::move(name)}, linop_{std::move(op)}
{
    context_->register_name(linop_.get(), name_);
}

} // namespace miscibility::instrument
