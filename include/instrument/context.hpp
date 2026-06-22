#pragma once

#include "instrument/timing.hpp"

#include <cstdint>
#include <ginkgo/ginkgo.hpp>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <type_traits>
#include <unordered_map>

namespace miscibility::instrument {

class Context;

/// The scalar value type an operator works in, recorded on every :cpp:`OperatorHandle`.
///
/// The operator DSL (``+``/``-``/``*``) is type-erased at the handle level, so it carries
/// this tag to recover the value type and build a matching ``gko::Combination`` /
/// ``gko::Composition``. ``Unknown`` marks handles with no single value type (an adopted raw
/// operator, a heterogeneous block) — using them in the DSL throws.
enum class ScalarType : std::uint8_t {
    Float,   ///< ``float``.
    Double,  ///< ``double``.
    Unknown, ///< No single value type (adopted/heterogeneous operator).
};

/// Maps a value type to its :cpp:`ScalarType` tag.
template<class T> [[nodiscard]] constexpr ScalarType scalar_type_of() noexcept
{
    if constexpr (std::is_same_v<T, float>) {
        return ScalarType::Float;
    }
    else if constexpr (std::is_same_v<T, double>) {
        return ScalarType::Double;
    }
    else {
        return ScalarType::Unknown;
    }
}

/// A per-operator Ginkgo logger that times that one operator's applies by name.
///
/// Unlike a propagating executor-level logger, a ``NamedApplyLogger`` is attached to exactly
/// one operator and carries its name as a member, so the apply hot path needs **no registry
/// lookup and no lock**: on each apply it simply opens (and later closes) an :cpp:`instrument`
/// timing region with its stored name. Because nested operators each carry their own logger,
/// the timing regions nest correctly through the manual timing stack.
///
/// The logger's lifetime is tied to the operator it is attached to (Ginkgo holds it by
/// ``shared_ptr``), which is also what reference-counts its entry in the owning
/// :cpp:`Context`'s name registry: the entry is added when the logger is created and removed
/// when the last reference to the logger goes away.
class NamedApplyLogger : public gko::log::Logger {
public:
    /// Builds a logger that times applies of the operator at ``key`` under ``name``.
    ///
    /// :param context: The context whose name registry this logger maintains an entry in;
    ///     must outlive the logger.
    /// :param key: The operator pointer this logger is attached to (the registry key).
    /// :param name: The region name reported for the operator's applies.
    NamedApplyLogger(Context* context, const gko::LinOp* key, std::string name);

    /// Removes this logger's entry from the context's name registry.
    ~NamedApplyLogger() override;

    NamedApplyLogger(const NamedApplyLogger&) = delete;
    NamedApplyLogger& operator=(const NamedApplyLogger&) = delete;
    NamedApplyLogger(NamedApplyLogger&&) = delete;
    NamedApplyLogger& operator=(NamedApplyLogger&&) = delete;

    /// Opens the timing region named after this operator when an apply begins.
    void on_linop_apply_started(const gko::LinOp* /*system*/, const gko::LinOp* /*right_hand_side*/,
                                const gko::LinOp* /*solution*/) const override
    {
        start(name_);
    }

    /// Closes the timing region when the apply completes.
    void on_linop_apply_completed(const gko::LinOp* /*system*/, const gko::LinOp* /*right_hand_side*/,
                                  const gko::LinOp* /*solution*/) const override
    {
        stop(name_);
    }

private:
    Context* context_;
    const gko::LinOp* key_;
    std::string name_;
};

/// Owns the Ginkgo executor and the machinery that ties operator applies to named timings.
///
/// A ``Context`` is the root every operator wrapper hangs from: it holds the executor that
/// data lives on and a registry mapping each wrapped ``gko::LinOp`` to a human-readable name.
/// When timing is enabled, each wrapper attaches a :cpp:`NamedApplyLogger` to its operator, so
/// every ``apply`` on a registered operator is recorded as an :cpp:`instrument` timing region
/// named after that operator — a timing report then reads in terms of the objects the program
/// built rather than anonymous Ginkgo calls.
///
/// The name registry is **not** consulted on the apply hot path (each logger carries its own
/// name), so applies take no registry lock. The registry exists only for introspection via
/// :cpp:`name_of`, and its entries are reference-counted against the loggers that own them, so
/// they are released when the operators they name are destroyed.
///
/// Downstream libraries embed a ``Context`` inside their own context type. It is neither
/// copyable nor movable, so callers hold it by reference and let the wrappers register
/// themselves through their back-pointer to it. **Every operator and handle built on a
/// ``Context`` must be destroyed before the ``Context`` itself.**
///
/// .. code-block:: cpp
///
///     Context ctx;                       // ReferenceExecutor, timing on
///     auto exec = ctx.executor();
///     // ... build wrappers on exec; their applies now show up in query()/report()
class Context {
public:
    /// Creates a context over ``exec``, optionally enabling per-operator apply timing.
    ///
    /// :param exec: Executor that owns the data; defaults to a fresh ``ReferenceExecutor``.
    /// :param enable_timing: When true, wrappers attach a :cpp:`NamedApplyLogger` to their
    ///     operator so applies are timed; when false, the context is a plain
    ///     executor-plus-registry holder and records nothing.
    explicit Context(std::shared_ptr<const gko::Executor> exec = gko::ReferenceExecutor::create(),
                     bool enable_timing = true) :
        executor_{std::move(exec)}, timing_enabled_{enable_timing}
    {
    }

    Context(const Context&) = delete;
    Context& operator=(const Context&) = delete;
    Context(Context&&) = delete;
    Context& operator=(Context&&) = delete;

    ~Context() = default;

    /// The executor that data and operators live on.
    [[nodiscard]] const std::shared_ptr<const gko::Executor>& executor() const noexcept { return executor_; }

    /// Whether this context attaches timing loggers and records applies.
    [[nodiscard]] bool timing_enabled() const noexcept { return timing_enabled_; }

    /// True when the executor is a host executor (Reference/OMP), where the wrappers'
    /// element access and iteration are valid. False for device executors (CUDA/HIP/...).
    [[nodiscard]] bool is_host() const noexcept { return executor_->get_master() == executor_; }

    /// Records (or overwrites) the name to report for an operator and adds a reference to it.
    ///
    /// The entry is keyed on the raw pointer ``apply`` reports. Entries are reference-counted:
    /// a matching :cpp:`unregister_name` releases one reference and the entry is erased at
    /// zero. A manual ``register_name`` with no matching ``unregister_name`` therefore persists
    /// for the life of the context.
    ///
    /// :param op: The operator to name; the raw ``gko::LinOp`` pointer.
    /// :param name: The human-readable name to report for it.
    void register_name(const gko::LinOp* op, std::string name);

    /// Releases one reference to an operator's registry entry, erasing it at zero references.
    ///
    /// :param op: The operator pointer to release.
    void unregister_name(const gko::LinOp* op);

    /// Looks up the name registered for an operator.
    ///
    /// :param op: The operator pointer to resolve.
    /// :returns: The registered name, or ``std::nullopt`` if ``op`` is not registered.
    [[nodiscard]] std::optional<std::string> name_of(const gko::LinOp* op) const;

private:
    struct Entry {
        std::string name;
        int refs = 0;
    };

    std::shared_ptr<const gko::Executor> executor_;
    bool timing_enabled_ = false;
    mutable std::mutex registry_mutex_;
    std::unordered_map<const gko::LinOp*, Entry> registry_;
};

/// The value-semantic base every operator wrapper is built on.
///
/// An ``OperatorHandle`` *owns* a ``gko::LinOp`` rather than *being* one: it holds a shared
/// pointer to the wrapped operator, a back-pointer to its :cpp:`Context`, the name the
/// operator is registered under, and (when timing is on) the :cpp:`NamedApplyLogger` attached
/// to it. Derived wrappers (``Vector``, ``DenseMatrix``, ``Preconditioner``, composites, ...)
/// add typed access on top of this type-erased base.
///
/// A handle converts implicitly to ``std::shared_ptr<const gko::LinOp>``, which is the seam
/// that lets a wrapper be passed straight into raw Ginkgo APIs that expect a ``LinOp`` — a
/// solver factory's ``generate``, ``BlockOperator::create``, and so on:
///
/// .. code-block:: cpp
///
///     auto solver = factory->generate(my_handle);   // implicit conversion
///
/// Handles are copyable: a copy shares the same underlying operator, context, name, logger,
/// and scalar-type tag.
class OperatorHandle {
public:
    /// The wrapped operator (may be null for a not-yet-bound deferred handle).
    [[nodiscard]] const std::shared_ptr<gko::LinOp>& linop() const noexcept { return linop_; }

    /// Implicitly exposes the wrapped operator for raw Ginkgo APIs that take a ``LinOp``.
    operator std::shared_ptr<const gko::LinOp>() const noexcept { return linop_; }

    /// The context this handle belongs to, used by the operator DSL to recover a context
    /// from its operands.
    [[nodiscard]] Context& context() const noexcept { return *context_; }

    /// The name this operator is registered and timed under.
    [[nodiscard]] const std::string& name() const noexcept { return name_; }

    /// The scalar value type the operator works in (used by the operator DSL).
    [[nodiscard]] ScalarType scalar_type() const noexcept { return scalar_type_; }

    /// The dimensions of the wrapped operator (``{0, 0}`` when not yet bound).
    [[nodiscard]] gko::dim<2> size() const noexcept { return linop_ ? linop_->get_size() : gko::dim<2>{}; }

protected:
    /// Wraps ``op``, ties it to ``ctx``, and (when timing is on) attaches a timing logger.
    ///
    /// :param ctx: The context the handle binds to; must outlive the handle.
    /// :param name: The name to register and report the operator under.
    /// :param op: The Ginkgo operator to wrap.
    /// :param scalar_type: The value type the operator works in.
    OperatorHandle(Context& ctx, std::string name, std::shared_ptr<gko::LinOp> op,
                   ScalarType scalar_type = ScalarType::Unknown) :
        context_{&ctx}, name_{std::move(name)}, linop_{std::move(op)}, scalar_type_{scalar_type}
    {
        attach_logger();
    }

    /// Constructs a handle with no operator yet, to be set later with :cpp:`rebind`.
    ///
    /// Used by operators whose ``LinOp`` does not exist until a matrix is bound (the iterative
    /// solver). :cpp:`size` is ``{0, 0}`` and the implicit ``LinOp`` conversion is null until
    /// :cpp:`rebind` is called.
    OperatorHandle(Context& ctx, std::string name, ScalarType scalar_type) :
        context_{&ctx}, name_{std::move(name)}, scalar_type_{scalar_type}
    {
    }

    /// Replaces the wrapped operator, moving the timing logger onto the new one.
    ///
    /// Detaches and releases the logger/registration of the previous operator (if any), adopts
    /// ``op``, and re-attaches a fresh timing logger under the same name.
    void rebind(std::shared_ptr<gko::LinOp> op)
    {
        if (linop_) {
            if (logger_) {
                linop_->remove_logger(logger_.get());
            }
            else {
                context_->unregister_name(linop_.get());
            }
        }
        logger_.reset();
        linop_ = std::move(op);
        attach_logger();
    }

private:
    void attach_logger()
    {
        if (!linop_) {
            return;
        }
        if (context_->timing_enabled()) {
            logger_ = std::make_shared<NamedApplyLogger>(context_, linop_.get(), name_);
            linop_->add_logger(logger_);
        }
        else {
            context_->register_name(linop_.get(), name_);
        }
    }

    Context* context_;
    std::string name_;
    std::shared_ptr<gko::LinOp> linop_;
    std::shared_ptr<NamedApplyLogger> logger_;
    ScalarType scalar_type_ = ScalarType::Unknown;
};

inline NamedApplyLogger::NamedApplyLogger(Context* context, const gko::LinOp* key, std::string name) :
    gko::log::Logger(gko::log::Logger::linop_apply_started_mask | gko::log::Logger::linop_apply_completed_mask),
    context_{context},
    key_{key},
    name_{std::move(name)}
{
    context_->register_name(key_, name_);
}

inline NamedApplyLogger::~NamedApplyLogger() { context_->unregister_name(key_); }

inline void Context::register_name(const gko::LinOp* op, std::string name)
{
    const std::lock_guard<std::mutex> lock{registry_mutex_};
    Entry& entry = registry_[op];
    entry.name = std::move(name);
    ++entry.refs;
}

inline void Context::unregister_name(const gko::LinOp* op)
{
    const std::lock_guard<std::mutex> lock{registry_mutex_};
    if (auto it = registry_.find(op); it != registry_.end() && --it->second.refs <= 0) {
        registry_.erase(it);
    }
}

inline std::optional<std::string> Context::name_of(const gko::LinOp* op) const
{
    const std::lock_guard<std::mutex> lock{registry_mutex_};
    if (auto it = registry_.find(op); it != registry_.end()) {
        return it->second.name;
    }
    return std::nullopt;
}

} // namespace miscibility::instrument
