#pragma once

#include "instrument/array.hpp"
#include "instrument/context.hpp"
#include "instrument/preconditioner.hpp"
#include "instrument/sparse_matrix.hpp"
#include "instrument/vector.hpp"

#include <chrono>
#include <cstdint>
#include <ginkgo/ginkgo.hpp>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace miscibility::instrument {

/// Which Krylov iterative solver an :cpp:`IterativeSolver` runs.
enum class SolverType : std::uint8_t {
    Cg,        ///< Conjugate gradient (SPD systems).
    Fcg,       ///< Flexible CG.
    PipeCg,    ///< Pipelined CG.
    Bicg,      ///< BiCG.
    Bicgstab,  ///< BiCGSTAB (nonsymmetric).
    Cgs,       ///< Conjugate gradient squared.
    Gmres,     ///< GMRES (nonsymmetric).
    CbGmres,   ///< Compressed-basis GMRES.
    Gcr,       ///< Generalized conjugate residual.
    Idr,       ///< Induced dimension reduction.
    Minres,    ///< Minimal residual (symmetric indefinite).
    Ir,        ///< Iterative refinement / Richardson.
    Chebyshev, ///< Chebyshev iteration.
};

/// The stopping criteria for a solve; the set ones are OR-combined.
///
/// Every field left unset contributes no criterion. At least one must be set —
/// :cpp:`IterativeSolver`'s constructor rejects an entirely empty ``StopOptions``
/// to avoid a solve that can never terminate.
struct StopOptions {
    std::optional<unsigned> max_iters = std::nullopt;                  ///< Stop after this many iterations.
    std::optional<double> rel_reduction = std::nullopt;                ///< Stop when ``‖r_k‖/‖r_0‖`` falls below this.
    std::optional<double> abs_tol = std::nullopt;                      ///< Stop when ``‖r_k‖`` falls below this.
    std::optional<double> rhs_tol = std::nullopt;                      ///< Stop when ``‖r_k‖/‖b‖`` falls below this.
    std::optional<std::chrono::nanoseconds> time_limit = std::nullopt; ///< Stop after this much wall time.
    bool implicit_residual = false;                                    ///< Use the cheaper recurrence residual norm.

    /// True when no criterion is set.
    [[nodiscard]] bool empty() const noexcept
    {
        return !max_iters && !rel_reduction && !abs_tol && !rhs_tol && !time_limit;
    }
};

/// Configuration for an :cpp:`IterativeSolver`.
struct SolverOptions {
    SolverType type = SolverType::Cg; ///< Which solver to run.
    StopOptions stop;                 ///< Stopping criteria (at least one required).
    unsigned krylov_dim = 100;        ///< Restart length for ``Gmres`` / ``CbGmres``.
    std::optional<PrecondOptions> preconditioner =
        std::nullopt;                ///< Preconditioner to generate from ``A`` per solve, if set.
    bool zero_initial_guess = false; ///< Zero ``x`` before solving instead of using it as a guess.
};

/// The outcome of a solve, populated from Ginkgo's convergence logger.
struct Statistics {
    bool converged = false;         ///< Whether a stopping criterion other than the iteration cap was met.
    gko::size_type iterations = 0;  ///< Iterations performed.
    double residual_norm = 0.0;     ///< Final residual norm ``‖r‖``.
    double relative_residual = 0.0; ///< ``residual_norm / ‖b‖``.
    bool regenerated = false;       ///< Whether this solve rebuilt the bound solver and preconditioner.
};

/// A configurable iterative solver that caches its bound solver and rebuilds only when the matrix changes.
///
/// One class fronts all of Ginkgo's Krylov solvers (selected through
/// :cpp:`SolverOptions`), hiding the factory/``generate`` two-phase model behind a
/// single bundled :cpp:`solve`. The expensive ``generate`` (and any preconditioner
/// build) is reused across solves and only redone when the system matrix changes —
/// detected through :cpp:`SparseMatrix::revision`, so an in-place value update in a
/// Newton or implicit-ODE loop correctly triggers a rebuild while a genuinely
/// unchanged matrix does not.
///
/// .. code-block:: cpp
///
///     IterativeSolver<double> solver{ctx, "cg",
///         {.type = SolverType::Cg, .stop = {.rel_reduction = 1e-10}}};
///     auto stats = solver.solve(a, b, x);
///     if (stats.converged) { /* x holds the solution */ }
///
/// Because it derives from :cpp:`OperatorHandle`, once it has been bound (by a :cpp:`solve`
/// or :cpp:`prepare`) the solver itself converts to a ``gko::LinOp`` — so a solver can be
/// dropped into a block, a composite, or used as another solver's preconditioner. Before the
/// first bind its ``LinOp`` is null and :cpp:`size` is ``{0, 0}``.
///
/// The cache makes an instance stateful and it is therefore non-copyable: use one solver per
/// thread and do not share it across concurrent solves.
///
/// :tparam T: Floating-point value type.
template<Scalar T = double> class IterativeSolver : public OperatorHandle {
public:
    /// Configures the solver; no matrix is bound yet.
    ///
    /// :param ctx: Context the solver lives on; its executor and timing registry are used.
    /// :param name: Name reported for the solve in timing output.
    /// :param opts: Solver type, stopping criteria, and preconditioner settings.
    /// :throws std::invalid_argument: if ``opts.stop`` has no criterion set.
    IterativeSolver(Context& ctx, std::string name, SolverOptions opts);

    IterativeSolver(const IterativeSolver&) = delete;
    IterativeSolver& operator=(const IterativeSolver&) = delete;
    IterativeSolver(IterativeSolver&&) = delete;
    IterativeSolver& operator=(IterativeSolver&&) = delete;
    ~IterativeSolver() = default;

    /// Solves ``A x = b``, rebuilding the bound solver only if ``matrix`` changed since the last solve.
    ///
    /// Change is detected by the matrix's ``LinOp`` identity and
    /// :cpp:`SparseMatrix::revision`, so an in-place value refill is picked up.
    ///
    /// :param matrix: The system matrix.
    /// :param b: Right-hand side.
    /// :param x: Solution; also the initial guess unless ``zero_initial_guess`` is set.
    /// :returns: Convergence statistics for this solve.
    Statistics solve(const SparseMatrix<T>& matrix, const Vector<T>& b, Vector<T>& x);

    /// Solves with a general operator (dense, block, matrix-free, a factorization).
    ///
    /// Caching keys on the operator's identity alone — arbitrary operators have no
    /// value-refill contract — so mutate-in-place is not detected here. Do not mix this
    /// overload and the :cpp:`SparseMatrix` overload on one solver instance: the two track
    /// the cache differently (this one ignores the revision counter).
    ///
    /// :param matrix: The system operator.
    /// :param b: Right-hand side.
    /// :param x: Solution; also the initial guess unless ``zero_initial_guess`` is set.
    /// :returns: Convergence statistics for this solve.
    Statistics solve(const OperatorHandle& matrix, const Vector<T>& b, Vector<T>& x);

    /// Eagerly builds and caches the bound solver for ``matrix``.
    ///
    /// Pays the ``generate`` / factorization cost now so the next :cpp:`solve` on
    /// the same matrix reports ``regenerated == false``, keeping it out of a hot or
    /// timed region.
    void prepare(const SparseMatrix<T>& matrix);

    /// Discards the cache so the next :cpp:`solve` rebuilds the bound solver.
    void force_regenerate() noexcept;

    /// The options this solver was configured with.
    [[nodiscard]] const SolverOptions& options() const noexcept { return options_; }

private:
    [[nodiscard]] std::vector<std::shared_ptr<const gko::stop::CriterionFactory>> build_criteria() const;
    [[nodiscard]] std::shared_ptr<gko::LinOp>
    build_bound(const std::shared_ptr<const gko::LinOp>& system,
                const std::shared_ptr<const gko::LinOp>& preconditioner) const;
    // Builds (and binds) the solver for `system`, generating a preconditioner from `matrix`
    // when one is configured and `matrix` is non-null, then records the cache key.
    void regenerate(const std::shared_ptr<const gko::LinOp>& system, const SparseMatrix<T>* matrix);
    Statistics run_apply(const Vector<T>& b, Vector<T>& x, bool regenerated);

    SolverOptions options_;

    const gko::LinOp* cached_key_ = nullptr;
    std::uint64_t cached_revision_ = 0;
    bool has_cache_ = false;
};

template<Scalar T>
IterativeSolver<T>::IterativeSolver(Context& ctx, std::string name, SolverOptions opts) :
    OperatorHandle(ctx, std::move(name), scalar_type_of<T>()), options_{opts}
{
    if (options_.stop.empty()) {
        throw std::invalid_argument{
            "miscibility::instrument::IterativeSolver requires at least one stopping criterion"};
    }
}

template<Scalar T>
std::vector<std::shared_ptr<const gko::stop::CriterionFactory>> IterativeSolver<T>::build_criteria() const
{
    auto exec = context().executor();
    const StopOptions& stop = options_.stop;
    std::vector<std::shared_ptr<const gko::stop::CriterionFactory>> criteria;

    if (stop.max_iters) {
        criteria.push_back(
            gko::stop::Iteration::build().with_max_iters(static_cast<gko::size_type>(*stop.max_iters)).on(exec));
    }
    auto add_residual_criterion = [&](double factor, gko::stop::mode baseline) {
        if (stop.implicit_residual) {
            criteria.push_back(
                gko::stop::ImplicitResidualNorm<T>::build().with_reduction_factor(factor).with_baseline(baseline).on(
                    exec));
        }
        else {
            criteria.push_back(
                gko::stop::ResidualNorm<T>::build().with_reduction_factor(factor).with_baseline(baseline).on(exec));
        }
    };
    if (stop.rel_reduction) {
        add_residual_criterion(*stop.rel_reduction, gko::stop::mode::initial_resnorm);
    }
    if (stop.abs_tol) {
        add_residual_criterion(*stop.abs_tol, gko::stop::mode::absolute);
    }
    if (stop.rhs_tol) {
        add_residual_criterion(*stop.rhs_tol, gko::stop::mode::rhs_norm);
    }
    if (stop.time_limit) {
        criteria.push_back(gko::stop::Time::build().with_time_limit(*stop.time_limit).on(exec));
    }
    return criteria;
}

template<Scalar T>
std::shared_ptr<gko::LinOp>
IterativeSolver<T>::build_bound(const std::shared_ptr<const gko::LinOp>& system,
                                const std::shared_ptr<const gko::LinOp>& preconditioner) const
{
    auto exec = context().executor();
    auto criteria = build_criteria();

    auto make = [&]<class Solver>() -> std::shared_ptr<gko::LinOp> {
        auto params = Solver::build();
        params.with_criteria(criteria);
        if constexpr (requires { params.with_generated_preconditioner(preconditioner); }) {
            if (preconditioner) {
                params.with_generated_preconditioner(preconditioner);
            }
        }
        if constexpr (requires { params.with_krylov_dim(gko::size_type{}); }) {
            params.with_krylov_dim(static_cast<gko::size_type>(options_.krylov_dim));
        }
        return gko::share(params.on(exec)->generate(system));
    };

    switch (options_.type) {
    case SolverType::Cg:
        return make.template operator()<gko::solver::Cg<T>>();
    case SolverType::Fcg:
        return make.template operator()<gko::solver::Fcg<T>>();
    case SolverType::PipeCg:
        return make.template operator()<gko::solver::PipeCg<T>>();
    case SolverType::Bicg:
        return make.template operator()<gko::solver::Bicg<T>>();
    case SolverType::Bicgstab:
        return make.template operator()<gko::solver::Bicgstab<T>>();
    case SolverType::Cgs:
        return make.template operator()<gko::solver::Cgs<T>>();
    case SolverType::Gmres:
        return make.template operator()<gko::solver::Gmres<T>>();
    case SolverType::CbGmres:
        return make.template operator()<gko::solver::CbGmres<T>>();
    case SolverType::Gcr:
        return make.template operator()<gko::solver::Gcr<T>>();
    case SolverType::Idr:
        return make.template operator()<gko::solver::Idr<T>>();
    case SolverType::Minres:
        return make.template operator()<gko::solver::Minres<T>>();
    case SolverType::Ir:
        return make.template operator()<gko::solver::Ir<T>>();
    case SolverType::Chebyshev:
        return make.template operator()<gko::solver::Chebyshev<T>>();
    }
    throw std::logic_error{"miscibility::instrument::IterativeSolver unknown solver type"};
}

template<Scalar T>
void IterativeSolver<T>::regenerate(const std::shared_ptr<const gko::LinOp>& system, const SparseMatrix<T>* matrix)
{
    std::shared_ptr<const gko::LinOp> preconditioner;
    if (matrix != nullptr && options_.preconditioner) {
        Preconditioner<T> generated{context(), name() + "_precond", *matrix, *options_.preconditioner};
        preconditioner = generated.linop();
    }
    rebind(build_bound(system, preconditioner));
    cached_key_ = system.get();
    cached_revision_ = matrix != nullptr ? matrix->revision() : 0;
    has_cache_ = true;
}

template<Scalar T> Statistics IterativeSolver<T>::run_apply(const Vector<T>& b, Vector<T>& x, bool regenerated)
{
    if (options_.zero_initial_guess) {
        x.fill(T{0});
    }
    auto exec = context().executor();
    std::shared_ptr<gko::log::Convergence<T>> logger = gko::share(gko::log::Convergence<T>::create());
    linop()->add_logger(logger);
    linop()->apply(b.linop(), x.linop());
    linop()->remove_logger(logger.get());

    Statistics stats;
    stats.regenerated = regenerated;
    stats.converged = logger->has_converged();
    stats.iterations = logger->get_num_iterations();
    if (const auto* residual = logger->get_residual_norm()) {
        stats.residual_norm =
            exec->copy_val_to_host(gko::as<gko::matrix::Dense<gko::remove_complex<T>>>(residual)->get_const_values());
    }
    const auto rhs_norm = b.norm2();
    stats.relative_residual = rhs_norm != 0 ? stats.residual_norm / rhs_norm : 0.0;
    return stats;
}

template<Scalar T> Statistics IterativeSolver<T>::solve(const SparseMatrix<T>& matrix, const Vector<T>& b, Vector<T>& x)
{
    const gko::LinOp* key = matrix.linop().get();
    const std::uint64_t revision = matrix.revision();
    const bool regenerate_needed = !has_cache_ || cached_key_ != key || cached_revision_ != revision;
    if (regenerate_needed) {
        regenerate(matrix.linop(), &matrix);
    }
    return run_apply(b, x, regenerate_needed);
}

template<Scalar T> Statistics IterativeSolver<T>::solve(const OperatorHandle& matrix, const Vector<T>& b, Vector<T>& x)
{
    const gko::LinOp* key = matrix.linop().get();
    const bool regenerate_needed = !has_cache_ || cached_key_ != key;
    if (regenerate_needed) {
        regenerate(matrix.linop(), nullptr);
    }
    return run_apply(b, x, regenerate_needed);
}

template<Scalar T> void IterativeSolver<T>::prepare(const SparseMatrix<T>& matrix)
{
    regenerate(matrix.linop(), &matrix);
}

template<Scalar T> void IterativeSolver<T>::force_regenerate() noexcept { has_cache_ = false; }

} // namespace miscibility::instrument
