#pragma once

#include "instrument/array.hpp"
#include "instrument/context.hpp"
#include "instrument/sparse_matrix.hpp"
#include "instrument/vector.hpp"

#include <cstdint>
#include <ginkgo/ginkgo.hpp>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>

namespace miscibility::instrument {

/// Which preconditioner a :cpp:`Preconditioner` builds.
enum class PrecondType : std::uint8_t {
    None,        ///< Identity — no preconditioning.
    Jacobi,      ///< Diagonal (point) Jacobi.
    BlockJacobi, ///< Block Jacobi; block size from :cpp:`PrecondOptions::max_block_size`.
    Ilu,         ///< Incomplete LU.
    Ic,          ///< Incomplete Cholesky.
    Isai,        ///< Incomplete sparse approximate inverse.
    Sor,         ///< Successive over-relaxation.
    GaussSeidel, ///< Gauss–Seidel.
    Custom,      ///< User-provided custom preconditioner.
};

/// Storage precision a preconditioner is built and applied in.
///
/// Selecting a precision below the system's lets the preconditioner use less
/// memory and bandwidth; pair it with :cpp:`PrecondOptions::refine` to recover
/// the system's accuracy at apply time.
enum class Precision : std::uint8_t {
    Double,   ///< Full ``double`` precision.
    Single,   ///< ``float``.
    Half,     ///< IEEE half (``gko::half``).
    BFloat16, ///< bfloat16 (``gko::bfloat16``).
};

/// Configuration for building a :cpp:`Preconditioner`.
struct PrecondOptions {
    PrecondType type = PrecondType::Jacobi; ///< Which preconditioner to build.
    Precision storage = Precision::Double;  ///< Precision the data is stored and applied in.
    bool refine = false;                    ///< Wrap in iterative refinement so apply is full-precision accurate.
    unsigned refine_iters = 1;              ///< Refinement sweeps when :cpp:`refine` is set.
    gko::size_type max_block_size = 1;      ///< Block size for ``BlockJacobi``.
};

namespace detail {

template<class T, class Low>
class PrecisionCastOp : public gko::EnableLinOp<PrecisionCastOp<T, Low>>,
                        public gko::EnableCreateMethod<PrecisionCastOp<T, Low>> {
public:
    explicit PrecisionCastOp(std::shared_ptr<const gko::Executor> exec) :
        gko::EnableLinOp<PrecisionCastOp<T, Low>>(std::move(exec))
    {
    }

    PrecisionCastOp(std::shared_ptr<const gko::Executor> exec, std::shared_ptr<const gko::LinOp> inner) :
        gko::EnableLinOp<PrecisionCastOp<T, Low>>(exec, inner->get_size()), inner_{std::move(inner)}
    {
    }

protected:
    void apply_impl(const gko::LinOp* b, gko::LinOp* x) const override;
    void apply_impl(const gko::LinOp* alpha, const gko::LinOp* b, const gko::LinOp* beta, gko::LinOp* x) const override;

private:
    std::shared_ptr<const gko::LinOp> inner_;
};

template<class T, class Low> void PrecisionCastOp<T, Low>::apply_impl(const gko::LinOp* b, gko::LinOp* x) const
{
    // FIXME: I do not know if this is necessary. Ginkgo is supposed to work with multi-precision semi-automatically
    using high_dense = gko::matrix::Dense<T>;
    using low_dense = gko::matrix::Dense<Low>;
    const auto* b_high = gko::as<high_dense>(b);
    auto* x_high = gko::as<high_dense>(x);
    auto exec = this->get_executor();

    auto b_low = low_dense::create(exec, b_high->get_size());
    b_high->convert_to(b_low);
    auto x_low = low_dense::create(exec, x_high->get_size());
    inner_->apply(b_low, x_low);
    x_low->convert_to(x_high);
}

template<class T, class Low>
void PrecisionCastOp<T, Low>::apply_impl(const gko::LinOp* alpha, const gko::LinOp* b, const gko::LinOp* beta,
                                         gko::LinOp* x) const
{
    // FIXME: I do not know if this is necessary. Ginkgo is supposed to work with multi-precision semi-automatically
    auto* x_high = gko::as<gko::matrix::Dense<T>>(x);
    auto correction = x_high->clone();
    apply_impl(b, correction.get());
    x_high->scale(beta);
    x_high->add_scaled(alpha, correction);
}

template<class V>
std::shared_ptr<gko::LinOp> build_simple_preconditioner(const std::shared_ptr<const gko::Executor>& exec,
                                                        std::shared_ptr<const gko::LinOp> matrix, PrecondType type,
                                                        gko::size_type max_block_size)
{
    // FIXME: Jacobi moves the matrix. Not sure if that is correct.
    switch (type) {
    case PrecondType::None:
        return gko::share(gko::matrix::Identity<V>::create(exec, matrix->get_size()[0]));
    case PrecondType::Jacobi:
        return gko::share(
            gko::preconditioner::Jacobi<V, int>::build().with_max_block_size(1U).on(exec)->generate(std::move(matrix)));
    case PrecondType::BlockJacobi:
        return gko::share(gko::preconditioner::Jacobi<V, int>::build()
                              .with_max_block_size(static_cast<gko::uint32>(max_block_size))
                              .on(exec)
                              ->generate(std::move(matrix)));
    default:
        throw std::runtime_error{"miscibility::instrument::Preconditioner reduced precision supports only "
                                 "None/Jacobi/BlockJacobi"};
    }
}

} // namespace detail

/// A preconditioner ``M`` whose apply computes ``z = M``\ :sup:`-1`\ ``r``.
///
/// One class covers the common preconditioners, selected at construction through
/// :cpp:`PrecondOptions`. Being an :cpp:`OperatorHandle`, it converts to a
/// ``gko::LinOp`` and drops into a solver via ``with_generated_preconditioner``.
///
/// .. code-block:: cpp
///
///     SparseMatrix<double> a{ctx, "A", pattern};
///     Preconditioner<double> m{ctx, "M", a, {.type = PrecondType::Jacobi}};
///     m.apply(residual, correction);   // correction = M^-1 * residual
///
/// The preconditioner can be stored in a precision below the system's (see
/// :cpp:`Precision`); with :cpp:`PrecondOptions::refine` set it is wrapped in
/// iterative refinement so its apply is still accurate to ``T``, and callers see
/// an ordinary full-precision preconditioner regardless of storage.
///
/// :tparam T: Floating-point value type of the system.
template<Scalar T = double> class Preconditioner : public OperatorHandle {
public:
    /// Generates a preconditioner from ``matrix`` according to ``opts``.
    ///
    /// :param ctx: Context the preconditioner lives on and registers with.
    /// :param name: Name reported for this operator in timing output.
    /// :param matrix: The system matrix to precondition.
    /// :param opts: Preconditioner type, storage precision, and refinement settings.
    Preconditioner(Context& ctx, std::string name, const SparseMatrix<T>& matrix, PrecondOptions opts = {});

    /// Adopts a caller-supplied preconditioner operator (the bring-your-own / matrix-free route).
    ///
    /// :param ctx: Context the preconditioner lives on and registers with.
    /// :param name: Name reported for this operator in timing output.
    /// :param custom: Any ``LinOp`` whose apply computes ``M``\ :sup:`-1`\ ``r``.
    Preconditioner(Context& ctx, std::string name, std::shared_ptr<const gko::LinOp> custom);

    /// Applies the preconditioner: ``z = M``\ :sup:`-1`\ ``r``.
    ///
    /// :param r: The residual to precondition.
    /// :param z: The result; overwritten, and must not alias ``r``.
    void apply(const Vector<T>& r, Vector<T>& z) const;

    /// The preconditioner type (``None`` for an adopted custom operator).
    [[nodiscard]] PrecondType type() const noexcept { return type_; }
    /// The storage precision.
    [[nodiscard]] Precision storage() const noexcept { return storage_; }

private:
    static std::shared_ptr<gko::LinOp> generate(Context& ctx, const SparseMatrix<T>& matrix,
                                                const PrecondOptions& opts);

    static std::shared_ptr<gko::LinOp> build_double(const std::shared_ptr<const gko::Executor>& exec,
                                                    std::shared_ptr<const gko::LinOp> matrix,
                                                    const PrecondOptions& opts);

    template<class Low>
    static std::shared_ptr<gko::LinOp> build_reduced(const std::shared_ptr<const gko::Executor>& exec,
                                                     const SparseMatrix<T>& matrix, const PrecondOptions& opts);

    static std::shared_ptr<gko::LinOp> refine(const std::shared_ptr<const gko::Executor>& exec,
                                              std::shared_ptr<const gko::LinOp> system,
                                              std::shared_ptr<const gko::LinOp> preconditioner, unsigned iters);

    PrecondType type_ = PrecondType::None;
    Precision storage_ = Precision::Double;
};

template<Scalar T>
Preconditioner<T>::Preconditioner(Context& ctx, std::string name, const SparseMatrix<T>& matrix, PrecondOptions opts) :
    OperatorHandle(ctx, std::move(name), generate(ctx, matrix, opts)), type_{opts.type}, storage_{opts.storage}
{
}

template<Scalar T>
Preconditioner<T>::Preconditioner(Context& ctx, std::string name, std::shared_ptr<const gko::LinOp> custom) :
    OperatorHandle(ctx, std::move(name), std::const_pointer_cast<gko::LinOp>(std::move(custom))), type_(PrecondType::Custom)
{
}

template<Scalar T>
std::shared_ptr<gko::LinOp> Preconditioner<T>::build_double(const std::shared_ptr<const gko::Executor>& exec,
                                                            std::shared_ptr<const gko::LinOp> matrix,
                                                            const PrecondOptions& opts)
{
    // FIXME: Add support for remaining preconditioners
    switch (opts.type) {
    case PrecondType::None:
    case PrecondType::Jacobi:
    case PrecondType::BlockJacobi:
        return detail::build_simple_preconditioner<T>(exec, std::move(matrix), opts.type, opts.max_block_size);
    case PrecondType::Ilu: {
        auto factors = gko::factorization::ParIlu<T, int>::build().on(exec)->generate(std::move(matrix));
        return gko::share(gko::preconditioner::Ilu<>::build().on(exec)->generate(std::move(factors)));
    }
    case PrecondType::Ic: {
        auto factors = gko::factorization::ParIc<T, int>::build().on(exec)->generate(std::move(matrix));
        return gko::share(gko::preconditioner::Ic<>::build().on(exec)->generate(std::move(factors)));
    }
    default:
        throw std::runtime_error{"miscibility::instrument::Preconditioner type not supported in this build"};
    }
}

template<Scalar T>
template<class Low>
std::shared_ptr<gko::LinOp> Preconditioner<T>::build_reduced(const std::shared_ptr<const gko::Executor>& exec,
                                                             const SparseMatrix<T>& matrix, const PrecondOptions& opts)
{
    // FIXME: see if this is necessary due to multi-precision build
    auto matrix_low = gko::matrix::Csr<Low, int>::create(exec);
    gko::as<gko::matrix::Csr<T, int>>(matrix.linop().get())->convert_to(matrix_low);
    auto inner = detail::build_simple_preconditioner<Low>(exec, std::move(matrix_low), opts.type, opts.max_block_size);
    return gko::share(detail::PrecisionCastOp<T, Low>::create(exec, std::move(inner)));
}

template<Scalar T>
std::shared_ptr<gko::LinOp> Preconditioner<T>::refine(const std::shared_ptr<const gko::Executor>& exec,
                                                      std::shared_ptr<const gko::LinOp> system,
                                                      std::shared_ptr<const gko::LinOp> preconditioner, unsigned iters)
{
    return gko::share(
        gko::solver::Ir<T>::build()
            .with_generated_solver(std::move(preconditioner))
            .with_criteria(gko::stop::Iteration::build().with_max_iters(static_cast<gko::size_type>(iters)))
            .on(exec)
            ->generate(std::move(system)));
}

template<Scalar T>
std::shared_ptr<gko::LinOp> Preconditioner<T>::generate(Context& ctx, const SparseMatrix<T>& matrix,
                                                        const PrecondOptions& opts)
{
    auto exec = ctx.executor();
    std::shared_ptr<gko::LinOp> preconditioner;
    switch (opts.storage) {
    case Precision::Double:
        preconditioner = build_double(exec, matrix.linop(), opts);
        break;
    case Precision::Single:
        preconditioner = build_reduced<float>(exec, matrix, opts);
        break;
    case Precision::Half:
        preconditioner = build_reduced<gko::half>(exec, matrix, opts);
        break;
    case Precision::BFloat16:
        preconditioner = build_reduced<gko::bfloat16>(exec, matrix, opts);
        break;
    }
    if (opts.refine) {
        preconditioner = refine(exec, matrix.linop(), preconditioner, opts.refine_iters);
    }
    return preconditioner;
}

template<Scalar T> void Preconditioner<T>::apply(const Vector<T>& r, Vector<T>& z) const
{
    linop()->apply(r.linop(), z.linop());
}

} // namespace miscibility::instrument
