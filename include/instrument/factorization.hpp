#pragma once

#include "instrument/array.hpp"
#include "instrument/context.hpp"
#include "instrument/dense_matrix.hpp"
#include "instrument/vector.hpp"

#include <algorithm>
#include <cstdint>
#include <ginkgo/ginkgo.hpp>
#include <lapacke.h>
#include <memory>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>

namespace miscibility::instrument {

/// Which dense factorization a :cpp:`Factorization` computes.
///
/// The choice should match what is known about the matrix: ``LU`` for a general
/// square system, ``Cholesky`` for symmetric positive-definite, ``LDLT`` for
/// symmetric indefinite, and ``QR`` for square systems or overdetermined
/// least-squares.
enum class FactorKind : std::uint8_t {
    LU,       ///< General square system (partial-pivoted LU).
    Cholesky, ///< Symmetric positive-definite.
    LDLT,     ///< Symmetric indefinite.
    QR,       ///< Square system or overdetermined least-squares.
};

/// Thrown when a factorization cannot proceed: a singular matrix for ``LU``/``LDLT``
/// or a non-positive-definite matrix for ``Cholesky``. The message names the kind
/// and the offending index.
class FactorizationError : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

namespace detail {

template<Scalar T> lapack_int lapacke_getrf(lapack_int m, lapack_int n, T* a, lapack_int lda, lapack_int* ipiv)
{
    if constexpr (std::is_same_v<T, double>) {
        return LAPACKE_dgetrf(LAPACK_ROW_MAJOR, m, n, a, lda, ipiv);
    }
    else {
        return LAPACKE_sgetrf(LAPACK_ROW_MAJOR, m, n, a, lda, ipiv);
    }
}

template<Scalar T>
lapack_int lapacke_getrs(lapack_int n, lapack_int nrhs, const T* a, lapack_int lda, const lapack_int* ipiv, T* b,
                         lapack_int ldb)
{
    if constexpr (std::is_same_v<T, double>) {
        return LAPACKE_dgetrs(LAPACK_ROW_MAJOR, 'N', n, nrhs, a, lda, ipiv, b, ldb);
    }
    else {
        return LAPACKE_sgetrs(LAPACK_ROW_MAJOR, 'N', n, nrhs, a, lda, ipiv, b, ldb);
    }
}

template<Scalar T> lapack_int lapacke_potrf(char uplo, lapack_int n, T* a, lapack_int lda)
{
    if constexpr (std::is_same_v<T, double>) {
        return LAPACKE_dpotrf(LAPACK_ROW_MAJOR, uplo, n, a, lda);
    }
    else {
        return LAPACKE_spotrf(LAPACK_ROW_MAJOR, uplo, n, a, lda);
    }
}

template<Scalar T>
lapack_int lapacke_potrs(char uplo, lapack_int n, lapack_int nrhs, const T* a, lapack_int lda, T* b, lapack_int ldb)
{
    if constexpr (std::is_same_v<T, double>) {
        return LAPACKE_dpotrs(LAPACK_ROW_MAJOR, uplo, n, nrhs, a, lda, b, ldb);
    }
    else {
        return LAPACKE_spotrs(LAPACK_ROW_MAJOR, uplo, n, nrhs, a, lda, b, ldb);
    }
}

template<Scalar T> lapack_int lapacke_sytrf(char uplo, lapack_int n, T* a, lapack_int lda, lapack_int* ipiv)
{
    if constexpr (std::is_same_v<T, double>) {
        return LAPACKE_dsytrf(LAPACK_ROW_MAJOR, uplo, n, a, lda, ipiv);
    }
    else {
        return LAPACKE_ssytrf(LAPACK_ROW_MAJOR, uplo, n, a, lda, ipiv);
    }
}

template<Scalar T>
lapack_int lapacke_sytrs(char uplo, lapack_int n, lapack_int nrhs, const T* a, lapack_int lda, const lapack_int* ipiv,
                         T* b, lapack_int ldb)
{
    if constexpr (std::is_same_v<T, double>) {
        return LAPACKE_dsytrs(LAPACK_ROW_MAJOR, uplo, n, nrhs, a, lda, ipiv, b, ldb);
    }
    else {
        return LAPACKE_ssytrs(LAPACK_ROW_MAJOR, uplo, n, nrhs, a, lda, ipiv, b, ldb);
    }
}

template<Scalar T> lapack_int lapacke_geqrf(lapack_int m, lapack_int n, T* a, lapack_int lda, T* tau)
{
    if constexpr (std::is_same_v<T, double>) {
        return LAPACKE_dgeqrf(LAPACK_ROW_MAJOR, m, n, a, lda, tau);
    }
    else {
        return LAPACKE_sgeqrf(LAPACK_ROW_MAJOR, m, n, a, lda, tau);
    }
}

template<Scalar T>
lapack_int lapacke_ormqr(lapack_int m, lapack_int nrhs, lapack_int k, const T* a, lapack_int lda, const T* tau, T* c,
                         lapack_int ldc)
{
    if constexpr (std::is_same_v<T, double>) {
        return LAPACKE_dormqr(LAPACK_ROW_MAJOR, 'L', 'T', m, nrhs, k, a, lda, tau, c, ldc);
    }
    else {
        return LAPACKE_sormqr(LAPACK_ROW_MAJOR, 'L', 'T', m, nrhs, k, a, lda, tau, c, ldc);
    }
}

template<Scalar T>
lapack_int lapacke_trtrs(lapack_int n, lapack_int nrhs, const T* a, lapack_int lda, T* b, lapack_int ldb)
{
    if constexpr (std::is_same_v<T, double>) {
        return LAPACKE_dtrtrs(LAPACK_ROW_MAJOR, 'U', 'N', 'N', n, nrhs, a, lda, b, ldb);
    }
    else {
        return LAPACKE_strtrs(LAPACK_ROW_MAJOR, 'U', 'N', 'N', n, nrhs, a, lda, b, ldb);
    }
}

/// The ``LinOp`` behind :cpp:`Factorization`: a factored matrix that applies as ``A``\ :sup:`-1`.
///
/// It holds a host copy of the factors (and the pivots or Householder scalars the
/// kind needs) and solves ``A x = b`` in :cpp:`apply_impl` with the matching
/// LAPACKE routine, so applying it is a triangular solve rather than a product.
/// A multi-column ``b`` solves every column at once. This is an internal type;
/// callers use :cpp:`Factorization`.
///
/// :tparam T: Floating-point value type.
template<Scalar T>
class DenseFactorizationOp : public gko::EnableLinOp<DenseFactorizationOp<T>>,
                             public gko::EnableCreateMethod<DenseFactorizationOp<T>> {
public:
    /// Constructs an empty operator on ``exec`` (used by Ginkgo's polymorphic-object machinery).
    explicit DenseFactorizationOp(std::shared_ptr<const gko::Executor> exec) :
        gko::EnableLinOp<DenseFactorizationOp<T>>(std::move(exec)), kind_{FactorKind::LU}
    {
    }

    /// Copies ``A``'s values and factors them immediately, so a bad matrix fails here.
    ///
    /// :param exec: Executor the operator lives on.
    /// :param shape: Shape of ``A`` as ``{rows, cols}``.
    /// :param values_row_major: ``A``'s entries in row-major order, ``rows*cols`` of them.
    /// :param kind: Which factorization to compute.
    /// :throws FactorizationError: if the factorization breaks down (singular or not positive-definite).
    DenseFactorizationOp(std::shared_ptr<const gko::Executor> exec, gko::dim<2> shape, const T* values_row_major,
                         FactorKind kind);

    /// The factorization kind this operator was built with.
    [[nodiscard]] FactorKind kind() const noexcept { return kind_; }
    /// Always false: a direct solve does not use ``x`` as an initial guess.
    [[nodiscard]] bool apply_uses_initial_guess() const override { return false; }

protected:
    void apply_impl(const gko::LinOp* b, gko::LinOp* x) const override;
    void apply_impl(const gko::LinOp* alpha, const gko::LinOp* b, const gko::LinOp* beta, gko::LinOp* x) const override;

private:
    using dense_type = gko::matrix::Dense<T>;

    void factorize();
    void solve_into(const dense_type& b, dense_type& x) const;

    [[nodiscard]] lapack_int rows() const noexcept { return static_cast<lapack_int>(system_shape_[0]); }
    [[nodiscard]] lapack_int cols() const noexcept { return static_cast<lapack_int>(system_shape_[1]); }

    gko::dim<2> system_shape_;
    FactorKind kind_;
    gko::array<T> factors_;
    gko::array<lapack_int> pivots_;
    gko::array<T> tau_;
};

template<Scalar T>
DenseFactorizationOp<T>::DenseFactorizationOp(std::shared_ptr<const gko::Executor> exec, gko::dim<2> shape,
                                              const T* values_row_major, FactorKind kind) :
    gko::EnableLinOp<DenseFactorizationOp<T>>(exec, gko::dim<2>{shape[1], shape[0]}),
    system_shape_{shape},
    kind_{kind},
    factors_{exec->get_master(), shape[0] * shape[1]}
{
    std::copy_n(values_row_major, shape[0] * shape[1], factors_.get_data());
    factorize();
}

template<Scalar T> void DenseFactorizationOp<T>::factorize()
{
    T* a = factors_.get_data();
    switch (kind_) {
    case FactorKind::LU: {
        pivots_ =
            gko::array<lapack_int>{factors_.get_executor(), static_cast<gko::size_type>(std::min(rows(), cols()))};
        const lapack_int info = lapacke_getrf<T>(rows(), cols(), a, cols(), pivots_.get_data());
        if (info > 0) {
            throw FactorizationError{"LU factorization found a singular matrix (zero pivot at index " +
                                     std::to_string(info) + ", 1-based)"};
        }
        break;
    }
    case FactorKind::Cholesky: {
        const lapack_int info = lapacke_potrf<T>('U', cols(), a, cols());
        if (info > 0) {
            throw FactorizationError{"Cholesky factorization found a non-positive-definite matrix (leading minor " +
                                     std::to_string(info) + ", 1-based)"};
        }
        break;
    }
    case FactorKind::LDLT: {
        pivots_ = gko::array<lapack_int>{factors_.get_executor(), static_cast<gko::size_type>(cols())};
        const lapack_int info = lapacke_sytrf<T>('U', cols(), a, cols(), pivots_.get_data());
        if (info > 0) {
            throw FactorizationError{"LDL^T factorization found a singular block (index " + std::to_string(info) +
                                     ", 1-based)"};
        }
        break;
    }
    case FactorKind::QR: {
        tau_ = gko::array<T>{factors_.get_executor(), static_cast<gko::size_type>(std::min(rows(), cols()))};
        lapacke_geqrf<T>(rows(), cols(), a, cols(), tau_.get_data());
        break;
    }
    }
}

template<Scalar T> void DenseFactorizationOp<T>::solve_into(const dense_type& b, dense_type& x) const
{
    const lapack_int nrhs = static_cast<lapack_int>(b.get_size()[1]);
    const T* a = factors_.get_const_data();

    const auto b_stride = static_cast<lapack_int>(b.get_stride());
    const auto x_stride = static_cast<lapack_int>(x.get_stride());

    if (kind_ == FactorKind::QR) {
        // gather b (rows x nrhs) into a compact buffer, form Q^T b, then back-substitute R.
        gko::array<T> work{factors_.get_executor(),
                           static_cast<gko::size_type>(rows()) * static_cast<gko::size_type>(nrhs)};
        for (lapack_int r = 0; r < rows(); ++r) {
            std::copy_n(b.get_const_values() + (r * b_stride), nrhs, work.get_data() + (r * nrhs));
        }
        lapacke_ormqr<T>(rows(), nrhs, cols(), a, cols(), tau_.get_const_data(), work.get_data(), nrhs);
        const lapack_int info = lapacke_trtrs<T>(cols(), nrhs, a, cols(), work.get_data(), nrhs);
        if (info > 0) {
            throw FactorizationError{"QR solve found a rank-deficient system (zero on the R diagonal at index " +
                                     std::to_string(info) + ", 1-based)"};
        }
        for (lapack_int r = 0; r < cols(); ++r) {
            std::copy_n(work.get_data() + (r * nrhs), nrhs, x.get_values() + (r * x_stride));
        }
        return;
    }

    const lapack_int n = cols();
    gko::array<T> work{factors_.get_executor(), static_cast<gko::size_type>(n) * static_cast<gko::size_type>(nrhs)};
    for (lapack_int r = 0; r < n; ++r) {
        std::copy_n(b.get_const_values() + (r * b_stride), nrhs, work.get_data() + (r * nrhs));
    }
    switch (kind_) {
    case FactorKind::LU:
        lapacke_getrs<T>(n, nrhs, a, n, pivots_.get_const_data(), work.get_data(), nrhs);
        break;
    case FactorKind::Cholesky:
        lapacke_potrs<T>('U', n, nrhs, a, n, work.get_data(), nrhs);
        break;
    case FactorKind::LDLT:
        lapacke_sytrs<T>('U', n, nrhs, a, n, pivots_.get_const_data(), work.get_data(), nrhs);
        break;
    case FactorKind::QR:
        break;
    }
    for (lapack_int r = 0; r < n; ++r) {
        std::copy_n(work.get_data() + (r * nrhs), nrhs, x.get_values() + (r * x_stride));
    }
}

template<Scalar T> void DenseFactorizationOp<T>::apply_impl(const gko::LinOp* b, gko::LinOp* x) const
{
    solve_into(*gko::as<dense_type>(b), *gko::as<dense_type>(x));
}

template<Scalar T>
void DenseFactorizationOp<T>::apply_impl(const gko::LinOp* alpha, const gko::LinOp* b, const gko::LinOp* beta,
                                         gko::LinOp* x) const
{
    auto dense_x = gko::as<dense_type>(x);
    auto solution = dense_x->clone();
    solve_into(*gko::as<dense_type>(b), *solution);
    dense_x->scale(beta);
    dense_x->add_scaled(alpha, solution);
}

} // namespace detail

/// A direct dense solver: factor a matrix once with LAPACKE, then solve many right-hand sides.
///
/// A factorization *is* a linear solver, so a ``Factorization`` behaves as the
/// operator ``A``\ :sup:`-1`: applying it solves ``A x = b``. Because it derives
/// from :cpp:`OperatorHandle` it converts to a ``gko::LinOp`` and can be handed to
/// anything that consumes one — used directly, as a solver's preconditioner, or as
/// a block.
///
/// .. code-block:: cpp
///
///     Context ctx;
///     DenseMatrix<double> a{ctx, "A", {{2.0, 1.0}, {1.0, 3.0}}};
///     Factorization<double> lu{ctx, "A_lu", a};   // factored once here
///     Vector<double> b{ctx, "b", {3.0, 5.0}};
///     Vector<double> x{ctx, "x", 2};
///     lu.solve(b, x);                              // and reused per right-hand side
///
/// The factors are an independent copy of the matrix's values, so mutating the
/// source ``DenseMatrix`` afterwards does not affect an existing factorization.
/// With :cpp:`FactorKind::QR` an overdetermined matrix (more rows than columns)
/// yields the least-squares solution rather than an exact one. Complex value types
/// are out of scope.
///
/// :tparam T: Floating-point value type.
template<Scalar T = double> class Factorization : public OperatorHandle {
public:
    /// Factors ``matrix`` with the chosen method.
    ///
    /// :param ctx: Context the factorization lives on and registers with.
    /// :param name: Name reported for this operator in timing output.
    /// :param matrix: The matrix to factor; its values are copied.
    /// :param kind: Which factorization to compute.
    /// :throws FactorizationError: if the matrix is singular (``LU``/``LDLT``) or not
    ///     positive-definite (``Cholesky``).
    Factorization(Context& ctx, std::string name, const DenseMatrix<T>& matrix, FactorKind kind = FactorKind::LU);

    /// Solves ``A x = b`` for a single right-hand side.
    ///
    /// :param b: Right-hand side; length equal to the matrix's row count.
    /// :param x: Solution; length equal to the matrix's column count.
    void solve(const Vector<T>& b, Vector<T>& x) const;

    /// The factorization kind in use.
    [[nodiscard]] FactorKind kind() const noexcept { return kind_; }

private:
    static std::shared_ptr<gko::LinOp> factor(Context& ctx, const DenseMatrix<T>& matrix, FactorKind kind);

    FactorKind kind_;
};

template<Scalar T>
Factorization<T>::Factorization(Context& ctx, std::string name, const DenseMatrix<T>& matrix, FactorKind kind) :
    OperatorHandle(ctx, std::move(name), factor(ctx, matrix, kind), scalar_type_of<T>()), kind_{kind}
{
}

template<Scalar T>
std::shared_ptr<gko::LinOp> Factorization<T>::factor(Context& ctx, const DenseMatrix<T>& matrix, FactorKind kind)
{
    const gko::dim<2> shape = matrix.shape();
    gko::array<T> values{ctx.executor()->get_master(), shape[0] * shape[1]};
    matrix.copy_values(values.get_data());
    auto op = detail::DenseFactorizationOp<T>::create(ctx.executor(), shape, values.get_const_data(), kind);
    return gko::share(std::move(op));
}

template<Scalar T> void Factorization<T>::solve(const Vector<T>& b, Vector<T>& x) const
{
    linop()->apply(b.linop(), x.linop());
}

} // namespace miscibility::instrument
