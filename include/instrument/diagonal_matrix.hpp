/**
 * @file diagonal_matrix.hpp
 * @brief A square diagonal matrix backed by a Vector, with its own matrix products and solve.
 *
 * @code{.cpp}
 * miscibility::instrument::DiagonalMatrix<double> d{2, 3, 4}; // 3x3, diagonal {2,3,4}
 * auto y = d * miscibility::instrument::Vector<double>{1, 1, 1}; // {2, 3, 4}
 * auto x = d.solve({2, 3, 4});                                   // {1, 1, 1}
 * @endcode
 *
 * @par What it is
 * A DiagonalMatrix is essentially a `Vector<T, N>` reinterpreted as the main diagonal of an
 * `N x N` matrix; all off-diagonal entries are zero. It models @ref MatrixOperator, so it can
 * serve as a block of a block matrix anywhere a dense matrix could.
 *
 * @par Why a dedicated type (no factorization)
 * A diagonal matrix is trivial to apply and to invert, so -- unlike @ref DenseMatrix, which hands
 * solving to a separate @ref LUFactorization -- DiagonalMatrix exposes solve() / solve_into()
 * directly as members: solving `D*x = b` is the elementwise quotient `x_i = b_i / d_i`. To make
 * that solve cheap and branch-free, the type caches a @ref singular() flag (true iff any diagonal
 * entry is exactly zero), maintained on construction and on every set().
 *
 * @par Products
 * The matrix-vector product `y_i <- alpha*d_i*x_i + beta*y_i` and the diagonal-times-dense
 * matrix-matrix product `C <- alpha*D*op(B) + beta*C` (which scales row `i` of `op(B)` by `d_i`)
 * are computed with the backing Vector's vectorized elementwise kernels rather than BLAS. The
 * `op`/`opB` transpose flags follow the @ref Transpose convention; for the matrix-vector product
 * `op` only affects the (square) length check, since a diagonal is its own transpose.
 *
 * @par Static vs dynamic shape
 * Like @ref Vector, the shape may be static (`DiagonalMatrix<double, 3>`, diagonal stored inline)
 * or dynamic (`DiagonalMatrix<double>`, heap-backed); the static-storage stack caveats of Vector
 * apply.
 *
 * @par Header-only
 * C++23. Builds only on @ref Vector and @ref DenseMatrix; no BLAS/LAPACK.
 */

#pragma once

#include "instrument/matrix.hpp"
#include "instrument/vector.hpp"

#include <cstddef>
#include <initializer_list>
#include <stdexcept>
#include <utility>

namespace miscibility::instrument {

/**
 * @brief A square diagonal matrix: a `Vector<T, N>` of diagonal entries viewed as a matrix.
 * @tparam T Element type (an IEEE floating-point @ref Scalar).
 * @tparam N Diagonal length / matrix order, or @c dynamic (the default) for a runtime shape.
 *
 * Logical shape is `N x N`; only the `N` diagonal entries are stored. Models @ref MatrixOperator.
 * See the file overview for the products, the member solve(), and the cached @ref singular() flag.
 */
template<Scalar T, std::size_t N = dynamic> class DiagonalMatrix {
public:
    using value_type = T;          ///< Element type.
    using size_type = std::size_t; ///< Dimension / index type.

    /// @brief True iff this is the runtime-shape (heap-backed) specialization.
    static constexpr bool is_dynamic = (N == dynamic);

    // -- construction ---------------------------------------------------------

    /// @brief Default: a static `N x N` zero diagonal, or an empty 0x0 dynamic matrix.
    DiagonalMatrix() { recompute_singular(); }

    /// @brief Construct a runtime-shape `n x n` zero diagonal. Dynamic shape only. @param n Matrix order.
    explicit DiagonalMatrix(size_type n)
        requires is_dynamic
        : diag_(n)
    {
        recompute_singular();
    }

    /**
     * @brief Construct from the diagonal entries given as a braced list.
     * @param diag The diagonal entries; the order is `diag.size()`. For a static shape the list
     *             length must equal @c N.
     * @throws std::invalid_argument if a static shape's list length is not @c N.
     */
    DiagonalMatrix(std::initializer_list<T> diag) : diag_(diag) { recompute_singular(); }

    /// @brief Adopt an existing Vector as the diagonal. @param diag The diagonal entries (moved in).
    explicit DiagonalMatrix(Vector<T, N> diag) : diag_(std::move(diag)) { recompute_singular(); }

    // -- dimensions / access --------------------------------------------------

    [[nodiscard]] size_type rows() const noexcept { return diag_.size(); }    ///< Row count (the order `n`).
    [[nodiscard]] size_type columns() const noexcept { return diag_.size(); } ///< Column count (the order `n`).
    /// @brief Diagonal length `n` (the number of stored entries, **not** `n*n`). @return The order `n`.
    [[nodiscard]] size_type size() const noexcept { return diag_.size(); }
    [[nodiscard]] bool empty() const noexcept { return diag_.size() == 0; } ///< True iff the order is 0.

    /// @brief Unchecked read of diagonal entry @p i. @param i Index in `[0, size())`. @return Const reference to it.
    [[nodiscard]] const T& operator[](size_type i) const noexcept { return diag_[i]; }

    /// @brief Bounds-checked read of diagonal entry @p i. @param i Index. @return Const reference to it.
    /// @throws std::out_of_range if `i >= size()`.
    [[nodiscard]] const T& at(size_type i) const { return diag_.at(i); }

    /// @brief The backing diagonal vector (read-only seam). @return Const reference to the diagonal.
    [[nodiscard]] const Vector<T, N>& diagonal() const noexcept { return diag_; }
    /// @brief Const pointer to the contiguous diagonal buffer. @return The diagonal data pointer.
    [[nodiscard]] const T* data() const noexcept { return diag_.data(); }

    // -- mutation -------------------------------------------------------------

    /**
     * @brief Write diagonal entry @p i, keeping the cached @ref singular() flag exact.
     * @param i     Index of the entry to write.
     * @param value New value for entry @p i.
     * @throws std::out_of_range if `i >= size()`.
     *
     * The only mutating operation -- no mutable element reference is ever handed out -- so the
     * zero-count behind singular() can be updated in O(1) on every change.
     */
    void set(size_type i, T value)
    {
        if (i >= diag_.size()) {
            throw std::out_of_range{"miscibility::instrument::DiagonalMatrix::set"};
        }
        const T old = diag_[i];
        if (old == T(0) && value != T(0)) {
            --zero_count_;
        }
        else if (old != T(0) && value == T(0)) {
            ++zero_count_;
        }
        diag_[i] = value;
        singular_ = (zero_count_ > 0);
    }

    // -- singularity ----------------------------------------------------------

    /// @brief True iff any diagonal entry is exactly zero (so the matrix is not invertible).
    /// @return The cached singular flag.
    [[nodiscard]] bool singular() const noexcept { return singular_; }

    // -- matrix-vector product ------------------------------------------------

    /**
     * @brief Matrix-vector product into a pre-allocated vector: `y_i <- alpha*d_i*x_i + beta*y_i`.
     * @param x     Right operand, length `size()`.
     * @param y     Destination, written in place; must already be length `size()`.
     * @param alpha Scale applied to the product (default 1).
     * @param beta  Scale applied to the existing @p y before accumulation (default 0).
     * @param op    Whether to use `D` (@c None) or `D^T` (@c Transposed); has no effect on the
     *              result (a diagonal is its own transpose), only on the length check.
     * @throws std::invalid_argument if @p x or @p y has the wrong length.
     */
    void multiply_into(const Vector<T>& x, Vector<T>& y, T alpha = T(1), T beta = T(0),
                       Transpose op = Transpose::None) const
    {
        (void)op; // a diagonal is its own transpose; op only affects the (square) length check
        const size_type n = diag_.size();
        if (x.size() != n || y.size() != n) {
            throw std::invalid_argument{"miscibility::instrument::DiagonalMatrix matrix-vector size mismatch"};
        }
        // TODO: try and remove this allocation
        Vector<T> p(n);
        p.copy(diag_);
        p.elementwise_product(x); // p_i = d_i * x_i
        y.scale(beta);
        y.add_scaled(alpha, p); // y_i <- alpha*d_i*x_i + beta*y_i
    }

    /**
     * @brief Matrix-vector product `D*x`, returning a fresh vector.
     * @param x  Right operand, length `size()`.
     * @param op Whether to use `D` (@c None) or `D^T` (@c Transposed) -- same result either way.
     * @return The product, a new `Vector<T>` of length `size()`.
     * @throws std::invalid_argument if @p x has the wrong length.
     */
    [[nodiscard]] Vector<T> multiply(const Vector<T>& x, Transpose op = Transpose::None) const
    {
        Vector<T> y(diag_.size());
        multiply_into(x, y, T(1), T(0), op);
        return y;
    }

    /// @brief `D*x` as a fresh vector. @param x Operand of length `size()`. @return The product.
    /// @throws std::invalid_argument on a length mismatch.
    [[nodiscard]] Vector<T> operator*(const Vector<T>& x) const { return multiply(x); }

    // -- matrix-matrix product (diagonal-times-dense row scaling) -------------

    /**
     * @brief Diagonal-times-dense product into a pre-allocated matrix: `C <- alpha*D*op(B) + beta*C`.
     * @param b     Right operand. `op(B)` must have `size()` rows.
     * @param c     Destination, written in place; must already be `size()` x `(cols of op(B))`.
     * @param alpha Scale applied to the product (default 1).
     * @param beta  Scale applied to the existing @p c before accumulation (default 0).
     * @param opB   Whether to use `B` (@c None) or `B^T` (@c Transposed).
     * @throws std::invalid_argument if the shapes are incompatible.
     *
     * Equivalent to scaling row `i` of `op(B)` by the diagonal entry `d_i`.
     */
    void multiply_into(const DenseMatrix<T>& b, DenseMatrix<T>& c, T alpha = T(1), T beta = T(0),
                       Transpose opB = Transpose::None) const
    {
        const size_type n = diag_.size();
        const size_type brows = (opB == Transpose::None) ? b.rows() : b.columns();
        const size_type bcols = (opB == Transpose::None) ? b.columns() : b.rows();
        if (brows != n || c.rows() != n || c.columns() != bcols) {
            throw std::invalid_argument{"miscibility::instrument::DiagonalMatrix matrix-matrix size mismatch"};
        }
        // TODO: try and remove allocations
        // Each result column j is alpha*(d (.) op(B)_{:,j}) + beta*C_{:,j}.
        Vector<T> col(n);
        Vector<T> ccol(n);
        for (size_type j = 0; j < bcols; ++j) {
            for (size_type i = 0; i < n; ++i) {
                col[i] = (opB == Transpose::None) ? b(i, j) : b(j, i);
                ccol[i] = c(i, j);
            }
            col.elementwise_product(diag_); // col_i = d_i * op(B)_{i,j}
            ccol.scale(beta);
            ccol.add_scaled(alpha, col);
            for (size_type i = 0; i < n; ++i) {
                c(i, j) = ccol[i];
            }
        }
    }

    /**
     * @brief Diagonal-times-dense product `D*op(B)`, returning a fresh matrix.
     * @param b   Right operand. `op(B)` must have `size()` rows.
     * @param opB Whether to use `B` (@c None) or `B^T` (@c Transposed).
     * @return The product, a new `DenseMatrix<T>` of shape `size()` x `(cols of op(B))`.
     * @throws std::invalid_argument if the shapes are incompatible.
     */
    [[nodiscard]] DenseMatrix<T> multiply(const DenseMatrix<T>& b, Transpose opB = Transpose::None) const
    {
        const size_type bcols = (opB == Transpose::None) ? b.columns() : b.rows();
        DenseMatrix<T> c(diag_.size(), bcols);
        multiply_into(b, c, T(1), T(0), opB);
        return c;
    }

    /// @brief `D*B` as a fresh matrix. @param b Right operand with `size()` rows. @return The product.
    /// @throws std::invalid_argument on a shape mismatch.
    [[nodiscard]] DenseMatrix<T> operator*(const DenseMatrix<T>& b) const { return multiply(b); }

    // -- solve ----------------------------------------------------------------

    /**
     * @brief Solve `D*x = b` into a pre-allocated vector: `x_i = b_i / d_i`.
     * @param b Right-hand side, length `size()`. Not modified.
     * @param x Destination, written in place; must already be length `size()`.
     * @throws std::invalid_argument if @p b or @p x has the wrong length.
     * @throws std::runtime_error if the matrix is singular() (some diagonal entry is zero).
     *
     * The singular() flag is checked once up front; the division itself is then an unguarded,
     * vectorized elementwise quotient.
     */
    void solve_into(const Vector<T>& b, Vector<T>& x) const
    {
        const size_type n = diag_.size();
        if (b.size() != n || x.size() != n) {
            throw std::invalid_argument{"miscibility::instrument::DiagonalMatrix solve size mismatch"};
        }
        if (singular_) {
            throw std::runtime_error{"miscibility::instrument::DiagonalMatrix: singular matrix"};
        }
        x.copy(b);
        x.elementwise_quotient(diag_); // x_i = b_i / d_i
    }

    /**
     * @brief Solve `D*x = b`, returning a fresh solution vector.
     * @param b Right-hand side, length `size()`.
     * @return The solution `x`, a new `Vector<T>` of length `size()`.
     * @throws std::invalid_argument if @p b has the wrong length.
     * @throws std::runtime_error if the matrix is singular().
     */
    [[nodiscard]] Vector<T> solve(const Vector<T>& b) const
    {
        Vector<T> x(diag_.size());
        solve_into(b, x);
        return x;
    }

private:
    /// @brief Recount zero diagonal entries and refresh the singular flag (called on construction).
    void recompute_singular() noexcept
    {
        zero_count_ = 0;
        for (size_type i = 0; i < diag_.size(); ++i) {
            if (diag_[i] == T(0)) {
                ++zero_count_;
            }
        }
        singular_ = (zero_count_ > 0);
    }

    Vector<T, N> diag_{};      ///< The diagonal entries.
    size_type zero_count_{0};  ///< Number of zero diagonal entries (drives @ref singular()).
    bool singular_{false};     ///< Cached `zero_count_ > 0`.
};

} // namespace miscibility::instrument
