/**
 * @file zero_matrix.hpp
 * @brief A sizes-only matrix of all zeros that models the @ref MatrixOperator seam.
 *
 * @code{.cpp}
 * miscibility::instrument::ZeroMatrix<double> z(3, 4); // a 3x4 all-zeros matrix
 * @endcode
 *
 * @par What it is
 * A ZeroMatrix carries *no element storage* -- only its `rows()` x `columns()` shape. Its
 * matrix-vector product is the action of an all-zeros matrix, which is why it satisfies
 * @ref MatrixOperator without holding any data. It exists mainly to fill the empty cells
 * of a block matrix, where an explicit zero block is needed to pin down the block shape.
 *
 * @par Behavior
 * Because `op(Z)*x` is identically zero, the gemv form `y <- alpha*op(Z)*x + beta*y`
 * collapses to `y <- beta*y` -- the @p alpha and @p x operands never affect the result
 * (they are still range-checked). With the default `beta == 0` the product simply
 * zero-fills its destination.
 *
 * @par Header-only
 * C++23. Trivially copyable.
 */

#pragma once

#include "instrument/matrix.hpp"
#include "instrument/vector.hpp"

#include <cstddef>
#include <stdexcept>

namespace miscibility::instrument {

/**
 * @brief A matrix of all zeros, represented by its shape alone (no element storage).
 * @tparam T Element type (an IEEE floating-point @ref Scalar).
 *
 * Models @ref MatrixOperator, so it can stand in anywhere a matrix-vector operand is
 * expected -- in particular as a zero block of a block matrix. The matrix-vector product
 * applies `y <- beta*y` (see the file overview); `op` does not change the result values
 * but does select which dimension each operand is checked against.
 */
template<Scalar T> class ZeroMatrix {
public:
    using value_type = T;          ///< Element type.
    using size_type = std::size_t; ///< Dimension / index type.

    /// @brief Construct an all-zeros matrix of the given shape. @param rows Row count. @param cols Column count.
    ZeroMatrix(size_type rows, size_type cols) : rows_(rows), cols_(cols) {}

    [[nodiscard]] size_type rows() const noexcept { return rows_; }    ///< Logical row count.
    [[nodiscard]] size_type columns() const noexcept { return cols_; } ///< Logical column count.
    /// @brief Logical element count, `rows() * columns()`. @return The element count.
    [[nodiscard]] size_type size() const noexcept { return rows_ * cols_; }
    [[nodiscard]] bool empty() const noexcept { return size() == 0; } ///< True iff `size() == 0`.

    /**
     * @brief Matrix-vector product into a pre-allocated vector: `y <- alpha*op(Z)*x + beta*y`.
     * @param x     Right operand. Length `columns()` for @c None, `rows()` for @c Transposed. Only its
     *              length is used -- the product term `alpha*op(Z)*x` is identically zero.
     * @param y     Destination, written in place. Length `rows()` for @c None, `columns()` for
     *              @c Transposed; must already be sized.
     * @param alpha Scale of the (zero) product; has no effect on the result (default 1).
     * @param beta  Scale applied to the existing @p y (default 0, which zero-fills @p y).
     * @param op    Whether to use `Z` (@c None) or `Z^T` (@c Transposed); selects the operand lengths.
     * @throws std::invalid_argument if @p x or @p y has the wrong length for @p op.
     */
    void multiply_into(const Vector<T>& x, Vector<T>& y, T alpha = T(1), T beta = T(0),
                       Transpose op = Transpose::None) const
    {
        (void)alpha; // the alpha*Z*x term is identically zero, so alpha is irrelevant
        const size_type xlen = (op == Transpose::None) ? cols_ : rows_;
        const size_type ylen = (op == Transpose::None) ? rows_ : cols_;
        if (x.size() != xlen || y.size() != ylen) {
            throw std::invalid_argument{"miscibility::instrument::ZeroMatrix matrix-vector size mismatch"};
        }
        y.scale(beta); // y <- beta*y
    }

    /**
     * @brief Matrix-vector product `op(Z)*x`, returning a fresh all-zero vector.
     * @param x  Right operand (see multiply_into() for the length rule).
     * @param op Whether to use `Z` (@c None) or `Z^T` (@c Transposed).
     * @return A new all-zero `Vector<T>` of length `rows()` (or `columns()` if @c Transposed).
     * @throws std::invalid_argument if @p x has the wrong length.
     */
    [[nodiscard]] Vector<T> multiply(const Vector<T>& x, Transpose op = Transpose::None) const
    {
        Vector<T> y((op == Transpose::None) ? rows_ : cols_);
        // FIXME: Can this be removed?
        multiply_into(x, y, T(1), T(0), op);
        return y;
    }

    /// @brief `Z*x` as a fresh all-zero vector. @param x Operand of length `columns()`. @return The (zero) product.
    /// @throws std::invalid_argument on a length mismatch.
    [[nodiscard]] Vector<T> operator*(const Vector<T>& x) const { return multiply(x); }

private:
    size_type rows_{0}; ///< Logical row count.
    size_type cols_{0}; ///< Logical column count.
};

} // namespace miscibility::instrument
