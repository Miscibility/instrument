/**
 * @file block_matrix.hpp
 * @brief A grid of type-erased sub-matrices (BlockMatrix) and a sequence of sub-vectors
 *        (BlockVector), supporting the block matrix-vector product `y = A*x`.
 *
 * SKELETON: every BlockVector/BlockMatrix member below currently throws
 * `std::runtime_error{"not implemented"}` -- this header pins down the interface so the
 * unit tests compile and (uniformly) fail. tdd-3-implement fills in the bodies.
 */

#pragma once

#include "instrument/matrix.hpp"
#include "instrument/vector.hpp"

#include <cstddef>
#include <initializer_list>
#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>

namespace miscibility::instrument {

/**
 * @brief A sequence of sub-vectors -- the vector operand/result of a @ref BlockMatrix product.
 * @tparam T Element type (an IEEE floating-point @ref Scalar).
 */
template<class T> class BlockVector {
public:
    using value_type = T;          ///< Element type.
    using size_type = std::size_t; ///< Index / dimension type.

    /// @brief Construct an empty block vector (zero blocks).
    BlockVector() = default;

    /// @brief Adopt a sequence of sub-vectors. @param blocks The block sub-vectors (moved in).
    explicit BlockVector(std::vector<Vector<T>> blocks) : blocks_(std::move(blocks)) {}

    /// @brief Construct from a braced list of sub-vectors. @param blocks The block sub-vectors.
    BlockVector(std::initializer_list<Vector<T>> blocks) : blocks_(blocks.begin(), blocks.end()) {}

    /// @brief Number of sub-vector blocks. @return The block count.
    [[nodiscard]] size_type block_count() const noexcept { return blocks_.size(); }

    /// @brief Access sub-vector @p i (checked). @param i Block index. @return The sub-vector.
    /// @throws std::out_of_range if @p i is out of range.
    [[nodiscard]] Vector<T>& block(size_type i)
    {
        (void)i;
        throw std::runtime_error{"not implemented"};
    }

    /// @brief Access sub-vector @p i (checked, const). @param i Block index. @return The sub-vector.
    /// @throws std::out_of_range if @p i is out of range.
    [[nodiscard]] const Vector<T>& block(size_type i) const
    {
        (void)i;
        throw std::runtime_error{"not implemented"};
    }

    /// @brief Total length summed across every block. @return The total element count.
    [[nodiscard]] size_type size() const { throw std::runtime_error{"not implemented"}; }

    /// @brief Append a sub-vector. @param v The sub-vector to append (moved in).
    void push_block(Vector<T> v)
    {
        (void)v;
        throw std::runtime_error{"not implemented"};
    }

private:
    std::vector<Vector<T>> blocks_; ///< The owned sub-vectors.
};

namespace detail {

/**
 * @internal
 * @brief Abstract base for a type-erased block: the minimal @ref MatrixOperator surface.
 * @tparam T Element type.
 */
template<class T> class AnyMatrixOperator {
public:
    using value_type = T;          ///< Element type.
    using size_type = std::size_t; ///< Index / dimension type.

    AnyMatrixOperator() = default;
    AnyMatrixOperator(const AnyMatrixOperator&) = default;
    AnyMatrixOperator(AnyMatrixOperator&&) = default;
    AnyMatrixOperator& operator=(const AnyMatrixOperator&) = default;
    AnyMatrixOperator& operator=(AnyMatrixOperator&&) = default;
    virtual ~AnyMatrixOperator() = default;

    [[nodiscard]] virtual size_type rows() const = 0;    ///< Block row count.
    [[nodiscard]] virtual size_type columns() const = 0; ///< Block column count.
    /// @brief Forwarded gemv: `y <- alpha*op(A)*x + beta*y`.
    virtual void multiply_into(const Vector<T>& x, Vector<T>& y, T alpha, T beta, Transpose op) const = 0;
};

/**
 * @internal
 * @brief Concrete model wrapping any @p M that satisfies @ref MatrixOperator, forwarding the seam.
 * @tparam T Element type.
 * @tparam M The wrapped concrete matrix type.
 */
template<class T, class M>
    requires MatrixOperator<M, T>
class AnyMatrixModel final : public AnyMatrixOperator<T> {
public:
    using size_type = std::size_t; ///< Index / dimension type.

    /// @brief Wrap (move in) a concrete matrix. @param matrix The matrix to type-erase.
    explicit AnyMatrixModel(M matrix) : matrix_(std::move(matrix)) {}

    [[nodiscard]] size_type rows() const override { throw std::runtime_error{"not implemented"}; }
    [[nodiscard]] size_type columns() const override { throw std::runtime_error{"not implemented"}; }
    void multiply_into(const Vector<T>& x, Vector<T>& y, T alpha, T beta, Transpose op) const override
    {
        (void)x;
        (void)y;
        (void)alpha;
        (void)beta;
        (void)op;
        throw std::runtime_error{"not implemented"};
    }

    /// @brief Recover the wrapped concrete object. @return Reference to the stored matrix.
    [[nodiscard]] M& underlying() noexcept { return matrix_; }
    /// @brief Recover the wrapped concrete object (const). @return Reference to the stored matrix.
    [[nodiscard]] const M& underlying() const noexcept { return matrix_; }

private:
    M matrix_; ///< The wrapped concrete matrix.
};

} // namespace detail

/**
 * @brief A `block_rows x block_cols` grid of type-erased sub-matrices.
 * @tparam T Element type (an IEEE floating-point @ref Scalar).
 *
 * Each cell holds any type satisfying `MatrixOperator<M, T>`, type-erased behind a
 * `std::unique_ptr<detail::AnyMatrixOperator<T>>`. Cells start empty and must be set (use
 * @ref ZeroMatrix for structural zeros) before a product is taken.
 */
template<class T> class BlockMatrix {
public:
    using value_type = T;          ///< Element type.
    using size_type = std::size_t; ///< Index / dimension type.

    /// @brief Construct an (empty) grid of the given block dimensions.
    /// @param block_rows Number of block-rows. @param block_cols Number of block-columns.
    BlockMatrix(size_type block_rows, size_type block_cols)
        : block_rows_(block_rows), block_cols_(block_cols), grid_(block_rows * block_cols)
    {
    }

    /// @brief Number of block-rows. @return The block-row count.
    [[nodiscard]] size_type block_rows() const noexcept { return block_rows_; }
    /// @brief Number of block-columns. @return The block-column count.
    [[nodiscard]] size_type block_columns() const noexcept { return block_cols_; }

    /// @brief Install (move) @p matrix into grid cell `(i, j)`, type-erased.
    /// @tparam M The concrete matrix type. @param i Block-row. @param j Block-col. @param matrix The block.
    /// @throws std::out_of_range if `(i, j)` is outside the grid.
    template<class M>
        requires MatrixOperator<M, T>
    void set_block(size_type i, size_type j, M matrix)
    {
        (void)i;
        (void)j;
        (void)matrix;
        throw std::runtime_error{"not implemented"};
    }

    /// @brief Access the type-erased block at `(i, j)`. @param i Block-row. @param j Block-col.
    /// @return The type-erased block. @throws std::out_of_range if out of grid; std::logic_error if unset.
    [[nodiscard]] detail::AnyMatrixOperator<T>& block(size_type i, size_type j)
    {
        (void)i;
        (void)j;
        throw std::runtime_error{"not implemented"};
    }

    /// @brief Access the type-erased block at `(i, j)` (const). @param i Block-row. @param j Block-col.
    /// @return The type-erased block. @throws std::out_of_range if out of grid; std::logic_error if unset.
    [[nodiscard]] const detail::AnyMatrixOperator<T>& block(size_type i, size_type j) const
    {
        (void)i;
        (void)j;
        throw std::runtime_error{"not implemented"};
    }

    /// @brief Recover the underlying concrete object at `(i, j)`.
    /// @tparam M The expected concrete type. @param i Block-row. @param j Block-col. @return The concrete block.
    /// @throws std::out_of_range if out of grid; std::bad_cast if the block is not an @p M.
    template<class M> [[nodiscard]] M& block_as(size_type i, size_type j)
    {
        (void)i;
        (void)j;
        throw std::runtime_error{"not implemented"};
    }

    /// @brief Recover the underlying concrete object at `(i, j)` (const).
    /// @tparam M The expected concrete type. @param i Block-row. @param j Block-col. @return The concrete block.
    /// @throws std::out_of_range if out of grid; std::bad_cast if the block is not an @p M.
    template<class M> [[nodiscard]] const M& block_as(size_type i, size_type j) const
    {
        (void)i;
        (void)j;
        throw std::runtime_error{"not implemented"};
    }

    /// @brief Total row count = Σ of block-row heights. @return The row count.
    [[nodiscard]] size_type rows() const { throw std::runtime_error{"not implemented"}; }
    /// @brief Total column count = Σ of block-column widths. @return The column count.
    [[nodiscard]] size_type columns() const { throw std::runtime_error{"not implemented"}; }

    /// @brief Block matrix-vector product into a destination: `y <- alpha*op(A)*x + beta*y`, block-wise.
    /// @param x Operand block vector. @param y Destination block vector. @param alpha Product scale.
    /// @param beta Destination scale. @param op None or Transposed.
    /// @throws std::invalid_argument on structural/length mismatch; std::logic_error if a cell is unset.
    void multiply_into(const BlockVector<T>& x, BlockVector<T>& y, T alpha = T(1), T beta = T(0),
                       Transpose op = Transpose::None) const
    {
        (void)x;
        (void)y;
        (void)alpha;
        (void)beta;
        (void)op;
        throw std::runtime_error{"not implemented"};
    }

    /// @brief Block matrix-vector product, allocating the result. @param x Operand. @param op None/Transposed.
    /// @return The product `op(A)*x` as a fresh block vector.
    [[nodiscard]] BlockVector<T> multiply(const BlockVector<T>& x, Transpose op = Transpose::None) const
    {
        (void)x;
        (void)op;
        throw std::runtime_error{"not implemented"};
    }

    /// @brief Block matrix-vector product `A*x`. @param x Operand. @return The product as a fresh block vector.
    [[nodiscard]] BlockVector<T> operator*(const BlockVector<T>& x) const
    {
        (void)x;
        throw std::runtime_error{"not implemented"};
    }

private:
    size_type block_rows_{0};                                       ///< Number of block-rows.
    size_type block_cols_{0};                                       ///< Number of block-columns.
    std::vector<std::unique_ptr<detail::AnyMatrixOperator<T>>> grid_; ///< Row-major grid of blocks.
};

} // namespace miscibility::instrument
