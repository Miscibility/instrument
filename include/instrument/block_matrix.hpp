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
    [[nodiscard]] Vector<T>& block(size_type i) { return blocks_.at(i); }

    /// @brief Access sub-vector @p i (checked, const). @param i Block index. @return The sub-vector.
    /// @throws std::out_of_range if @p i is out of range.
    [[nodiscard]] const Vector<T>& block(size_type i) const { return blocks_.at(i); }

    /// @brief Total length summed across every block. @return The total element count.
    [[nodiscard]] size_type size() const
    {
        size_type total = 0;
        for (const auto& b : blocks_) {
            total += b.size();
        }
        return total;
    }

    /// @brief Append a sub-vector. @param v The sub-vector to append (moved in).
    void push_block(Vector<T> v) { blocks_.push_back(std::move(v)); }

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

    [[nodiscard]] size_type rows() const override { return matrix_.rows(); }
    [[nodiscard]] size_type columns() const override { return matrix_.columns(); }
    void multiply_into(const Vector<T>& x, Vector<T>& y, T alpha, T beta, Transpose op) const override
    {
        matrix_.multiply_into(x, y, alpha, beta, op);
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
        grid_.at(flat_index(i, j)) = std::make_unique<detail::AnyMatrixModel<T, M>>(std::move(matrix));
    }

    /// @brief Access the type-erased block at `(i, j)`. @param i Block-row. @param j Block-col.
    /// @return The type-erased block. @throws std::out_of_range if out of grid; std::logic_error if unset.
    [[nodiscard]] detail::AnyMatrixOperator<T>& block(size_type i, size_type j)
    {
        return *cell(i, j);
    }

    /// @brief Access the type-erased block at `(i, j)` (const). @param i Block-row. @param j Block-col.
    /// @return The type-erased block. @throws std::out_of_range if out of grid; std::logic_error if unset.
    [[nodiscard]] const detail::AnyMatrixOperator<T>& block(size_type i, size_type j) const
    {
        return *cell(i, j);
    }

    /// @brief Recover the underlying concrete object at `(i, j)`.
    /// @tparam M The expected concrete type. @param i Block-row. @param j Block-col. @return The concrete block.
    /// @throws std::out_of_range if out of grid; std::bad_cast if the block is not an @p M.
    template<class M> [[nodiscard]] M& block_as(size_type i, size_type j)
    {
        return dynamic_cast<detail::AnyMatrixModel<T, M>&>(block(i, j)).underlying();
    }

    /// @brief Recover the underlying concrete object at `(i, j)` (const).
    /// @tparam M The expected concrete type. @param i Block-row. @param j Block-col. @return The concrete block.
    /// @throws std::out_of_range if out of grid; std::bad_cast if the block is not an @p M.
    template<class M> [[nodiscard]] const M& block_as(size_type i, size_type j) const
    {
        return dynamic_cast<const detail::AnyMatrixModel<T, M>&>(block(i, j)).underlying();
    }

    /// @brief Total row count = Σ of block-row heights. @return The row count.
    [[nodiscard]] size_type rows() const
    {
        size_type total = 0;
        for (size_type i = 0; i < block_rows_; ++i) {
            total += block(i, 0).rows();
        }
        return total;
    }

    /// @brief Total column count = Σ of block-column widths. @return The column count.
    [[nodiscard]] size_type columns() const
    {
        size_type total = 0;
        for (size_type j = 0; j < block_cols_; ++j) {
            total += block(0, j).columns();
        }
        return total;
    }

    /// @brief Block matrix-vector product into a destination: `y <- alpha*op(A)*x + beta*y`, block-wise.
    /// @param x Operand block vector. @param y Destination block vector. @param alpha Product scale.
    /// @param beta Destination scale. @param op None or Transposed.
    /// @throws std::invalid_argument on structural/length mismatch; std::logic_error if a cell is unset.
    void multiply_into(const BlockVector<T>& x, BlockVector<T>& y, T alpha = T(1), T beta = T(0),
                       Transpose op = Transpose::None) const
    {
        // TODO: make this a pre-processed step done once? Then the check is very simple during multiply.
        // Gather each block-row's height and block-column's width, validating that the grid is
        // fully populated and that the shared dimensions are consistent across the grid.
        std::vector<size_type> heights(block_rows_);
        std::vector<size_type> widths(block_cols_);
        for (size_type i = 0; i < block_rows_; ++i) {
            for (size_type j = 0; j < block_cols_; ++j) {
                const detail::AnyMatrixOperator<T>& a = block(i, j); // throws logic_error if unset
                if (j == 0) {
                    heights[i] = a.rows();
                } else if (a.rows() != heights[i]) {
                    throw std::invalid_argument{
                        "miscibility::instrument::BlockMatrix inconsistent block-row heights"};
                }
                if (i == 0) {
                    widths[j] = a.columns();
                } else if (a.columns() != widths[j]) {
                    throw std::invalid_argument{
                        "miscibility::instrument::BlockMatrix inconsistent block-column widths"};
                }
            }
        }

        // For None the operand spans the block-columns and the result the block-rows; Transposed
        // swaps the two. `src`/`dst` name the operand/result sizes for the active orientation.
        const std::vector<size_type>& src = (op == Transpose::None) ? widths : heights;
        const std::vector<size_type>& dst = (op == Transpose::None) ? heights : widths;
        if (x.block_count() != src.size() || y.block_count() != dst.size()) {
            throw std::invalid_argument{"miscibility::instrument::BlockMatrix operand block-count mismatch"};
        }
        for (size_type k = 0; k < src.size(); ++k) {
            if (x.block(k).size() != src[k]) {
                throw std::invalid_argument{"miscibility::instrument::BlockMatrix operand sub-vector size mismatch"};
            }
        }
        for (size_type k = 0; k < dst.size(); ++k) {
            if (y.block(k).size() != dst[k]) {
                throw std::invalid_argument{"miscibility::instrument::BlockMatrix result sub-vector size mismatch"};
            }
        }

        // Orchestrate: accumulate every contributing block into each destination sub-vector. The
        // first contribution applies the caller's beta; the rest use beta = 1 to sum in place.
        for (size_type d = 0; d < dst.size(); ++d) {
            for (size_type s = 0; s < src.size(); ++s) {
                const size_type i = (op == Transpose::None) ? d : s;
                const size_type j = (op == Transpose::None) ? s : d;
                const T beta_s = (s == 0) ? beta : T(1);
                block(i, j).multiply_into(x.block(s), y.block(d), alpha, beta_s, op);
            }
        }
    }

    /// @brief Block matrix-vector product, allocating the result. @param x Operand. @param op None/Transposed.
    /// @return The product `op(A)*x` as a fresh block vector.
    [[nodiscard]] BlockVector<T> multiply(const BlockVector<T>& x, Transpose op = Transpose::None) const
    {
        BlockVector<T> y;
        if (op == Transpose::None) {
            for (size_type i = 0; i < block_rows_; ++i) {
                y.push_block(Vector<T>(block(i, 0).rows()));
            }
        } else {
            for (size_type j = 0; j < block_cols_; ++j) {
                y.push_block(Vector<T>(block(0, j).columns()));
            }
        }
        multiply_into(x, y, T(1), T(0), op);
        return y;
    }

    /// @brief Block matrix-vector product `A*x`. @param x Operand. @return The product as a fresh block vector.
    [[nodiscard]] BlockVector<T> operator*(const BlockVector<T>& x) const { return multiply(x); }

private:
    /// @brief Row-major flat index into @ref grid_ for cell `(i, j)`, bounds-checked against the grid.
    /// @throws std::out_of_range if `(i, j)` is outside the grid.
    [[nodiscard]] size_type flat_index(size_type i, size_type j) const
    {
        if (i >= block_rows_ || j >= block_cols_) {
            throw std::out_of_range{"miscibility::instrument::BlockMatrix block index out of range"};
        }
        return (i * block_cols_) + j;
    }

    /// @brief The (set) block at `(i, j)`. @throws std::out_of_range if out of grid; std::logic_error if unset.
    [[nodiscard]] const std::unique_ptr<detail::AnyMatrixOperator<T>>& cell(size_type i, size_type j) const
    {
        const auto& ptr = grid_.at(flat_index(i, j));
        if (!ptr) {
            throw std::logic_error{"miscibility::instrument::BlockMatrix block (i, j) was never set"};
        }
        return ptr;
    }

    size_type block_rows_{0};                                       ///< Number of block-rows.
    size_type block_cols_{0};                                       ///< Number of block-columns.
    std::vector<std::unique_ptr<detail::AnyMatrixOperator<T>>> grid_; ///< Row-major grid of blocks.
};

} // namespace miscibility::instrument
