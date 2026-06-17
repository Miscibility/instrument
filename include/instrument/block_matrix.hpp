/**
 * @file block_matrix.hpp
 * @brief A grid of type-erased sub-matrices (@ref miscibility::instrument::BlockMatrix) and a
 *        sequence of sub-vectors (@ref miscibility::instrument::BlockVector), supporting the block
 *        matrix-vector product `y = A*x`.
 *
 * @code{.cpp}
 * using namespace miscibility::instrument;
 *
 * BlockMatrix<double> a(2, 2);                                  // a 2x2 grid of blocks
 * a.set_block(0, 0, DenseMatrix<double>{{1, 2}, {3, 4}});       // any MatrixOperator type ...
 * a.set_block(0, 1, ZeroMatrix<double>(2, 2));                  // ... ZeroMatrix fills the gaps
 * a.set_block(1, 0, DiagonalMatrix<double>{5, 6});             // ... and the blocks may differ
 * a.set_block(1, 1, DenseMatrix<double>{{7, 8}, {9, 10}});
 *
 * BlockVector<double> x{Vector<double>{1, 2}, Vector<double>{3, 4}};
 * BlockVector<double> y = a * x;                                // y.block(0), y.block(1)
 * @endcode
 *
 * @par What it is
 * A @ref BlockMatrix is a `block_rows x block_cols` grid of sub-matrices and a @ref BlockVector is
 * a sequence of sub-vectors; together they support a single operation -- the block matrix-vector
 * product `y = A*x` (and its gemv variants). Each block may be a *different* concrete matrix type
 * (a @ref DenseMatrix, @ref DiagonalMatrix, a CSR matrix, @ref ZeroMatrix, ...), so the blocks are
 * stored type-erased behind the @ref MatrixOperator seam.
 *
 * @par Host-side orchestration
 * The block product is pure host-side orchestration: it loops over the block grid and forwards each
 * cell to that block's own `multiply_into`, accumulating partial results into the destination
 * sub-vectors. The real compute therefore lives inside each block; the virtual dispatch is only the
 * outer loop. This keeps @ref BlockMatrix non-templated on its block types and is GPU-portable -- a
 * future GPU block type slots in without touching @ref BlockMatrix.
 *
 * @par Scope
 * Only @ref BlockMatrix x @ref BlockVector matrix-vector products are supported: there is no block
 * matrix-matrix product, and no mixing with plain @ref Vector / @ref DenseMatrix operands.
 *
 * @par Header-only
 * C++23. Builds on @ref Vector and the @ref MatrixOperator seam from @ref matrix.hpp.
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
 * @brief A sequence of sub-vectors -- the vector operand and result of a @ref BlockMatrix product.
 * @tparam T Element type (an IEEE floating-point @ref Scalar).
 *
 * A BlockVector owns its blocks (each a @ref Vector). It is the partner type to @ref BlockMatrix:
 * when computing `y = A*x` the operand @p x is split into one sub-vector per block-column and the
 * result @p y holds one sub-vector per block-row (the roles flip under @ref Transpose::Transposed).
 * Blocks may have *different* lengths -- the lengths must match the corresponding block dimensions
 * of the @ref BlockMatrix, which validates them at product time.
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

    // -- BLAS-1 operations (block-wise orchestration) -------------------------

    T dot(const BlockVector<T>& /*other*/) const { throw std::runtime_error{"not implemented"}; }

    // noexcept stubs cannot throw (that would std::terminate the test binary), so they return a
    // wrong sentinel / do nothing; their tests fail via assertions until tdd-3 fills them in.
    [[nodiscard]] T euclidean_norm() const noexcept { return T(-1); }

    [[nodiscard]] T absolute_sum() const noexcept { return T(-1); }

    BlockVector& add_scaled(T /*a*/, const BlockVector<T>& /*x*/) { throw std::runtime_error{"not implemented"}; }

    BlockVector& scale(T /*a*/) noexcept { return *this; }

    BlockVector& copy(const BlockVector<T>& /*src*/) { throw std::runtime_error{"not implemented"}; }

    void fill(T /*value*/) noexcept {}

    // -- convenience operators (mirror Vector) --------------------------------

    BlockVector& operator*=(T a) noexcept { return scale(a); }
    BlockVector& operator/=(T a) noexcept { return scale(T(1) / a); }
    BlockVector& operator+=(const BlockVector& x) { return add_scaled(T(1), x); }
    BlockVector& operator-=(const BlockVector& x) { return add_scaled(T(-1), x); }

private:
    /// @brief Throw if @p other is not conformable (block count / per-block sizes) with this vector.
    void check_conformable(const BlockVector<T>& /*other*/) const { throw std::runtime_error{"not implemented"}; }

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
 * @brief A `block_rows x block_cols` grid of type-erased sub-matrices, modelling a block matrix.
 * @tparam T Element type (an IEEE floating-point @ref Scalar).
 *
 * @par Heterogeneous, type-erased blocks
 * Each cell holds any type satisfying `MatrixOperator<M, T>`, stored type-erased behind a
 * `std::unique_ptr`. Because the block type is erased, one grid can mix @ref DenseMatrix,
 * @ref DiagonalMatrix, @ref ZeroMatrix, a CSR matrix, and any future @ref MatrixOperator -- and
 * @ref BlockMatrix itself stays templated only on the element type @p T.
 *
 * @par Populate before use
 * A freshly constructed grid is *empty*: every cell must be set with set_block() before a product
 * is taken. There is no implicit zero -- use a right-sized @ref ZeroMatrix for a structural-zero
 * block. Taking a product (or querying rows()/columns()) while a cell is unset throws
 * `std::logic_error`.
 *
 * @par Lazy dimension checking
 * Block-row heights and block-column widths are derived from the blocks themselves -- the height of
 * block-row `I` is the `rows()` of any block in that row, and the width of block-column `J` is the
 * `columns()` of any block in that column. Because the grid may be filled incrementally, these
 * dimensions are validated for consistency lazily, at product time, not in set_block().
 *
 * @see BlockVector, MatrixOperator, ZeroMatrix
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

    /**
     * @brief Install (move) @p matrix into grid cell `(i, j)`, type-erased behind the block seam.
     *
     * @p M may be any type satisfying `MatrixOperator<M, T>`, so different cells can hold different
     * concrete matrix types. Replaces whatever was previously in the cell.
     *
     * @code{.cpp}
     * BlockMatrix<double> a(2, 2);
     * a.set_block(0, 0, DenseMatrix<double>{{1, 2}, {3, 4}});
     * a.set_block(0, 1, ZeroMatrix<double>(2, 2)); // explicit structural-zero block
     * @endcode
     *
     * @tparam M The concrete matrix type (must model `MatrixOperator<M, T>`).
     * @param i Block-row index. @param j Block-column index. @param matrix The block, moved in.
     * @throws std::out_of_range if `(i, j)` is outside the grid.
     */
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

    /**
     * @brief Recover the underlying concrete object at `(i, j)`, undoing the type erasure.
     *
     * Use this to reach a block's concrete API (e.g. read a @ref DenseMatrix entry); @p M must name
     * the exact type that was passed to set_block() for this cell.
     *
     * @code{.cpp}
     * DenseMatrix<double>& d = a.block_as<DenseMatrix<double>>(0, 0);
     * double top_left = d(0, 0);
     * @endcode
     *
     * @tparam M The expected concrete type. @param i Block-row. @param j Block-column.
     * @return Reference to the stored concrete block.
     * @throws std::out_of_range if `(i, j)` is outside the grid; std::logic_error if the cell is
     *         unset; std::bad_cast if the stored block is not an @p M.
     */
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

    /**
     * @brief Block matrix-vector product into a pre-sized destination: `y <- alpha*op(A)*x + beta*y`.
     *
     * Computed block-wise. For @c None, `y_I <- alpha * Σ_J A_IJ * x_J + beta * y_I`; for
     * @c Transposed the roles of the block-row index `I` and block-column index `J` swap and each
     * block is applied transposed. Each destination sub-vector accumulates its contributions in
     * place: the first contributing block applies the caller's @p beta and the rest use `beta = 1`,
     * so @p beta scales each destination block exactly once (not once per source block) and no
     * temporary is needed.
     *
     * @param x     Operand block vector. For @c None its `block_count()` and each sub-vector length
     *              must match the block-column widths; for @c Transposed, the block-row heights.
     * @param y     Destination block vector, written in place; must already be sized. For @c None
     *              its `block_count()` and sub-vector lengths must match the block-row heights; for
     *              @c Transposed, the block-column widths.
     * @param alpha Scale applied to the product term (default 1).
     * @param beta  Scale applied to the existing @p y, once per destination block (default 0).
     * @param op    Whether to apply `A` (@c None) or `A^T` (@c Transposed).
     * @throws std::logic_error if any grid cell is unset.
     * @throws std::invalid_argument if block-row heights or block-column widths are inconsistent,
     *         or if @p x / @p y has the wrong block count or a wrong sub-vector length.
     */
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

    /**
     * @brief Block matrix-vector product `op(A)*x`, returning a freshly allocated result.
     *
     * Allocates a result @ref BlockVector sized to the block-row heights (or the block-column widths
     * when @p op is @c Transposed) and fills it via multiply_into() with `beta == 0`.
     *
     * @param x  Operand block vector (see multiply_into() for the length rules).
     * @param op Whether to apply `A` (@c None) or `A^T` (@c Transposed).
     * @return A new @ref BlockVector holding `op(A)*x`.
     * @throws std::logic_error if any grid cell is unset; std::invalid_argument on a
     *         structural/length mismatch (see multiply_into()).
     */
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

    /// @brief Block matrix-vector product `A*x` (the @c None case of multiply()).
    /// @param x Operand block vector. @return A new @ref BlockVector holding `A*x`.
    /// @throws std::logic_error if any cell is unset; std::invalid_argument on a size mismatch.
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
