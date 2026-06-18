/**
 * @file block_array.hpp
 * @brief A sequence of owned @ref miscibility::instrument::Array blocks
 *        (@ref miscibility::instrument::BlockArray) mirroring Array's BLAS-1-style surface
 *        block-wise.
 *
 * @code{.cpp}
 * using namespace miscibility::instrument;
 *
 * BlockArray<double> x{Array<double>{1, 2}, Array<double>{3, 4}};
 * x.scale(2.0);          // block-wise
 * double s = x.sum();    // summed across blocks
 * @endcode
 *
 * @par What it is
 * A BlockArray owns a sequence of @ref Array blocks and forwards the operations Array exposes
 * (after the Vector->Array rename: `dot`/`euclidean_norm` removed, `sum` added) to each block,
 * so the same solver code runs unchanged over a flat @ref Array or a blocked operand. Each
 * operation is pure host-side orchestration: it loops the blocks and forwards to that block's
 * own @ref Array member (summing the partials for the reductions). There is no cross-block SIMD
 * and no flattening copy, so the per-block SIMD/zero-pad guarantees are inherited unchanged.
 * Binary operands must be *conformable* -- equal block_count() and equal per-block size() -- or
 * `std::invalid_argument` is thrown.
 *
 * @par Header-only
 * C++23. Depends only on @ref array.hpp.
 */

#pragma once

#include "instrument/array.hpp"

#include <cstddef>
#include <initializer_list>
#include <stdexcept>
#include <utility>
#include <vector>

namespace miscibility::instrument {

/**
 * @brief A sequence of owned @ref Array blocks mirroring Array's BLAS-1-style surface block-wise.
 * @tparam T Element type (an IEEE floating-point @ref Scalar).
 *
 * Blocks may have *different* lengths. The BLAS-1-style surface -- sum(), absolute_sum(),
 * add_scaled(), scale(), copy(), fill(), and the `*=` / `/=` / `+=` / `-=` operators -- is
 * forwarded to the per-block @ref Array members. Componentwise transforms are intentionally
 * *not* forwarded (see block-array.md).
 */
template<Scalar T> class BlockArray {
public:
    using value_type = T;          ///< Element type.
    using size_type = std::size_t; ///< Index / dimension type.

    /// @brief Construct an empty block array (zero blocks).
    BlockArray() = default;

    /// @brief Adopt a sequence of blocks. @param blocks The block arrays (moved in).
    explicit BlockArray(std::vector<Array<T>> blocks) : blocks_(std::move(blocks)) {}

    /// @brief Construct from a braced list of blocks. @param blocks The block arrays.
    BlockArray(std::initializer_list<Array<T>> blocks) : blocks_(blocks.begin(), blocks.end()) {}

    /// @brief Number of blocks. @return The block count.
    [[nodiscard]] size_type block_count() const noexcept { return blocks_.size(); }

    /// @brief Access block @p i (checked). @param i Block index. @return The block array.
    /// @throws std::out_of_range if @p i is out of range.
    [[nodiscard]] Array<T>& block(size_type i) { return blocks_.at(i); }

    /// @brief Access block @p i (checked, const). @param i Block index. @return The block array.
    /// @throws std::out_of_range if @p i is out of range.
    [[nodiscard]] const Array<T>& block(size_type i) const { return blocks_.at(i); }

    /// @brief Total length summed across every block. @return The total element count.
    [[nodiscard]] size_type size() const
    {
        size_type total = 0;
        for (const auto& b : blocks_) {
            total += b.size();
        }
        return total;
    }

    /// @brief Append a block. @param v The block array to append (moved in).
    void push_block(Array<T> v) { blocks_.push_back(std::move(v)); }

    // -- operations (block-wise orchestration) --------------------------------

    /// @brief Plain sum `Sum_k block(k).sum()` across all blocks. @return The summed value; `0` for empty.
    [[nodiscard]] T sum() const noexcept
    {
        T total{};
        for (const auto& b : blocks_) {
            total += b.sum();
        }
        return total;
    }

    /// @brief Sum of magnitudes `Sum_k block(k).absolute_sum()` across all blocks (asum).
    /// @return The summed absolute sum; `0` for an empty block array.
    [[nodiscard]] T absolute_sum() const noexcept
    {
        T total{};
        for (const auto& b : blocks_) {
            total += b.absolute_sum();
        }
        return total;
    }

    /**
     * @brief Scaled accumulate `this <- this + a*x`, block-wise (axpy).
     * @param a Scale factor applied to @p x.
     * @param x Operand, conformable with @c *this (same block count and per-block sizes).
     * @return `*this`, to allow chaining.
     * @throws std::invalid_argument on a block-count or per-block size mismatch.
     */
    BlockArray& add_scaled(T a, const BlockArray<T>& x)
    {
        check_conformable(x);
        for (size_type k = 0; k < blocks_.size(); ++k) {
            blocks_[k].add_scaled(a, x.blocks_[k]);
        }
        return *this;
    }

    /// @brief In place scaling `this <- a*this`, block-wise (scal).
    /// @param a Scale factor. @return `*this`, to allow chaining.
    BlockArray& scale(T a) noexcept
    {
        for (auto& b : blocks_) {
            b.scale(a);
        }
        return *this;
    }

    /**
     * @brief In place value copy `this <- src`, block-wise (`block(k).copy(src.block(k))`).
     * @param src Source, conformable with @c *this (same block count and per-block sizes).
     * @return `*this`, to allow chaining.
     * @throws std::invalid_argument on a block-count or per-block size mismatch.
     *
     * The destination keeps its own storage -- nothing is allocated or resized, so a mismatch is a
     * caller error (thrown) rather than a trigger to grow.
     */
    BlockArray& copy(const BlockArray<T>& src)
    {
        check_conformable(src);
        for (size_type k = 0; k < blocks_.size(); ++k) {
            blocks_[k].copy(src.blocks_[k]);
        }
        return *this;
    }

    /// @brief Set every element of every block to @p value (each block's pad stays zero).
    /// @param value Value written to every logical element of every block.
    void fill(T value) noexcept
    {
        for (auto& b : blocks_) {
            b.fill(value);
        }
    }

    // -- convenience operators (built on the operations above, mirror Array) --

    /// @brief `this <- a*this`. @param a Scale factor. @return `*this`.
    BlockArray& operator*=(T a) noexcept { return scale(a); }
    /// @brief `this <- (1/a)*this`. @param a Divisor. @return `*this`.
    BlockArray& operator/=(T a) noexcept { return scale(T(1) / a); }
    /// @brief `this <- this + x`. @param x Conformable operand. @return `*this`.
    /// @throws std::invalid_argument on a block-count or per-block size mismatch.
    BlockArray& operator+=(const BlockArray& x) { return add_scaled(T(1), x); }
    /// @brief `this <- this - x`. @param x Conformable operand. @return `*this`.
    /// @throws std::invalid_argument on a block-count or per-block size mismatch.
    BlockArray& operator-=(const BlockArray& x) { return add_scaled(T(-1), x); }

private:
    /**
     * @brief Throw unless @p other is conformable with @c *this: equal block_count() and equal
     *        per-block size() for every block. Reused by add_scaled() and copy().
     * @param other The operand to validate against @c *this.
     * @throws std::invalid_argument on a block-count or per-block size mismatch.
     */
    void check_conformable(const BlockArray<T>& other) const
    {
        if (other.blocks_.size() != blocks_.size()) {
            throw std::invalid_argument{"miscibility::instrument::BlockArray block-count mismatch"};
        }
        for (size_type k = 0; k < blocks_.size(); ++k) {
            if (other.blocks_[k].size() != blocks_[k].size()) {
                throw std::invalid_argument{"miscibility::instrument::BlockArray block size mismatch"};
            }
        }
    }

    std::vector<Array<T>> blocks_; ///< The owned blocks.
};

} // namespace miscibility::instrument
