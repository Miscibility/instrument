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

#include <cmath>
#include <cstddef>
#include <initializer_list>
#include <stdexcept>
#include <utility>
#include <vector>

namespace miscibility::instrument {

/**
 * @brief A sequence of owned @ref Array blocks mirroring Array's numeric surface block-wise.
 * @tparam T Element type (an IEEE floating-point @ref Scalar).
 *
 * Blocks may have *different* lengths. The full @ref Array numeric surface is forwarded to the
 * per-block members:
 * - reductions: sum(), absolute_sum(), index_of_max_magnitude(), max_magnitude();
 * - BLAS-1: scale(), add_scaled(), copy(), fill(), and the `*=` / `/=` / `+=` / `-=` operators;
 * - componentwise transforms: apply(), abs(), sqrt(), every Highway transcendental, and the
 *   binary elementwise_product() / elementwise_quotient().
 *
 * Each is pure host-side orchestration -- a loop over the blocks forwarding to that block's own
 * @ref Array member (summing the partials for the reductions). There is no cross-block SIMD and
 * no flattening copy, so the per-block SIMD/zero-pad guarantees are inherited unchanged; in
 * particular the componentwise transforms re-establish each block's zero pad themselves. Binary
 * operands must be *conformable* -- equal block_count() and equal per-block size().
 *
 * index_of_max_magnitude() reports an index into the logical *concatenation* of the blocks (a
 * global offset), to mirror @ref Array's flat contract.
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

    /// @brief Exchange contents with @p other. @param other Block array to swap with.
    void swap(BlockArray& other) noexcept { blocks_.swap(other.blocks_); }
    /// @brief ADL swap: exchange the contents of @p a and @p b.
    friend void swap(BlockArray& a, BlockArray& b) noexcept { a.swap(b); }

    /// @brief Restore each block's zero-pad invariant (forwards to every block's zero_pad()).
    void zero_pad() noexcept
    {
        for (auto& b : blocks_) {
            b.zero_pad();
        }
    }

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
     * @brief Index of the largest-magnitude element across all blocks (iamax).
     * @return The index into the logical concatenation of the blocks (a global offset) of the
     *         element with the greatest `|element|`; the smallest such index on ties. Returns
     *         `size()` for an empty block array (no element exists).
     */
    [[nodiscard]] size_type index_of_max_magnitude() const noexcept
    {
        const size_type total = size();
        size_type best = total; // "no element" sentinel (== total; never a valid index)
        T best_mag{};
        size_type offset = 0;
        for (const auto& b : blocks_) {
            if (!b.empty()) {
                const size_type local = b.index_of_max_magnitude();
                const T mag = std::abs(b[local]);
                if (best == total || mag > best_mag) { // strict '>': smallest global index wins ties
                    best = offset + local;
                    best_mag = mag;
                }
            }
            offset += b.size();
        }
        return best;
    }

    /// @brief Largest element magnitude across all blocks (related to iamax).
    /// @return The maximum `|element|` over every block; `T{}` (zero) for an empty block array.
    [[nodiscard]] T max_magnitude() const noexcept
    {
        T best{};
        for (const auto& b : blocks_) {
            const T m = b.max_magnitude();
            if (m > best) {
                best = m;
            }
        }
        return best;
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

    // -- componentwise transforms (block-wise; each block restores its own pad) -

    /**
     * @brief Generic in place unary lane-wise transform applied to every block.
     * @tparam F Highway functor with signature `Vec f(ScalableTag<T> d, Vec v)`.
     * @param f Lane-wise transform forwarded to each block's @ref Array::apply.
     * @return `*this`, to allow chaining.
     */
    template<class F> BlockArray& apply(F f) noexcept
    {
        for (auto& b : blocks_) {
            b.apply(f);
        }
        return *this;
    }

    /// @brief In place absolute value `b_i <- |b_i|`, block-wise. @return `*this`, for chaining.
    BlockArray& abs() noexcept { return for_each_block([](auto& b) { b.abs(); }); }
    /// @brief In place square root, block-wise. @return `*this`, for chaining.
    BlockArray& sqrt() noexcept { return for_each_block([](auto& b) { b.sqrt(); }); }
    /// @brief In place natural exponential, block-wise. @return `*this`, for chaining.
    BlockArray& exp() noexcept { return for_each_block([](auto& b) { b.exp(); }); }
    /// @brief In place base-2 exponential, block-wise. @return `*this`, for chaining.
    BlockArray& exp2() noexcept { return for_each_block([](auto& b) { b.exp2(); }); }
    /// @brief In place `exp(x) - 1`, block-wise. @return `*this`, for chaining.
    BlockArray& expm1() noexcept { return for_each_block([](auto& b) { b.expm1(); }); }
    /// @brief In place natural logarithm, block-wise. @return `*this`, for chaining.
    BlockArray& log() noexcept { return for_each_block([](auto& b) { b.log(); }); }
    /// @brief In place base-2 logarithm, block-wise. @return `*this`, for chaining.
    BlockArray& log2() noexcept { return for_each_block([](auto& b) { b.log2(); }); }
    /// @brief In place base-10 logarithm, block-wise. @return `*this`, for chaining.
    BlockArray& log10() noexcept { return for_each_block([](auto& b) { b.log10(); }); }
    /// @brief In place `log(1 + x)`, block-wise. @return `*this`, for chaining.
    BlockArray& log1p() noexcept { return for_each_block([](auto& b) { b.log1p(); }); }
    /// @brief In place sine (radians), block-wise. @return `*this`, for chaining.
    BlockArray& sin() noexcept { return for_each_block([](auto& b) { b.sin(); }); }
    /// @brief In place cosine (radians), block-wise. @return `*this`, for chaining.
    BlockArray& cos() noexcept { return for_each_block([](auto& b) { b.cos(); }); }
    /// @brief In place hyperbolic sine, block-wise. @return `*this`, for chaining.
    BlockArray& sinh() noexcept { return for_each_block([](auto& b) { b.sinh(); }); }
    /// @brief In place hyperbolic tangent, block-wise. @return `*this`, for chaining.
    BlockArray& tanh() noexcept { return for_each_block([](auto& b) { b.tanh(); }); }
    /// @brief In place arc sine (radians), block-wise. @return `*this`, for chaining.
    BlockArray& asin() noexcept { return for_each_block([](auto& b) { b.asin(); }); }
    /// @brief In place arc cosine (radians), block-wise. @return `*this`, for chaining.
    BlockArray& acos() noexcept { return for_each_block([](auto& b) { b.acos(); }); }
    /// @brief In place inverse hyperbolic sine, block-wise. @return `*this`, for chaining.
    BlockArray& asinh() noexcept { return for_each_block([](auto& b) { b.asinh(); }); }
    /// @brief In place inverse hyperbolic cosine, block-wise. @return `*this`, for chaining.
    BlockArray& acosh() noexcept { return for_each_block([](auto& b) { b.acosh(); }); }
    /// @brief In place arc tangent (radians), block-wise. @return `*this`, for chaining.
    BlockArray& atan() noexcept { return for_each_block([](auto& b) { b.atan(); }); }
    /// @brief In place inverse hyperbolic tangent, block-wise. @return `*this`, for chaining.
    BlockArray& atanh() noexcept { return for_each_block([](auto& b) { b.atanh(); }); }

    /**
     * @brief In place Hadamard (elementwise) product `this <- this * x`, block-wise.
     * @param x Operand, conformable with @c *this (same block count and per-block sizes).
     * @return `*this`, to allow chaining.
     * @throws std::invalid_argument on a block-count or per-block size mismatch.
     */
    BlockArray& elementwise_product(const BlockArray<T>& x)
    {
        check_conformable(x);
        for (size_type k = 0; k < blocks_.size(); ++k) {
            blocks_[k].elementwise_product(x.blocks_[k]);
        }
        return *this;
    }

    /**
     * @brief In place elementwise division `this <- this / x`, block-wise.
     * @param x Operand, conformable with @c *this (same block count and per-block sizes).
     * @return `*this`, to allow chaining.
     * @throws std::invalid_argument on a block-count or per-block size mismatch.
     */
    BlockArray& elementwise_quotient(const BlockArray<T>& x)
    {
        check_conformable(x);
        for (size_type k = 0; k < blocks_.size(); ++k) {
            blocks_[k].elementwise_quotient(x.blocks_[k]);
        }
        return *this;
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
    /// @brief Apply @p op to every block in order and return `*this` (the unary-transform helper).
    template<class Op> BlockArray& for_each_block(Op op) noexcept
    {
        for (auto& b : blocks_) {
            op(b);
        }
        return *this;
    }

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
