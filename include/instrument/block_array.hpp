#pragma once

#include "instrument/array.hpp"

#include <cmath>
#include <cstddef>
#include <initializer_list>
#include <stdexcept>
#include <utility>
#include <vector>

namespace miscibility::instrument {

/// A sequence of :cpp:`Array` blocks treated as one logical vector.
///
/// A block array stores several independent arrays and exposes the same
/// arithmetic vocabulary as a single :cpp:`Array` — reductions, scaling, AXPY,
/// element-wise math — applied across all blocks at once. Each operation works
/// block by block, so the blocks need not be the same length; two block arrays
/// are *conformable* when they have the same block count and matching lengths
/// block for block.
///
/// This suits problems whose state is naturally partitioned (for example
/// per-field or per-region unknowns) but should still be manipulated as a
/// single vector by a solver.
///
/// .. code-block:: cpp
///
///     BlockArray<double> u{Array<double>{1.0, 2.0}, Array<double>{3.0, 4.0, 5.0}};
///     u.scale(0.5).add_scaled(1.0, v);   // u = 0.5*u + v, across every block
///     double total = u.sum();
///
/// :tparam T: Floating-point element type of the underlying arrays.
template<Scalar T> class BlockArray {
public:
    using value_type = T;
    using size_type = std::size_t;

    /// Constructs a block array with no blocks.
    BlockArray() = default;

    /// Constructs a block array that takes ownership of ``blocks``.
    explicit BlockArray(std::vector<Array<T>> blocks) : blocks_(std::move(blocks)) {}

    /// Constructs a block array from a brace-enclosed list of arrays.
    BlockArray(std::initializer_list<Array<T>> blocks) : blocks_(blocks.begin(), blocks.end()) {}

    /// Number of blocks.
    [[nodiscard]] size_type block_count() const noexcept { return blocks_.size(); }

    /// Bounds-checked access to the ``i``-th block.
    ///
    /// :throws std::out_of_range: if ``i`` is not less than :cpp:`block_count`.
    [[nodiscard]] Array<T>& block(size_type i) { return blocks_.at(i); }

    /// Bounds-checked access to the ``i``-th block.
    ///
    /// :throws std::out_of_range: if ``i`` is not less than :cpp:`block_count`.
    [[nodiscard]] const Array<T>& block(size_type i) const { return blocks_.at(i); }

    /// Total number of elements summed over every block.
    [[nodiscard]] size_type size() const
    {
        size_type total = 0;
        for (const auto& b : blocks_) {
            total += b.size();
        }
        return total;
    }

    /// Appends ``v`` as a new trailing block.
    void push_block(Array<T> v) { blocks_.push_back(std::move(v)); }

    /// Swaps contents with ``other``.
    void swap(BlockArray& other) noexcept { blocks_.swap(other.blocks_); }
    /// Swaps the contents of two block arrays.
    friend void swap(BlockArray& a, BlockArray& b) noexcept { a.swap(b); }

    /// Re-zeros the SIMD padding of every block (see :cpp:`Array::zero_pad`).
    void zero_pad() noexcept
    {
        for (auto& b : blocks_) {
            b.zero_pad();
        }
    }

    /// Sum of all elements across every block.
    [[nodiscard]] T sum() const noexcept
    {
        T total{};
        for (const auto& b : blocks_) {
            total += b.sum();
        }
        return total;
    }

    /// Sum of the absolute values of all elements (the L1 norm).
    [[nodiscard]] T absolute_sum() const noexcept
    {
        T total{};
        for (const auto& b : blocks_) {
            total += b.absolute_sum();
        }
        return total;
    }

    /// Flattened index of the largest-magnitude element across all blocks.
    ///
    /// Indices are counted as if the blocks were concatenated in order. On ties
    /// the earliest such element wins. An empty block array reports :cpp:`size`.
    ///
    /// :returns: The concatenated-vector index of the largest-magnitude element,
    ///     or :cpp:`size` if there are no elements.
    [[nodiscard]] size_type index_of_max_magnitude() const noexcept
    {
        const size_type total = size();
        size_type best = total;
        T best_mag{};
        size_type offset = 0;
        for (const auto& b : blocks_) {
            if (!b.empty()) {
                const size_type local = b.index_of_max_magnitude();
                const T mag = std::abs(b[local]);
                if (best == total || mag > best_mag) {
                    best = offset + local;
                    best_mag = mag;
                }
            }
            offset += b.size();
        }
        return best;
    }

    /// Largest absolute element value over all blocks (the infinity norm).
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

    /// Adds a scaled block array in place (``y := y + a*x``), block by block.
    ///
    /// :param a: Scalar multiplier applied to ``x``.
    /// :param x: Block array to add; must be conformable with this one.
    /// :throws std::invalid_argument: if ``x`` differs in block count or in any block's length.
    BlockArray& add_scaled(T a, const BlockArray<T>& x)
    {
        check_conformable(x);
        for (size_type k = 0; k < blocks_.size(); ++k) {
            blocks_[k].add_scaled(a, x.blocks_[k]);
        }
        return *this;
    }

    /// Multiplies every element by ``a`` in place.
    BlockArray& scale(T a) noexcept
    {
        for (auto& b : blocks_) {
            b.scale(a);
        }
        return *this;
    }

    /// Copies the contents of ``src`` block by block.
    ///
    /// :param src: Source block array; must be conformable with this one.
    /// :throws std::invalid_argument: if ``src`` differs in block count or in any block's length.
    BlockArray& copy(const BlockArray<T>& src)
    {
        check_conformable(src);
        for (size_type k = 0; k < blocks_.size(); ++k) {
            blocks_[k].copy(src.blocks_[k]);
        }
        return *this;
    }

    /// Sets every element of every block to ``value``.
    void fill(T value) noexcept
    {
        for (auto& b : blocks_) {
            b.fill(value);
        }
    }

    /// Applies a SIMD lambda to every element of every block (see :cpp:`Array::apply`).
    ///
    /// :param f: Callable ``(descriptor, vector) -> vector`` applied across each block.
    template<class F> BlockArray& apply(F f) noexcept
    {
        for (auto& b : blocks_) {
            b.apply(f);
        }
        return *this;
    }

    /// Replaces each element with its absolute value.
    BlockArray& abs() noexcept
    {
        return for_each_block([](auto& b) { b.abs(); });
    }
    /// Replaces each element with its square root.
    BlockArray& sqrt() noexcept
    {
        return for_each_block([](auto& b) { b.sqrt(); });
    }
    /// Replaces each element ``x`` with ``e**x``.
    BlockArray& exp() noexcept
    {
        return for_each_block([](auto& b) { b.exp(); });
    }
    /// Replaces each element ``x`` with ``2**x``.
    BlockArray& exp2() noexcept
    {
        return for_each_block([](auto& b) { b.exp2(); });
    }
    /// Replaces each element ``x`` with ``e**x - 1``, accurate for small ``x``.
    BlockArray& expm1() noexcept
    {
        return for_each_block([](auto& b) { b.expm1(); });
    }
    /// Replaces each element with its natural logarithm.
    BlockArray& log() noexcept
    {
        return for_each_block([](auto& b) { b.log(); });
    }
    /// Replaces each element with its base-2 logarithm.
    BlockArray& log2() noexcept
    {
        return for_each_block([](auto& b) { b.log2(); });
    }
    /// Replaces each element with its base-10 logarithm.
    BlockArray& log10() noexcept
    {
        return for_each_block([](auto& b) { b.log10(); });
    }
    /// Replaces each element ``x`` with ``log(1 + x)``, accurate for small ``x``.
    BlockArray& log1p() noexcept
    {
        return for_each_block([](auto& b) { b.log1p(); });
    }
    /// Replaces each element with its sine (argument in radians).
    BlockArray& sin() noexcept
    {
        return for_each_block([](auto& b) { b.sin(); });
    }
    /// Replaces each element with its cosine (argument in radians).
    BlockArray& cos() noexcept
    {
        return for_each_block([](auto& b) { b.cos(); });
    }
    /// Replaces each element with its hyperbolic sine.
    BlockArray& sinh() noexcept
    {
        return for_each_block([](auto& b) { b.sinh(); });
    }
    /// Replaces each element with its hyperbolic tangent.
    BlockArray& tanh() noexcept
    {
        return for_each_block([](auto& b) { b.tanh(); });
    }
    /// Replaces each element with its arc sine (result in radians).
    BlockArray& asin() noexcept
    {
        return for_each_block([](auto& b) { b.asin(); });
    }
    /// Replaces each element with its arc cosine (result in radians).
    BlockArray& acos() noexcept
    {
        return for_each_block([](auto& b) { b.acos(); });
    }
    /// Replaces each element with its inverse hyperbolic sine.
    BlockArray& asinh() noexcept
    {
        return for_each_block([](auto& b) { b.asinh(); });
    }
    /// Replaces each element with its inverse hyperbolic cosine.
    BlockArray& acosh() noexcept
    {
        return for_each_block([](auto& b) { b.acosh(); });
    }
    /// Replaces each element with its arc tangent (result in radians).
    BlockArray& atan() noexcept
    {
        return for_each_block([](auto& b) { b.atan(); });
    }
    /// Replaces each element with its inverse hyperbolic tangent.
    BlockArray& atanh() noexcept
    {
        return for_each_block([](auto& b) { b.atanh(); });
    }

    /// Multiplies element-by-element by ``x`` (the Hadamard product), block by block.
    ///
    /// :param x: Block array of factors; must be conformable with this one.
    /// :throws std::invalid_argument: if ``x`` differs in block count or in any block's length.
    BlockArray& elementwise_product(const BlockArray<T>& x)
    {
        check_conformable(x);
        for (size_type k = 0; k < blocks_.size(); ++k) {
            blocks_[k].elementwise_product(x.blocks_[k]);
        }
        return *this;
    }

    /// Divides element-by-element by ``x``, block by block.
    ///
    /// :param x: Block array of divisors; must be conformable with this one.
    /// :throws std::invalid_argument: if ``x`` differs in block count or in any block's length.
    BlockArray& elementwise_quotient(const BlockArray<T>& x)
    {
        check_conformable(x);
        for (size_type k = 0; k < blocks_.size(); ++k) {
            blocks_[k].elementwise_quotient(x.blocks_[k]);
        }
        return *this;
    }

    /// Scales the block array by ``a`` (see :cpp:`scale`).
    BlockArray& operator*=(T a) noexcept { return scale(a); }
    /// Divides the block array by ``a``.
    BlockArray& operator/=(T a) noexcept { return scale(T(1) / a); }
    /// Adds ``x`` element-wise (see :cpp:`add_scaled`).
    BlockArray& operator+=(const BlockArray& x) { return add_scaled(T(1), x); }
    /// Subtracts ``x`` element-wise.
    BlockArray& operator-=(const BlockArray& x) { return add_scaled(T(-1), x); }

private:
    template<class Op> BlockArray& for_each_block(Op op) noexcept
    {
        for (auto& b : blocks_) {
            op(b);
        }
        return *this;
    }

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

    std::vector<Array<T>> blocks_;
};

} // namespace miscibility::instrument
