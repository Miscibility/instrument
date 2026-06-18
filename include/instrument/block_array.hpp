#pragma once

#include "instrument/array.hpp"

#include <cmath>
#include <cstddef>
#include <initializer_list>
#include <stdexcept>
#include <utility>
#include <vector>

namespace miscibility::instrument {

template<Scalar T> class BlockArray {
public:
    using value_type = T;
    using size_type = std::size_t;

    BlockArray() = default;

    explicit BlockArray(std::vector<Array<T>> blocks) : blocks_(std::move(blocks)) {}

    BlockArray(std::initializer_list<Array<T>> blocks) : blocks_(blocks.begin(), blocks.end()) {}

    [[nodiscard]] size_type block_count() const noexcept { return blocks_.size(); }

    [[nodiscard]] Array<T>& block(size_type i) { return blocks_.at(i); }

    [[nodiscard]] const Array<T>& block(size_type i) const { return blocks_.at(i); }

    [[nodiscard]] size_type size() const
    {
        size_type total = 0;
        for (const auto& b : blocks_) {
            total += b.size();
        }
        return total;
    }

    void push_block(Array<T> v) { blocks_.push_back(std::move(v)); }

    void swap(BlockArray& other) noexcept { blocks_.swap(other.blocks_); }
    friend void swap(BlockArray& a, BlockArray& b) noexcept { a.swap(b); }

    void zero_pad() noexcept
    {
        for (auto& b : blocks_) {
            b.zero_pad();
        }
    }

    [[nodiscard]] T sum() const noexcept
    {
        T total{};
        for (const auto& b : blocks_) {
            total += b.sum();
        }
        return total;
    }

    [[nodiscard]] T absolute_sum() const noexcept
    {
        T total{};
        for (const auto& b : blocks_) {
            total += b.absolute_sum();
        }
        return total;
    }

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

    BlockArray& add_scaled(T a, const BlockArray<T>& x)
    {
        check_conformable(x);
        for (size_type k = 0; k < blocks_.size(); ++k) {
            blocks_[k].add_scaled(a, x.blocks_[k]);
        }
        return *this;
    }

    BlockArray& scale(T a) noexcept
    {
        for (auto& b : blocks_) {
            b.scale(a);
        }
        return *this;
    }

    BlockArray& copy(const BlockArray<T>& src)
    {
        check_conformable(src);
        for (size_type k = 0; k < blocks_.size(); ++k) {
            blocks_[k].copy(src.blocks_[k]);
        }
        return *this;
    }

    void fill(T value) noexcept
    {
        for (auto& b : blocks_) {
            b.fill(value);
        }
    }

    template<class F> BlockArray& apply(F f) noexcept
    {
        for (auto& b : blocks_) {
            b.apply(f);
        }
        return *this;
    }

    BlockArray& abs() noexcept
    {
        return for_each_block([](auto& b) { b.abs(); });
    }
    BlockArray& sqrt() noexcept
    {
        return for_each_block([](auto& b) { b.sqrt(); });
    }
    BlockArray& exp() noexcept
    {
        return for_each_block([](auto& b) { b.exp(); });
    }
    BlockArray& exp2() noexcept
    {
        return for_each_block([](auto& b) { b.exp2(); });
    }
    BlockArray& expm1() noexcept
    {
        return for_each_block([](auto& b) { b.expm1(); });
    }
    BlockArray& log() noexcept
    {
        return for_each_block([](auto& b) { b.log(); });
    }
    BlockArray& log2() noexcept
    {
        return for_each_block([](auto& b) { b.log2(); });
    }
    BlockArray& log10() noexcept
    {
        return for_each_block([](auto& b) { b.log10(); });
    }
    BlockArray& log1p() noexcept
    {
        return for_each_block([](auto& b) { b.log1p(); });
    }
    BlockArray& sin() noexcept
    {
        return for_each_block([](auto& b) { b.sin(); });
    }
    BlockArray& cos() noexcept
    {
        return for_each_block([](auto& b) { b.cos(); });
    }
    BlockArray& sinh() noexcept
    {
        return for_each_block([](auto& b) { b.sinh(); });
    }
    BlockArray& tanh() noexcept
    {
        return for_each_block([](auto& b) { b.tanh(); });
    }
    BlockArray& asin() noexcept
    {
        return for_each_block([](auto& b) { b.asin(); });
    }
    BlockArray& acos() noexcept
    {
        return for_each_block([](auto& b) { b.acos(); });
    }
    BlockArray& asinh() noexcept
    {
        return for_each_block([](auto& b) { b.asinh(); });
    }
    BlockArray& acosh() noexcept
    {
        return for_each_block([](auto& b) { b.acosh(); });
    }
    BlockArray& atan() noexcept
    {
        return for_each_block([](auto& b) { b.atan(); });
    }
    BlockArray& atanh() noexcept
    {
        return for_each_block([](auto& b) { b.atanh(); });
    }

    BlockArray& elementwise_product(const BlockArray<T>& x)
    {
        check_conformable(x);
        for (size_type k = 0; k < blocks_.size(); ++k) {
            blocks_[k].elementwise_product(x.blocks_[k]);
        }
        return *this;
    }

    BlockArray& elementwise_quotient(const BlockArray<T>& x)
    {
        check_conformable(x);
        for (size_type k = 0; k < blocks_.size(); ++k) {
            blocks_[k].elementwise_quotient(x.blocks_[k]);
        }
        return *this;
    }

    BlockArray& operator*=(T a) noexcept { return scale(a); }
    BlockArray& operator/=(T a) noexcept { return scale(T(1) / a); }
    BlockArray& operator+=(const BlockArray& x) { return add_scaled(T(1), x); }
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
