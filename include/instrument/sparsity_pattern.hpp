#pragma once

#include "instrument/vector.hpp"

#include <algorithm>
#include <cstddef>
#include <initializer_list>
#include <span>
#include <stdexcept>
#include <vector>

namespace miscibility::instrument {

template<Scalar T> struct MatrixEntry {
    std::size_t row;
    std::size_t col;
    T value;
};

template<Scalar T> class SparsityPattern {
public:
    using value_type = T;
    using size_type = std::size_t;

    SparsityPattern(size_type rows, size_type cols, std::span<const MatrixEntry<T>> entries) : rows_(rows), cols_(cols)
    {
        // Validate every entry against the explicit shape before any work.
        for (const auto& e : entries) {
            if (e.row >= rows_ || e.col >= cols_) {
                throw std::out_of_range{"miscibility::instrument::SparsityPattern entry out of range"};
            }
        }

        // Sort into canonical row-major order (by row, then column). Stable so that
        // duplicate (row, col) entries coalesce in their original relative order.
        std::vector<MatrixEntry<T>> sorted(entries.begin(), entries.end());
        std::stable_sort(sorted.begin(), sorted.end(), [](const MatrixEntry<T>& a, const MatrixEntry<T>& b) {
            return (a.row != b.row) ? (a.row < b.row) : (a.col < b.col);
        });

        // Coalesce runs of equal (row, col) by summing their values. A run that sums
        // to zero is kept as an explicit structural slot (structure != numeric value).
        row_offsets_.assign(rows_ + 1, 0);
        for (size_type i = 0; i < sorted.size();) {
            const size_type row = sorted[i].row;
            const size_type col = sorted[i].col;
            T sum = sorted[i].value;
            size_type j = i + 1;
            for (; j < sorted.size() && sorted[j].row == row && sorted[j].col == col; ++j) {
                sum += sorted[j].value;
            }
            column_indices_.push_back(col);
            values_.push_back(sum);
            ++row_offsets_[row + 1]; // count nonzeros per row
            i = j;
        }

        // Prefix-sum the per-row counts into CSR row offsets.
        for (size_type r = 0; r < rows_; ++r) {
            row_offsets_[r + 1] += row_offsets_[r];
        }
    }

    SparsityPattern(size_type rows, size_type cols, std::initializer_list<MatrixEntry<T>> entries)
        : SparsityPattern(rows, cols, std::span<const MatrixEntry<T>>{entries.begin(), entries.size()})
    {
    }

    [[nodiscard]] size_type rows() const noexcept { return rows_; }
    [[nodiscard]] size_type columns() const noexcept { return cols_; }
    [[nodiscard]] size_type nonzeros() const noexcept { return values_.size(); }

    [[nodiscard]] const std::vector<size_type>& row_offsets() const noexcept { return row_offsets_; }
    [[nodiscard]] const std::vector<size_type>& column_indices() const noexcept { return column_indices_; }
    [[nodiscard]] const std::vector<T>& values() const noexcept { return values_; }

private:
    size_type rows_{0};
    size_type cols_{0};
    std::vector<size_type> row_offsets_;
    std::vector<size_type> column_indices_;
    std::vector<T> values_;
};

} // namespace miscibility::instrument
