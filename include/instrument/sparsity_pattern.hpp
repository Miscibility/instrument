/**
 * @file sparsity_pattern.hpp
 * @brief A storage-format-agnostic description of a sparse matrix's entries, in canonical CSR form.
 *
 * @code{.cpp}
 * using miscibility::instrument::SparsityPattern;
 * SparsityPattern<double> p(2, 2, {{0, 0, 1}, {0, 1, 2}, {1, 0, 3}, {1, 1, 4}});
 * // p.row_offsets() == {0, 2, 4}; p.column_indices() == {0, 1, 0, 1}; p.values() == {1, 2, 3, 4}
 * @endcode
 *
 * @par What it is
 * A SparsityPattern records *which* entries a sparse matrix holds and their values, decoupled from
 * any particular sparse storage class. It is assembled from a bag of @ref MatrixEntry objects and,
 * once built, exposes the canonical CSR arrays (`row_offsets`, `column_indices`, `values`) that a
 * concrete sparse matrix (e.g. a CSR matrix) copies to build or refill itself. Separating "what is
 * in the matrix" from "how it is stored" keeps the same source usable by future sparse formats.
 *
 * @par Assembly
 * The input entries may arrive in any order and may repeat a `(row, col)`: the pattern sorts them
 * into row-major order and coalesces duplicates by **summing** their values. A `(row, col)` whose
 * values happen to sum to zero is *retained* as an explicit structural slot -- the pattern records
 * structure, and a numeric cancellation does not drop a slot.
 *
 * @par Header-only
 * C++23. Depends only on @ref Scalar; no BLAS/SIMD.
 */

#pragma once

#include "instrument/vector.hpp"

#include <algorithm>
#include <cstddef>
#include <initializer_list>
#include <span>
#include <stdexcept>
#include <vector>

namespace miscibility::instrument {

/**
 * @brief One triplet of a sparse matrix: the value at `(row, col)`.
 * @tparam T Element type (an IEEE floating-point @ref Scalar).
 *
 * An aggregate; entries with the same `(row, col)` are summed when a @ref SparsityPattern is built.
 */
template<Scalar T> struct MatrixEntry {
    std::size_t row; ///< Row index.
    std::size_t col; ///< Column index.
    T value;         ///< Value at `(row, col)`.
};

/**
 * @brief A sparse matrix's entries in canonical CSR form, assembled from @ref MatrixEntry triplets.
 * @tparam T Element type (an IEEE floating-point @ref Scalar).
 *
 * Built once from an explicit `rows x cols` shape and a bag of entries (see the file overview for
 * the sort/coalesce rules), then read-only: the three CSR arrays are the seam a concrete sparse
 * matrix copies.
 */
template<Scalar T> class SparsityPattern {
public:
    using value_type = T;          ///< Element type.
    using size_type = std::size_t; ///< Dimension / index type.

    /**
     * @brief Assemble a pattern of the given shape from a span of entries.
     * @param rows    Row count of the matrix.
     * @param cols    Column count of the matrix.
     * @param entries The triplets, in any order; duplicates at the same `(row, col)` are summed.
     * @throws std::out_of_range if any entry has `row >= rows` or `col >= cols`.
     */
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

    /// @brief Convenience overload taking a braced list of entries. @param rows Row count. @param cols Column count.
    /// @param entries The triplets (see the span constructor). @throws std::out_of_range if any entry is out of range.
    SparsityPattern(size_type rows, size_type cols, std::initializer_list<MatrixEntry<T>> entries) :
        SparsityPattern(rows, cols, std::span<const MatrixEntry<T>>{entries.begin(), entries.size()})
    {
    }

    [[nodiscard]] size_type rows() const noexcept { return rows_; }    ///< Row count of the matrix.
    [[nodiscard]] size_type columns() const noexcept { return cols_; } ///< Column count of the matrix.
    /// @brief Number of stored `(row, col)` slots after coalescing. @return The nonzero count.
    [[nodiscard]] size_type nonzeros() const noexcept { return values_.size(); }

    /// @brief CSR row offsets, length `rows() + 1`; `row_offsets()[i]..[i+1]` delimits row `i`. Non-decreasing.
    /// @return The row-offset array.
    [[nodiscard]] const std::vector<size_type>& row_offsets() const noexcept { return row_offsets_; }
    /// @brief CSR column indices, length `nonzeros()`; strictly increasing within each row. @return The column indices.
    [[nodiscard]] const std::vector<size_type>& column_indices() const noexcept { return column_indices_; }
    /// @brief CSR values, length `nonzeros()`; the coalesced value of each stored slot. @return The values.
    [[nodiscard]] const std::vector<T>& values() const noexcept { return values_; }

private:
    size_type rows_{0};                     ///< Row count.
    size_type cols_{0};                     ///< Column count.
    std::vector<size_type> row_offsets_;    ///< CSR row offsets (length `rows_ + 1`).
    std::vector<size_type> column_indices_; ///< CSR column indices (length `nonzeros()`).
    std::vector<T> values_;                 ///< CSR coalesced values (length `nonzeros()`).
};

} // namespace miscibility::instrument
