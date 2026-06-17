#pragma once

#include "instrument/vector.hpp"

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
        (void)entries;
        throw std::runtime_error{"not implemented"};
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
