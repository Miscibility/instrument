#pragma once

#include "instrument/array.hpp"

#include <ginkgo/ginkgo.hpp>
#include <stdexcept>

namespace miscibility::instrument {

/// A host-side assembly buffer for building a sparse matrix entry by entry.
///
/// A ``SparsityPattern`` collects ``(row, col, value)`` contributions keyed on the
/// coordinate, accumulating repeated hits on the same entry — the natural pattern
/// for FEM/FVM assembly, where one ``(i, j)`` receives several element
/// contributions. When assembly is done, :cpp:`to_matrix_data` exports a sorted,
/// deduplicated listing that any concrete matrix format can read.
///
/// This is pure data: it carries no executor and no operator, so it does not
/// participate in the handle/timing machinery.
///
/// .. code-block:: cpp
///
///     SparsityPattern<double> pattern{gko::dim<2>{3, 3}};
///     pattern.add_value(0, 0, 1.0);
///     pattern.add_value(0, 0, 4.0);          // accumulates -> 5.0
///     auto data = pattern.to_matrix_data();  // feed to a matrix's read()
///
/// :tparam T: Floating-point value type.
/// :tparam I: Index type for rows and columns (defaults to ``int``).
template<Scalar T = double, class I = int> class SparsityPattern {
public:
    /// Creates an empty pattern of the given shape.
    ///
    /// :param size: The matrix shape as ``{rows, cols}``.
    explicit SparsityPattern(gko::dim<2> size);

    /// Adds ``value`` into entry ``(row, col)``, summing with anything already there.
    ///
    /// :throws std::out_of_range: if ``row`` or ``col`` is outside :cpp:`size`.
    void add_value(I row, I col, T value);

    /// Sets entry ``(row, col)`` to ``value``, replacing any accumulated value.
    ///
    /// :throws std::out_of_range: if ``row`` or ``col`` is outside :cpp:`size`.
    void set_value(I row, I col, T value);

    /// The current value at ``(row, col)``, or zero if nothing was stored there.
    ///
    /// :throws std::out_of_range: if ``row`` or ``col`` is outside :cpp:`size`.
    [[nodiscard]] T get_value(I row, I col) const;

    /// The matrix shape as ``{rows, cols}``.
    [[nodiscard]] gko::dim<2> size() const noexcept;
    /// The number of distinct stored entries.
    [[nodiscard]] gko::size_type num_nonzeros() const;

    /// Exports the entries as sorted, row-major, unique-coordinate ``matrix_data``.
    ///
    /// Explicitly inserted zeros are kept — structural zeros are not pruned.
    [[nodiscard]] gko::matrix_data<T, I> to_matrix_data() const;

private:
    void check_bounds(I row, I col) const;

    mutable gko::matrix_assembly_data<T, I> assembly_;
};

template<Scalar T, class I> SparsityPattern<T, I>::SparsityPattern(gko::dim<2> size) : assembly_{size} {}

template<Scalar T, class I> void SparsityPattern<T, I>::add_value(I row, I col, T value)
{
    check_bounds(row, col);
    assembly_.add_value(row, col, value);
}

template<Scalar T, class I> void SparsityPattern<T, I>::set_value(I row, I col, T value)
{
    check_bounds(row, col);
    assembly_.set_value(row, col, value);
}

template<Scalar T, class I> T SparsityPattern<T, I>::get_value(I row, I col) const
{
    check_bounds(row, col);
    return assembly_.get_value(row, col);
}

template<Scalar T, class I> gko::dim<2> SparsityPattern<T, I>::size() const noexcept { return assembly_.get_size(); }

template<Scalar T, class I> gko::size_type SparsityPattern<T, I>::num_nonzeros() const
{
    return assembly_.get_num_stored_elements();
}

template<Scalar T, class I> gko::matrix_data<T, I> SparsityPattern<T, I>::to_matrix_data() const
{
    return assembly_.get_ordered_data();
}

template<Scalar T, class I> void SparsityPattern<T, I>::check_bounds(I row, I col) const
{
    const gko::dim<2> shape = assembly_.get_size();
    if (row < 0 || col < 0 || static_cast<gko::size_type>(row) >= shape[0] ||
        static_cast<gko::size_type>(col) >= shape[1]) {
        throw std::out_of_range{"miscibility::instrument::SparsityPattern index out of range"};
    }
}

} // namespace miscibility::instrument
