/**
 * @file matrix.hpp
 * @brief A dense numeric matrix with a compile-time *or* runtime shape, backed by a Vector.
 *
 * @code{.cpp}
 * miscibility::instrument::DenseMatrix<double, 3, 3>   // static 3x3, stored inline (no heap)
 * miscibility::instrument::DenseMatrix<double>         // runtime shape, heap-backed
 * @endcode
 *
 * @par Storage & layout
 * A DenseMatrix is a thin **column-major** (Fortran-order) view over a single contiguous
 * `Vector<T, …>`: element `(i, j)` lives at offset `j*rows() + i`, and the leading dimension is
 * exactly `rows()`. The logical `rows()*columns()` elements are packed densely at the front of the
 * buffer, so the raw pointer from data() hands straight to CBLAS (`CblasColMajor`) and LAPACK
 * (`LAPACK_COL_MAJOR`) with no transpose or copy. Because the storage is a `Vector`, the buffer is
 * over-aligned and zero-padded just like any other Vector, and the matrix inherits every
 * elementwise kernel the Vector already provides.
 *
 * @par Static vs dynamic storage
 * A matrix is *static* when **both** `Rows` and `Cols` are given (`DenseMatrix<double, 3, 3>`) and
 * *dynamic* when **both** are `dynamic` (`DenseMatrix<double>`); a mixed shape is rejected with a
 * `static_assert`. A static matrix stores all `Rows*Cols` elements inline.
 *
 * @warning A static `DenseMatrix<T, R, C>` is stored **inline** (on the stack when it is a local),
 * and its footprint grows as the *product* `R*C` -- so it overflows the stack far more easily than
 * a static Vector. `DenseMatrix<double, 1000, 1000>` is ~8&nbsp;MB and will blow a typical stack
 * instantly. **Use compile-time sizes ONLY for small matrices** (2x2, 3x3, small block
 * dimensions); reach for the dynamic `DenseMatrix<double>` for anything large.
 *
 * @par A DenseMatrix always holds the matrix A
 * There is no in-place factorization and no hidden state: operator()/at always return the entries
 * of the matrix as stored. Factorizing a matrix produces a *separate* object (see the
 * factorization types) constructed from this one -- the source is either copied (preserved) or
 * moved-from (left a valid empty 0x0), never silently repurposed.
 *
 * @par Elementwise operations
 * The elementwise surface delegates to the backing Vector: scalar scale() / `*=` / `/=`,
 * add_scaled() / `+=` / `-=`, the Hadamard elementwise_product() / elementwise_quotient(), and the
 * full set of componentwise transforms (abs(), sqrt(), exp(), log(), the trigonometric and
 * hyperbolic ops, ...). The binary forms require the two operands to have the *same shape* (equal
 * `rows()` and `columns()`), not merely the same element count.
 *
 * @par Header-only
 * C++23. Backs onto CBLAS/LAPACK (OpenBLAS); the BLAS-2/BLAS-3 and factorization kernels live in
 * sibling headers.
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <stdexcept>
#include <utility>

#include <cblas.h>

#include "instrument/vector.hpp"

/// @namespace miscibility::instrument
/// @brief Instrument library for the Miscibility project; this header adds a dense matrix.
namespace miscibility::instrument {

/**
 * @brief Human-readable transpose flag for the matrix kernels; maps onto the CBLAS flag.
 *
 * Restricted to real (non-complex) operands, so there is no conjugate-transpose variant.
 */
enum class Transpose : std::uint8_t {
    None,       ///< Use the matrix as stored: `op(A) == A`.
    Transposed, ///< Use the transpose: `op(A) == A^T`.
};

/// @namespace miscibility::instrument::detail
/// @brief Implementation details. Not part of the public API; documented for maintainers.
namespace detail {

/// @internal @brief Map the human-readable Transpose enum onto the CBLAS flag.
[[nodiscard]] inline CBLAS_TRANSPOSE to_cblas(Transpose t) noexcept
{
    return t == Transpose::Transposed ? CblasTrans : CblasNoTrans;
}

/// @internal @brief True iff both dimensions are the runtime sentinel @ref dynamic.
template<std::size_t Rows, std::size_t Cols>
inline constexpr bool matrix_is_dynamic = (Rows == dynamic && Cols == dynamic);

/// @internal @brief Backing-Vector extent: heap (@ref dynamic) for dynamic, `Rows*Cols` inline otherwise.
template<std::size_t Rows, std::size_t Cols>
inline constexpr std::size_t matrix_extent =
    matrix_is_dynamic<Rows, Cols> ? dynamic : Rows * Cols;

} // namespace detail

/**
 * @brief A dense, column-major numeric matrix with a compile-time or runtime shape.
 * @tparam T    Element type (an IEEE floating-point @ref Scalar).
 * @tparam Rows Row count, or @c dynamic (the default) for a runtime, heap-backed shape.
 * @tparam Cols Column count, or @c dynamic (the default).
 *
 * Storage is a single column-major `Vector<T, …>` (element `(i, j)` at `j*rows() + i`, leading
 * dimension `rows()`). The shape must be fully static (both `Rows` and `Cols` given) or fully
 * dynamic (both @c dynamic); a mixed shape fails a `static_assert`. See the file overview for the
 * layout, the static-storage stack @warning, and the elementwise surface.
 *
 * @code{.cpp}
 * miscibility::instrument::DenseMatrix<double> a(3, 3);              // 3x3 zeros, heap-backed
 * miscibility::instrument::DenseMatrix<double, 2, 2> b{{1, 2}, {3, 4}}; // fixed 2x2, inline
 * b(0, 1) = 5.0;                                                    // row 0, column 1
 * b *= 2.0;                                                         // elementwise scale
 * @endcode
 *
 * @warning A static shape stores `Rows*Cols` elements inline; keep it small to avoid stack
 * exhaustion (see the file-level @warning).
 */
template<Scalar T, std::size_t Rows = dynamic, std::size_t Cols = dynamic>
class DenseMatrix {
    static_assert((Rows == dynamic) == (Cols == dynamic),
                  "DenseMatrix must be either fully static (both Rows and Cols given) "
                  "or fully dynamic (neither). Mixed shapes are not supported.");

    static constexpr std::size_t extent = detail::matrix_extent<Rows, Cols>;
    using storage = Vector<T, extent>;

public:
    using value_type = T;          ///< Element type.
    using size_type = std::size_t; ///< Dimension / index type.

    /// @brief True iff this is the runtime-shape (heap-backed) specialization.
    static constexpr bool is_dynamic = detail::matrix_is_dynamic<Rows, Cols>;

    // -- construction ---------------------------------------------------------

    /// @brief Default: a zero matrix of the static shape, or an empty 0x0 dynamic matrix.
    DenseMatrix()
    {
        if constexpr (!is_dynamic) {
            rows_ = Rows; // data_ is already a zero-filled inline Vector of Rows*Cols
            cols_ = Cols;
        }
    }

    /**
     * @brief Construct a runtime-shape, zero-filled matrix. Dynamic shape only.
     * @param rows Row count.
     * @param cols Column count.
     */
    DenseMatrix(size_type rows, size_type cols)
        requires is_dynamic
        : data_(rows * cols), rows_(rows), cols_(cols)
    {
    }

    /**
     * @brief Construct a runtime-shape matrix with every element set to @p value. Dynamic only.
     * @param rows  Row count.
     * @param cols  Column count.
     * @param value Value written to each element.
     */
    DenseMatrix(size_type rows, size_type cols, T value)
        requires is_dynamic
        : data_(rows * cols, value), rows_(rows), cols_(cols)
    {
    }

    /**
     * @brief Construct from a row-major nested braced list (natural reading order).
     * @param rows Rows of the matrix, each an inner braced list of equal length. For a static
     *             shape the nested list must be exactly `Rows`x`Cols`; for a dynamic shape the
     *             shape is inferred from the list.
     * @throws std::invalid_argument if the inner rows differ in length (ragged), or -- for a
     *         static shape -- if the nested-list shape is not `Rows`x`Cols`.
     *
     * The rows are read in natural order and transposed into the column-major buffer.
     * @code{.cpp}
     * miscibility::instrument::DenseMatrix<double, 2, 3> m{{1, 2, 3}, {4, 5, 6}};
     * // m(0,0)==1, m(1,2)==6; raw data() is column-major {1,4, 2,5, 3,6}.
     * @endcode
     */
    DenseMatrix(std::initializer_list<std::initializer_list<T>> rows)
    {
        const size_type nrows = rows.size();
        const size_type ncols = (nrows == 0) ? 0 : rows.begin()->size();
        for (const auto& row : rows) {
            if (row.size() != ncols) {
                throw std::invalid_argument{
                    "miscibility::instrument::DenseMatrix ragged initializer"};
            }
        }
        if constexpr (!is_dynamic) {
            if (nrows != Rows || ncols != Cols) {
                throw std::invalid_argument{"miscibility::instrument::DenseMatrix size mismatch"};
            }
            rows_ = Rows;
            cols_ = Cols;
        } else {
            data_ = storage(nrows * ncols);
            rows_ = nrows;
            cols_ = ncols;
        }
        // Transpose the row-major braced list into the column-major buffer.
        size_type i = 0;
        for (const auto& row : rows) {
            size_type j = 0;
            for (const T& v : row) {
                data_[(j * rows_) + i] = v;
                ++j;
            }
            ++i;
        }
    }

    // -- dimensions -----------------------------------------------------------

    [[nodiscard]] size_type rows() const noexcept { return rows_; }    ///< Logical row count.
    [[nodiscard]] size_type columns() const noexcept { return cols_; } ///< Logical column count.
    /// @brief Logical element count, `rows() * columns()`.
    [[nodiscard]] size_type size() const noexcept { return rows_ * cols_; }
    [[nodiscard]] bool empty() const noexcept { return size() == 0; } ///< True iff `size() == 0`.

    // -- element access (column-major: element (i, j) lives at j*rows() + i) ---

    /// @brief Unchecked access to element `(i, j)`. @param i Row in `[0, rows())`. @param j Column in `[0, columns())`. @return Reference to the element.
    [[nodiscard]] T& operator()(size_type i, size_type j) noexcept
    {
        return data_[(j * rows_) + i];
    }
    /// @brief Unchecked const access to element `(i, j)`. @param i Row index. @param j Column index. @return Const reference to the element.
    [[nodiscard]] const T& operator()(size_type i, size_type j) const noexcept
    {
        return data_[(j * rows_) + i];
    }

    /**
     * @brief Bounds-checked access to element `(i, j)`.
     * @param i Row index.
     * @param j Column index.
     * @return Reference to the element.
     * @throws std::out_of_range if `i >= rows()` or `j >= columns()`.
     */
    [[nodiscard]] T& at(size_type i, size_type j)
    {
        if (i >= rows_ || j >= cols_) {
            throw std::out_of_range{"miscibility::instrument::DenseMatrix::at"};
        }
        return data_[(j * rows_) + i];
    }
    /**
     * @brief Bounds-checked const access to element `(i, j)`.
     * @param i Row index.
     * @param j Column index.
     * @return Const reference to the element.
     * @throws std::out_of_range if `i >= rows()` or `j >= columns()`.
     */
    [[nodiscard]] const T& at(size_type i, size_type j) const
    {
        if (i >= rows_ || j >= cols_) {
            throw std::out_of_range{"miscibility::instrument::DenseMatrix::at"};
        }
        return data_[(j * rows_) + i];
    }

    /// @brief Pointer to the contiguous column-major buffer (leading dimension `rows()`). @return Buffer pointer.
    [[nodiscard]] T* data() noexcept { return data_.data(); }
    /// @brief Const pointer to the contiguous column-major buffer. @return Const buffer pointer.
    [[nodiscard]] const T* data() const noexcept { return data_.data(); }

    /// @brief The backing `Vector` (logical length `size()`); the seam for the elementwise surface. @return Reference to the backing Vector.
    [[nodiscard]] storage& as_vector() noexcept { return data_; }
    /// @brief The backing `Vector` (const). @return Const reference to the backing Vector.
    [[nodiscard]] const storage& as_vector() const noexcept { return data_; }

    // -- swap -----------------------------------------------------------------

    /// @brief Exchange contents (buffer and shape) with @p other. @param other Matrix to swap with.
    void swap(DenseMatrix& other) noexcept
    {
        using std::swap;
        swap(data_, other.data_);
        swap(rows_, other.rows_);
        swap(cols_, other.cols_);
    }
    /// @brief ADL swap: exchange the contents of @p a and @p b.
    friend void swap(DenseMatrix& a, DenseMatrix& b) noexcept { a.swap(b); }

    // -- elementwise surface (delegates to the backing Vector) ----------------

    /// @brief In place scalar scaling `A <- a*A`. @param a Scale factor. @return `*this`, for chaining.
    DenseMatrix& scale(T a)
    {
        data_.scale(a);
        return *this;
    }
    /// @brief `A <- a*A`. @param a Scale factor. @return `*this`.
    DenseMatrix& operator*=(T a) { return scale(a); }
    /// @brief `A <- (1/a)*A`. @param a Divisor. @return `*this`.
    DenseMatrix& operator/=(T a)
    {
        data_ /= a;
        return *this;
    }

    /**
     * @brief Scaled accumulate `this <- this + a*x` (elementwise).
     * @param a Scale factor applied to @p x.
     * @param x Operand of the same shape as `*this`.
     * @return `*this`.
     * @throws std::invalid_argument if @p x has a different shape (`rows()`/`columns()`).
     */
    DenseMatrix& add_scaled(T a, const DenseMatrix& x)
    {
        check_same_shape(x);
        data_.add_scaled(a, x.data_);
        return *this;
    }
    /// @brief `this <- this + x` (elementwise). @param x Same-shape operand. @return `*this`. @throws std::invalid_argument on a shape mismatch.
    DenseMatrix& operator+=(const DenseMatrix& x) { return add_scaled(T(1), x); }
    /// @brief `this <- this - x` (elementwise). @param x Same-shape operand. @return `*this`. @throws std::invalid_argument on a shape mismatch.
    DenseMatrix& operator-=(const DenseMatrix& x) { return add_scaled(T(-1), x); }

    /**
     * @brief In place Hadamard (elementwise) product `this_ij <- this_ij * x_ij`.
     * @param x Operand of the same shape as `*this`.
     * @return `*this`.
     * @throws std::invalid_argument if @p x has a different shape.
     */
    DenseMatrix& elementwise_product(const DenseMatrix& x)
    {
        check_same_shape(x);
        data_.elementwise_product(x.data_);
        return *this;
    }
    /**
     * @brief In place elementwise quotient `this_ij <- this_ij / x_ij`.
     * @param x Operand of the same shape as `*this`.
     * @return `*this`.
     * @throws std::invalid_argument if @p x has a different shape.
     */
    DenseMatrix& elementwise_quotient(const DenseMatrix& x)
    {
        check_same_shape(x);
        data_.elementwise_quotient(x.data_);
        return *this;
    }

    /// @brief In place absolute value `this_ij <- |this_ij|`. @return `*this`, for chaining.
    DenseMatrix& abs()
    {
        data_.abs();
        return *this;
    }
    /// @brief In place square root `this_ij <- sqrt(this_ij)`. @return `*this`. @note A negative element yields NaN (no exception).
    DenseMatrix& sqrt()
    {
        data_.sqrt();
        return *this;
    }
    /// @brief In place natural exponential `this_ij <- exp(this_ij)`. @return `*this`.
    DenseMatrix& exp()
    {
        data_.exp();
        return *this;
    }
    /// @brief In place base-2 exponential `this_ij <- 2^this_ij`. @return `*this`.
    DenseMatrix& exp2()
    {
        data_.exp2();
        return *this;
    }
    /// @brief In place `this_ij <- exp(this_ij) - 1`, accurate near zero. @return `*this`.
    DenseMatrix& expm1()
    {
        data_.expm1();
        return *this;
    }
    /// @brief In place natural logarithm `this_ij <- log(this_ij)`. @return `*this`. @note Defined for positive elements; zero yields -inf, negatives NaN.
    DenseMatrix& log()
    {
        data_.log();
        return *this;
    }
    /// @brief In place base-2 logarithm `this_ij <- log2(this_ij)`. @return `*this`. @note Defined for positive elements.
    DenseMatrix& log2()
    {
        data_.log2();
        return *this;
    }
    /// @brief In place base-10 logarithm `this_ij <- log10(this_ij)`. @return `*this`. @note Defined for positive elements.
    DenseMatrix& log10()
    {
        data_.log10();
        return *this;
    }
    /// @brief In place `this_ij <- log(1 + this_ij)`, accurate near zero. @return `*this`. @note Defined for elements > -1.
    DenseMatrix& log1p()
    {
        data_.log1p();
        return *this;
    }
    /// @brief In place sine (radians) `this_ij <- sin(this_ij)`. @return `*this`.
    DenseMatrix& sin()
    {
        data_.sin();
        return *this;
    }
    /// @brief In place cosine (radians) `this_ij <- cos(this_ij)`. @return `*this`.
    DenseMatrix& cos()
    {
        data_.cos();
        return *this;
    }
    /// @brief In place hyperbolic sine `this_ij <- sinh(this_ij)`. @return `*this`.
    DenseMatrix& sinh()
    {
        data_.sinh();
        return *this;
    }
    /// @brief In place hyperbolic tangent `this_ij <- tanh(this_ij)`. @return `*this`.
    DenseMatrix& tanh()
    {
        data_.tanh();
        return *this;
    }
    /// @brief In place arc sine (radians) `this_ij <- asin(this_ij)`. @return `*this`. @note Defined for elements in [-1, 1].
    DenseMatrix& asin()
    {
        data_.asin();
        return *this;
    }
    /// @brief In place arc cosine (radians) `this_ij <- acos(this_ij)`. @return `*this`. @note Defined for elements in [-1, 1].
    DenseMatrix& acos()
    {
        data_.acos();
        return *this;
    }
    /// @brief In place inverse hyperbolic sine `this_ij <- asinh(this_ij)`. @return `*this`.
    DenseMatrix& asinh()
    {
        data_.asinh();
        return *this;
    }
    /// @brief In place inverse hyperbolic cosine `this_ij <- acosh(this_ij)`. @return `*this`. @note Defined for elements >= 1.
    DenseMatrix& acosh()
    {
        data_.acosh();
        return *this;
    }
    /// @brief In place arc tangent (radians) `this_ij <- atan(this_ij)`. @return `*this`.
    DenseMatrix& atan()
    {
        data_.atan();
        return *this;
    }
    /// @brief In place inverse hyperbolic tangent `this_ij <- atanh(this_ij)`. @return `*this`. @note Defined for elements in (-1, 1).
    DenseMatrix& atanh()
    {
        data_.atanh();
        return *this;
    }

private:
    /// @brief Throw if @p x differs from this matrix's logical shape.
    void check_same_shape(const DenseMatrix& x) const
    {
        if (rows_ != x.rows_ || cols_ != x.cols_) {
            throw std::invalid_argument{"miscibility::instrument::DenseMatrix shape mismatch"};
        }
    }

    template<Scalar, std::size_t> friend class Vector;

    storage data_{};          ///< Column-major buffer (one contiguous backing Vector).
    size_type rows_{0};       ///< Logical row count.
    size_type cols_{0};       ///< Logical column count.
};

} // namespace miscibility::instrument
