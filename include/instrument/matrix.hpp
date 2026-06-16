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
 * @par Matrix products
 * multiply() / multiply_into() / `operator*` wrap CBLAS gemv (matrix-vector) and gemm
 * (matrix-matrix): `y <- alpha*op(A)*x + beta*y` and `C <- alpha*opA(A)*opB(B) + beta*C`. The
 * `multiply` form allocates a fresh result; the `multiply_into` form fills a caller-supplied
 * destination (with `alpha`/`beta` for the scale/add variants). All are `const` -- a product never
 * mutates the matrix.
 *
 * @par Factorizations
 * Factorizations are *separate* types (see @ref LUFactorization), each constructed from a
 * DenseMatrix (by copy or by move) and exposing `solve`. `A.lu()` is the convenience entry point.
 *
 * @par Header-only
 * C++23. Backs onto CBLAS and the LAPACK Fortran interface (OpenBLAS).
 */

#pragma once

#include "instrument/vector.hpp"

#include <algorithm>
#include <cblas.h>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <stdexcept>
#include <utility>
#include <vector>

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
inline constexpr std::size_t matrix_extent = matrix_is_dynamic<Rows, Cols> ? dynamic : Rows * Cols;

/// @internal @brief Dispatch `cblas_?gemv` on @p T: `y <- alpha*op(A)*x + beta*y` (column-major).
template<Scalar T>
void gemv(CBLAS_TRANSPOSE trans, int m, int n, T alpha, const T* a, int lda, const T* x, T beta, T* y) noexcept
{
    if constexpr (std::same_as<T, float>) {
        cblas_sgemv(CblasColMajor, trans, m, n, alpha, a, lda, x, 1, beta, y, 1);
    }
    else {
        cblas_dgemv(CblasColMajor, trans, m, n, alpha, a, lda, x, 1, beta, y, 1);
    }
}

/// @internal @brief Dispatch `cblas_?gemm` on @p T: `C <- alpha*opA(A)*opB(B) + beta*C` (column-major).
template<Scalar T>
void gemm(CBLAS_TRANSPOSE transa, CBLAS_TRANSPOSE transb, int m, int n, int k, T alpha, const T* a, int lda, const T* b,
          int ldb, T beta, T* c, int ldc) noexcept
{
    if constexpr (std::same_as<T, float>) {
        cblas_sgemm(CblasColMajor, transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
    }
    else {
        cblas_dgemm(CblasColMajor, transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
    }
}

} // namespace detail

template<Scalar T> class LUFactorization; // defined below; A.lu() is the entry point

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
template<Scalar T, std::size_t Rows = dynamic, std::size_t Cols = dynamic> class DenseMatrix {
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
                throw std::invalid_argument{"miscibility::instrument::DenseMatrix ragged initializer"};
            }
        }
        if constexpr (!is_dynamic) {
            if (nrows != Rows || ncols != Cols) {
                throw std::invalid_argument{"miscibility::instrument::DenseMatrix size mismatch"};
            }
            rows_ = Rows;
            cols_ = Cols;
        }
        else {
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

    /// @brief Unchecked access to element `(i, j)`. @param i Row in `[0, rows())`. @param j Column in `[0, columns())`.
    /// @return Reference to the element.
    [[nodiscard]] T& operator()(size_type i, size_type j) noexcept { return data_[(j * rows_) + i]; }
    /// @brief Unchecked const access to element `(i, j)`. @param i Row index. @param j Column index. @return Const
    /// reference to the element.
    [[nodiscard]] const T& operator()(size_type i, size_type j) const noexcept { return data_[(j * rows_) + i]; }

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

    /// @brief The backing `Vector` (logical length `size()`); the seam for the elementwise surface. @return Reference
    /// to the backing Vector.
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
    /// @brief `this <- this + x` (elementwise). @param x Same-shape operand. @return `*this`. @throws
    /// std::invalid_argument on a shape mismatch.
    DenseMatrix& operator+=(const DenseMatrix& x) { return add_scaled(T(1), x); }
    /// @brief `this <- this - x` (elementwise). @param x Same-shape operand. @return `*this`. @throws
    /// std::invalid_argument on a shape mismatch.
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
    /// @brief In place square root `this_ij <- sqrt(this_ij)`. @return `*this`. @note A negative element yields NaN (no
    /// exception).
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
    /// @brief In place natural logarithm `this_ij <- log(this_ij)`. @return `*this`. @note Defined for positive
    /// elements; zero yields -inf, negatives NaN.
    DenseMatrix& log()
    {
        data_.log();
        return *this;
    }
    /// @brief In place base-2 logarithm `this_ij <- log2(this_ij)`. @return `*this`. @note Defined for positive
    /// elements.
    DenseMatrix& log2()
    {
        data_.log2();
        return *this;
    }
    /// @brief In place base-10 logarithm `this_ij <- log10(this_ij)`. @return `*this`. @note Defined for positive
    /// elements.
    DenseMatrix& log10()
    {
        data_.log10();
        return *this;
    }
    /// @brief In place `this_ij <- log(1 + this_ij)`, accurate near zero. @return `*this`. @note Defined for elements >
    /// -1.
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
    /// @brief In place arc sine (radians) `this_ij <- asin(this_ij)`. @return `*this`. @note Defined for elements in
    /// [-1, 1].
    DenseMatrix& asin()
    {
        data_.asin();
        return *this;
    }
    /// @brief In place arc cosine (radians) `this_ij <- acos(this_ij)`. @return `*this`. @note Defined for elements in
    /// [-1, 1].
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
    /// @brief In place inverse hyperbolic cosine `this_ij <- acosh(this_ij)`. @return `*this`. @note Defined for
    /// elements >= 1.
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
    /// @brief In place inverse hyperbolic tangent `this_ij <- atanh(this_ij)`. @return `*this`. @note Defined for
    /// elements in (-1, 1).
    DenseMatrix& atanh()
    {
        data_.atanh();
        return *this;
    }

    // -- matrix-vector product (CBLAS gemv) -----------------------------------

    /**
     * @brief Matrix-vector product into a pre-allocated vector: `y <- alpha*op(A)*x + beta*y`.
     * @param x     Right operand. Length `columns()` for @c None, `rows()` for @c Transposed.
     * @param y     Destination, written in place. Length `rows()` for @c None, `columns()` for
     *              @c Transposed; must already be sized.
     * @param alpha Scale applied to the product (default 1).
     * @param beta  Scale applied to the existing @p y before accumulation (default 0).
     * @param op    Whether to use `A` (@c None) or `A^T` (@c Transposed).
     * @throws std::invalid_argument if @p x or @p y has the wrong length.
     */
    void multiply_into(const Vector<T>& x, Vector<T>& y, T alpha = T(1), T beta = T(0),
                       Transpose op = Transpose::None) const
    {
        const size_type xlen = (op == Transpose::None) ? cols_ : rows_;
        const size_type ylen = (op == Transpose::None) ? rows_ : cols_;
        if (x.size() != xlen || y.size() != ylen) {
            throw std::invalid_argument{"miscibility::instrument::DenseMatrix matrix-vector size mismatch"};
        }
        detail::gemv<T>(detail::to_cblas(op), static_cast<int>(rows_), static_cast<int>(cols_), alpha, data(),
                        static_cast<int>(rows_), x.data(), beta, y.data());
    }

    /**
     * @brief Matrix-vector product `op(A)*x`, returning a fresh vector.
     * @param x  Right operand (see multiply_into() for the length rule).
     * @param op Whether to use `A` (@c None) or `A^T` (@c Transposed).
     * @return The product, a new `Vector<T>` of length `rows()` (or `columns()` if @c Transposed).
     * @throws std::invalid_argument if @p x has the wrong length.
     */
    [[nodiscard]] Vector<T> multiply(const Vector<T>& x, Transpose op = Transpose::None) const
    {
        Vector<T> y((op == Transpose::None) ? rows_ : cols_);
        multiply_into(x, y, T(1), T(0), op);
        return y;
    }

    /// @brief `A*x` as a fresh vector. @param x Operand of length `columns()`. @return The product. @throws
    /// std::invalid_argument on a length mismatch.
    [[nodiscard]] Vector<T> operator*(const Vector<T>& x) const { return multiply(x); }

    // -- matrix-matrix product (CBLAS gemm) -----------------------------------

    /**
     * @brief Matrix-matrix product into a pre-allocated matrix: `C <- alpha*opA(A)*opB(B) + beta*C`.
     * @param b     Right operand.
     * @param c     Destination, written in place; must already be `(rows of opA(A))` x
     *              `(cols of opB(B))`.
     * @param alpha Scale applied to the product (default 1).
     * @param beta  Scale applied to the existing @p c before accumulation (default 0).
     * @param opA   Whether to use `A` (@c None) or `A^T` (@c Transposed).
     * @param opB   Whether to use `B` (@c None) or `B^T` (@c Transposed).
     * @throws std::invalid_argument if the inner dimensions disagree or @p c has the wrong shape.
     */
    void multiply_into(const DenseMatrix<T>& b, DenseMatrix<T>& c, T alpha = T(1), T beta = T(0),
                       Transpose opA = Transpose::None, Transpose opB = Transpose::None) const
    {
        const size_type m = (opA == Transpose::None) ? rows_ : cols_;
        const size_type k = (opA == Transpose::None) ? cols_ : rows_;
        const size_type kb = (opB == Transpose::None) ? b.rows() : b.columns();
        const size_type n = (opB == Transpose::None) ? b.columns() : b.rows();
        if (k != kb || c.rows() != m || c.columns() != n) {
            throw std::invalid_argument{"miscibility::instrument::DenseMatrix matrix-matrix size mismatch"};
        }
        detail::gemm<T>(detail::to_cblas(opA), detail::to_cblas(opB), static_cast<int>(m), static_cast<int>(n),
                        static_cast<int>(k), alpha, data(), static_cast<int>(rows_), b.data(),
                        static_cast<int>(b.rows()), beta, c.data(), static_cast<int>(c.rows()));
    }

    /**
     * @brief Matrix-matrix product `opA(A)*opB(B)`, returning a fresh matrix.
     * @param b   Right operand.
     * @param opA Whether to use `A` (@c None) or `A^T` (@c Transposed).
     * @param opB Whether to use `B` (@c None) or `B^T` (@c Transposed).
     * @return The product, a new `DenseMatrix<T>` of shape `(rows of opA(A))` x `(cols of opB(B))`.
     * @throws std::invalid_argument if the inner dimensions disagree.
     */
    [[nodiscard]] DenseMatrix<T> multiply(const DenseMatrix<T>& b, Transpose opA = Transpose::None,
                                          Transpose opB = Transpose::None) const
    {
        const size_type m = (opA == Transpose::None) ? rows_ : cols_;
        const size_type n = (opB == Transpose::None) ? b.columns() : b.rows();
        DenseMatrix<T> c(m, n);
        multiply_into(b, c, T(1), T(0), opA, opB);
        return c;
    }

    /// @brief `A*B` as a fresh matrix. @param b Right operand. @return The product. @throws std::invalid_argument on an
    /// inner-dimension mismatch.
    [[nodiscard]] DenseMatrix<T> operator*(const DenseMatrix<T>& b) const { return multiply(b); }

    // -- factorization entry point --------------------------------------------

    /// @brief LU-factorize a *copy* of this matrix (the matrix is preserved). @return The factorization. @throws
    /// std::invalid_argument if non-square.
    [[nodiscard]] LUFactorization<T> lu() const&;
    /// @brief LU-factorize by *moving* this matrix's buffer when it is a dynamic rvalue (the source is left empty 0x0;
    /// a static rvalue is copied). @return The factorization. @throws std::invalid_argument if non-square.
    [[nodiscard]] LUFactorization<T> lu() &&;

private:
    /// @brief Throw if @p x differs from this matrix's logical shape.
    void check_same_shape(const DenseMatrix& x) const
    {
        if (rows_ != x.rows_ || cols_ != x.cols_) {
            throw std::invalid_argument{"miscibility::instrument::DenseMatrix shape mismatch"};
        }
    }

    template<Scalar, std::size_t> friend class Vector;
    template<Scalar> friend class LUFactorization;

    storage data_{};    ///< Column-major buffer (one contiguous backing Vector).
    size_type rows_{0}; ///< Logical row count.
    size_type cols_{0}; ///< Logical column count.
};

// ---------------------------------------------------------------------------
// LAPACK factorization layer.
// ---------------------------------------------------------------------------

namespace detail {

/// @internal @brief LAPACK integer width (32-bit for the default OpenBLAS LP64 build).
using lapack_int = int;

// The LAPACK Fortran interface (provided by OpenBLAS). It is natively column-major, matching the
// DenseMatrix storage with no transpose; <lapacke.h> is deliberately not required.
extern "C" {
void sgetrf_(const lapack_int* m, const lapack_int* n, float* a, const lapack_int* lda, lapack_int* ipiv,
             lapack_int* info);
void dgetrf_(const lapack_int* m, const lapack_int* n, double* a, const lapack_int* lda, lapack_int* ipiv,
             lapack_int* info);
void sgetrs_(const char* trans, const lapack_int* n, const lapack_int* nrhs, const float* a, const lapack_int* lda,
             const lapack_int* ipiv, float* b, const lapack_int* ldb, lapack_int* info);
void dgetrs_(const char* trans, const lapack_int* n, const lapack_int* nrhs, const double* a, const lapack_int* lda,
             const lapack_int* ipiv, double* b, const lapack_int* ldb, lapack_int* info);
}

/// @internal @brief Dispatch `?getrf` on @p T: in-place LU factorization `P*A = L*U`.
template<Scalar T>
void getrf(lapack_int m, lapack_int n, T* a, lapack_int lda, lapack_int* ipiv, lapack_int* info) noexcept
{
    if constexpr (std::same_as<T, float>) {
        sgetrf_(&m, &n, a, &lda, ipiv, info);
    }
    else {
        dgetrf_(&m, &n, a, &lda, ipiv, info);
    }
}

/// @internal @brief Dispatch `?getrs` on @p T: solve from an existing LU factorization.
template<Scalar T>
void getrs(char trans, lapack_int n, lapack_int nrhs, const T* a, lapack_int lda, const lapack_int* ipiv, T* b,
           lapack_int ldb, lapack_int* info) noexcept
{
    if constexpr (std::same_as<T, float>) {
        sgetrs_(&trans, &n, &nrhs, a, &lda, ipiv, b, &ldb, info);
    }
    else {
        dgetrs_(&trans, &n, &nrhs, a, &lda, ipiv, b, &ldb, info);
    }
}

} // namespace detail

/**
 * @brief An LU factorization of a square matrix, with partial pivoting: `P*A = L*U`.
 * @tparam T Element type (an IEEE floating-point @ref Scalar).
 *
 * A factorization is a *solver*, not a matrix: it is built from a @ref DenseMatrix (taking that
 * matrix's data by copy or by move), factors once in its constructor, and thereafter only exposes
 * solve() / solve_into(). Keeping it separate from DenseMatrix means a DenseMatrix always holds
 * `A` and the two can never desynchronize. Build one with `A.lu()` (which picks copy vs. move from
 * the value category of `A`) or via the explicit constructors.
 *
 * The factorization is computed once and reused across as many right-hand sides as you like -- the
 * reason the type exists -- so the solve methods are `const`.
 *
 * @code{.cpp}
 * miscibility::instrument::DenseMatrix<double> A{{2, 1}, {1, 3}};
 * auto lu = A.lu();                 // A is preserved (lvalue -> copy)
 * auto x  = lu.solve({3.0, 5.0});   // x == {0.8, 1.4}
 * @endcode
 *
 * @note Always heap-backed, regardless of the source matrix's shape: a static source is copied
 * into a dynamic internal matrix.
 */
template<Scalar T> class LUFactorization {
public:
    using value_type = T;          ///< Element type.
    using size_type = std::size_t; ///< Dimension / index type.

    /**
     * @brief Copy @p a and factor it; the source matrix is preserved.
     * @tparam R Source row extent.
     * @tparam C Source column extent.
     * @param a The (square) matrix to factor. Any shape; a static source is copied here.
     * @throws std::invalid_argument if @p a is not square.
     * @throws std::runtime_error on an internal LAPACK argument error.
     */
    template<std::size_t R, std::size_t C>
    explicit LUFactorization(const DenseMatrix<T, R, C>& a) : lu_(DenseMatrix<T>(a.rows(), a.columns())), n_(a.rows())
    {
        require_square(a.rows(), a.columns());

        std::copy_n(a.data(), a.size(), lu_.data());
        factor();
    }

    /**
     * @brief Move @p a's buffer in and factor it; the source is left a valid empty 0x0.
     * @param a A dynamic (square) matrix; its buffer is stolen. (A static matrix has no movable
     *          buffer and is taken through the copying constructor instead.)
     * @throws std::invalid_argument if @p a is not square.
     * @throws std::runtime_error on an internal LAPACK argument error.
     */
    explicit LUFactorization(DenseMatrix<T>&& a) : lu_(std::move(a)), n_(a.rows())
    {
        require_square(a.rows(), a.columns());

        a = DenseMatrix<T>{}; // leave the moved-from source a valid empty 0x0
        factor();
    }

    /// @brief True iff the factorization hit an exactly-zero pivot (the system is singular). @return The singular flag.
    [[nodiscard]] bool singular() const noexcept { return singular_; }

    /// @brief Order `n` of the factored (square) system. @return The system order.
    [[nodiscard]] size_type order() const noexcept { return n_; }

    /**
     * @brief Solve `A*x = b` into a pre-allocated vector.
     * @param b Right-hand side, length `order()`. Not modified.
     * @param x Destination, written in place; must already be length `order()`.
     * @throws std::invalid_argument if @p b or @p x has the wrong length.
     * @throws std::runtime_error if the factorization is singular(), or on an internal LAPACK
     *         argument error.
     */
    void solve_into(const Vector<T>& b, Vector<T>& x) const
    {
        if (b.size() != n_ || x.size() != n_) {
            throw std::invalid_argument{"miscibility::instrument::LUFactorization solve size mismatch"};
        }
        if (singular_) {
            throw std::runtime_error{"miscibility::instrument::LUFactorization: singular matrix"};
        }
        std::copy_n(b.data(), n_, x.data()); // getrs overwrites its RHS with the solution
        detail::lapack_int info = 0;
        detail::getrs<T>('N', static_cast<detail::lapack_int>(n_), 1, lu_.data(), static_cast<detail::lapack_int>(n_),
                         pivots_.data(), x.data(), static_cast<detail::lapack_int>(n_), &info);
        if (info < 0) {
            throw std::runtime_error{"miscibility::instrument::LUFactorization: LAPACK argument error"};
        }
    }

    /**
     * @brief Solve `A*x = b`, returning a fresh solution vector.
     * @param b Right-hand side, length `order()`.
     * @return The solution `x`.
     * @throws std::invalid_argument if @p b has the wrong length.
     * @throws std::runtime_error if singular().
     */
    [[nodiscard]] Vector<T> solve(const Vector<T>& b) const
    {
        Vector<T> x(n_);
        solve_into(b, x);
        return x;
    }

private:
    /// @brief Throw if the shape is not square.
    static void require_square(size_type r, size_type c)
    {
        if (r != c) {
            throw std::invalid_argument{"miscibility::instrument::LUFactorization requires a square matrix"};
        }
    }

    /// @brief Run getrf in place over lu_, recording pivots and singularity.
    void factor()
    {
        singular_ = false;
        if (n_ == 0) {
            return;
        }
        pivots_.resize(n_);
        detail::lapack_int info = 0;
        detail::getrf<T>(static_cast<detail::lapack_int>(n_), static_cast<detail::lapack_int>(n_), lu_.data(),
                         static_cast<detail::lapack_int>(n_), pivots_.data(), &info);
        if (info < 0) {
            throw std::runtime_error{"miscibility::instrument::LUFactorization: LAPACK argument error"};
        }
        singular_ = (info > 0);
    }

    DenseMatrix<T> lu_{};                    ///< Packed L\\U factors (overwritten in place).
    std::vector<detail::lapack_int> pivots_; ///< Row pivots from getrf.
    bool singular_{false};                   ///< Set when getrf reports a zero pivot.
    size_type n_{0};                         ///< Order of the square system.
};

template<Scalar T, std::size_t Rows, std::size_t Cols> LUFactorization<T> DenseMatrix<T, Rows, Cols>::lu() const&
{
    return LUFactorization<T>{*this};
}

template<Scalar T, std::size_t Rows, std::size_t Cols> LUFactorization<T> DenseMatrix<T, Rows, Cols>::lu() &&
{
    return LUFactorization<T>{std::move(*this)};
}

} // namespace miscibility::instrument
