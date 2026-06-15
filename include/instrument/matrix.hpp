#pragma once

#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <stdexcept>
#include <utility>

#include <cblas.h>

#include "instrument/vector.hpp"

namespace miscibility::instrument {

// Human-readable wrapper over the CBLAS transpose flag. Real Scalar types only,
// so no conjugate variant is needed.
enum class Transpose : std::uint8_t { None, Transposed };

namespace detail {

/// @internal Map the human-readable Transpose enum onto the CBLAS flag.
[[nodiscard]] inline CBLAS_TRANSPOSE to_cblas(Transpose t) noexcept
{
    return t == Transpose::Transposed ? CblasTrans : CblasNoTrans;
}

/// @internal True iff both dimensions are the runtime sentinel.
template<std::size_t Rows, std::size_t Cols>
inline constexpr bool matrix_is_dynamic = (Rows == dynamic && Cols == dynamic);

/// @internal Backing-Vector extent: heap for dynamic, Rows*Cols inline otherwise.
template<std::size_t Rows, std::size_t Cols>
inline constexpr std::size_t matrix_extent =
    matrix_is_dynamic<Rows, Cols> ? dynamic : Rows * Cols;

} // namespace detail

template<Scalar T, std::size_t Rows = dynamic, std::size_t Cols = dynamic>
class DenseMatrix {
    static_assert((Rows == dynamic) == (Cols == dynamic),
                  "DenseMatrix must be either fully static (both Rows and Cols given) "
                  "or fully dynamic (neither). Mixed shapes are not supported.");

    static constexpr std::size_t extent = detail::matrix_extent<Rows, Cols>;
    using storage = Vector<T, extent>;

public:
    using value_type = T;
    using size_type = std::size_t;

    static constexpr bool is_dynamic = detail::matrix_is_dynamic<Rows, Cols>;

    // -- construction ---------------------------------------------------------

    DenseMatrix()
    {
        if constexpr (!is_dynamic) {
            rows_ = Rows; // data_ is already a zero-filled inline Vector of Rows*Cols
            cols_ = Cols;
        }
    }

    DenseMatrix(size_type rows, size_type cols)
        requires is_dynamic
        : data_(rows * cols), rows_(rows), cols_(cols)
    {
    }

    DenseMatrix(size_type rows, size_type cols, T value)
        requires is_dynamic
        : data_(rows * cols, value), rows_(rows), cols_(cols)
    {
    }

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

    [[nodiscard]] size_type rows() const noexcept { return rows_; }
    [[nodiscard]] size_type columns() const noexcept { return cols_; }
    [[nodiscard]] size_type size() const noexcept { return rows_ * cols_; }
    [[nodiscard]] bool empty() const noexcept { return size() == 0; }

    // -- element access (column-major: element (i, j) lives at j*rows() + i) ---

    [[nodiscard]] T& operator()(size_type i, size_type j) noexcept
    {
        return data_[(j * rows_) + i];
    }
    [[nodiscard]] const T& operator()(size_type i, size_type j) const noexcept
    {
        return data_[(j * rows_) + i];
    }

    [[nodiscard]] T& at(size_type i, size_type j)
    {
        if (i >= rows_ || j >= cols_) {
            throw std::out_of_range{"miscibility::instrument::DenseMatrix::at"};
        }
        return data_[(j * rows_) + i];
    }
    [[nodiscard]] const T& at(size_type i, size_type j) const
    {
        if (i >= rows_ || j >= cols_) {
            throw std::out_of_range{"miscibility::instrument::DenseMatrix::at"};
        }
        return data_[(j * rows_) + i];
    }

    [[nodiscard]] T* data() noexcept { return data_.data(); }
    [[nodiscard]] const T* data() const noexcept { return data_.data(); }

    [[nodiscard]] storage& as_vector() noexcept { return data_; }
    [[nodiscard]] const storage& as_vector() const noexcept { return data_; }

    // -- swap -----------------------------------------------------------------

    void swap(DenseMatrix& other) noexcept
    {
        using std::swap;
        swap(data_, other.data_);
        swap(rows_, other.rows_);
        swap(cols_, other.cols_);
    }
    friend void swap(DenseMatrix& a, DenseMatrix& b) noexcept { a.swap(b); }

    // -- elementwise surface (delegates to the backing Vector) ----------------

    DenseMatrix& scale(T a)
    {
        data_.scale(a);
        return *this;
    }
    DenseMatrix& operator*=(T a) { return scale(a); }
    DenseMatrix& operator/=(T a)
    {
        data_ /= a;
        return *this;
    }

    DenseMatrix& add_scaled(T a, const DenseMatrix& x)
    {
        check_same_shape(x);
        data_.add_scaled(a, x.data_);
        return *this;
    }
    DenseMatrix& operator+=(const DenseMatrix& x) { return add_scaled(T(1), x); }
    DenseMatrix& operator-=(const DenseMatrix& x) { return add_scaled(T(-1), x); }

    DenseMatrix& elementwise_product(const DenseMatrix& x)
    {
        check_same_shape(x);
        data_.elementwise_product(x.data_);
        return *this;
    }
    DenseMatrix& elementwise_quotient(const DenseMatrix& x)
    {
        check_same_shape(x);
        data_.elementwise_quotient(x.data_);
        return *this;
    }

    DenseMatrix& abs()
    {
        data_.abs();
        return *this;
    }
    DenseMatrix& sqrt()
    {
        data_.sqrt();
        return *this;
    }
    DenseMatrix& exp()
    {
        data_.exp();
        return *this;
    }
    DenseMatrix& exp2()
    {
        data_.exp2();
        return *this;
    }
    DenseMatrix& expm1()
    {
        data_.expm1();
        return *this;
    }
    DenseMatrix& log()
    {
        data_.log();
        return *this;
    }
    DenseMatrix& log2()
    {
        data_.log2();
        return *this;
    }
    DenseMatrix& log10()
    {
        data_.log10();
        return *this;
    }
    DenseMatrix& log1p()
    {
        data_.log1p();
        return *this;
    }
    DenseMatrix& sin()
    {
        data_.sin();
        return *this;
    }
    DenseMatrix& cos()
    {
        data_.cos();
        return *this;
    }
    DenseMatrix& sinh()
    {
        data_.sinh();
        return *this;
    }
    DenseMatrix& tanh()
    {
        data_.tanh();
        return *this;
    }
    DenseMatrix& asin()
    {
        data_.asin();
        return *this;
    }
    DenseMatrix& acos()
    {
        data_.acos();
        return *this;
    }
    DenseMatrix& asinh()
    {
        data_.asinh();
        return *this;
    }
    DenseMatrix& acosh()
    {
        data_.acosh();
        return *this;
    }
    DenseMatrix& atan()
    {
        data_.atan();
        return *this;
    }
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

    storage data_{};
    size_type rows_{0};
    size_type cols_{0};
};

} // namespace miscibility::instrument
