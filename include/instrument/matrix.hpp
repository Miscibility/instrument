#pragma once

#include <cstddef>
#include <initializer_list>
#include <stdexcept>

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

    DenseMatrix() { throw std::runtime_error{"not implemented"}; }

    DenseMatrix(size_type rows, size_type cols)
        requires is_dynamic
    {
        (void)rows;
        (void)cols;
        throw std::runtime_error{"not implemented"};
    }

    DenseMatrix(size_type rows, size_type cols, T value)
        requires is_dynamic
    {
        (void)rows;
        (void)cols;
        (void)value;
        throw std::runtime_error{"not implemented"};
    }

    DenseMatrix(std::initializer_list<std::initializer_list<T>> rows)
    {
        (void)rows;
        throw std::runtime_error{"not implemented"};
    }

    // -- dimensions -----------------------------------------------------------

    [[nodiscard]] size_type rows() const noexcept;
    [[nodiscard]] size_type columns() const noexcept;
    [[nodiscard]] size_type size() const noexcept;
    [[nodiscard]] bool empty() const noexcept;

    // -- element access -------------------------------------------------------

    [[nodiscard]] T& operator()(size_type i, size_type j) noexcept;
    [[nodiscard]] const T& operator()(size_type i, size_type j) const noexcept;

    [[nodiscard]] T& at(size_type i, size_type j);
    [[nodiscard]] const T& at(size_type i, size_type j) const;

    [[nodiscard]] T* data() noexcept;
    [[nodiscard]] const T* data() const noexcept;

    [[nodiscard]] storage& as_vector() noexcept;
    [[nodiscard]] const storage& as_vector() const noexcept;

    // -- swap -----------------------------------------------------------------

    void swap(DenseMatrix& other) noexcept;
    friend void swap(DenseMatrix& a, DenseMatrix& b) noexcept { a.swap(b); }

    // -- elementwise surface (delegates to the backing Vector) ----------------

    DenseMatrix& scale(T a);
    DenseMatrix& operator*=(T a);
    DenseMatrix& operator/=(T a);

    DenseMatrix& add_scaled(T a, const DenseMatrix& x);
    DenseMatrix& operator+=(const DenseMatrix& x);
    DenseMatrix& operator-=(const DenseMatrix& x);

    DenseMatrix& multiply(const DenseMatrix& x);
    DenseMatrix& divide(const DenseMatrix& x);

    DenseMatrix& abs();
    DenseMatrix& sqrt();
    DenseMatrix& exp();
    DenseMatrix& exp2();
    DenseMatrix& expm1();
    DenseMatrix& log();
    DenseMatrix& log2();
    DenseMatrix& log10();
    DenseMatrix& log1p();
    DenseMatrix& sin();
    DenseMatrix& cos();
    DenseMatrix& sinh();
    DenseMatrix& tanh();
    DenseMatrix& asin();
    DenseMatrix& acos();
    DenseMatrix& asinh();
    DenseMatrix& acosh();
    DenseMatrix& atan();
    DenseMatrix& atanh();

private:
    storage data_{};
    size_type rows_{0};
    size_type cols_{0};
};

// ---------------------------------------------------------------------------
// Stub definitions. Every member below throws "not implemented" until
// tdd-3-implement fills it in. The noexcept accessors turn that throw into
// std::terminate, but they are never reached: every test first constructs a
// DenseMatrix, and the constructors throw.
// ---------------------------------------------------------------------------

template<Scalar T, std::size_t Rows, std::size_t Cols>
auto DenseMatrix<T, Rows, Cols>::rows() const noexcept -> size_type
{
    throw std::runtime_error{"not implemented"};
}

template<Scalar T, std::size_t Rows, std::size_t Cols>
auto DenseMatrix<T, Rows, Cols>::columns() const noexcept -> size_type
{
    throw std::runtime_error{"not implemented"};
}

template<Scalar T, std::size_t Rows, std::size_t Cols>
auto DenseMatrix<T, Rows, Cols>::size() const noexcept -> size_type
{
    throw std::runtime_error{"not implemented"};
}

template<Scalar T, std::size_t Rows, std::size_t Cols>
bool DenseMatrix<T, Rows, Cols>::empty() const noexcept
{
    throw std::runtime_error{"not implemented"};
}

template<Scalar T, std::size_t Rows, std::size_t Cols>
T& DenseMatrix<T, Rows, Cols>::operator()(size_type i, size_type j) noexcept
{
    (void)i;
    (void)j;
    throw std::runtime_error{"not implemented"};
}

template<Scalar T, std::size_t Rows, std::size_t Cols>
const T& DenseMatrix<T, Rows, Cols>::operator()(size_type i, size_type j) const noexcept
{
    (void)i;
    (void)j;
    throw std::runtime_error{"not implemented"};
}

template<Scalar T, std::size_t Rows, std::size_t Cols>
T& DenseMatrix<T, Rows, Cols>::at(size_type i, size_type j)
{
    (void)i;
    (void)j;
    throw std::runtime_error{"not implemented"};
}

template<Scalar T, std::size_t Rows, std::size_t Cols>
const T& DenseMatrix<T, Rows, Cols>::at(size_type i, size_type j) const
{
    (void)i;
    (void)j;
    throw std::runtime_error{"not implemented"};
}

template<Scalar T, std::size_t Rows, std::size_t Cols>
T* DenseMatrix<T, Rows, Cols>::data() noexcept
{
    throw std::runtime_error{"not implemented"};
}

template<Scalar T, std::size_t Rows, std::size_t Cols>
const T* DenseMatrix<T, Rows, Cols>::data() const noexcept
{
    throw std::runtime_error{"not implemented"};
}

template<Scalar T, std::size_t Rows, std::size_t Cols>
auto DenseMatrix<T, Rows, Cols>::as_vector() noexcept -> storage&
{
    throw std::runtime_error{"not implemented"};
}

template<Scalar T, std::size_t Rows, std::size_t Cols>
auto DenseMatrix<T, Rows, Cols>::as_vector() const noexcept -> const storage&
{
    throw std::runtime_error{"not implemented"};
}

template<Scalar T, std::size_t Rows, std::size_t Cols>
void DenseMatrix<T, Rows, Cols>::swap(DenseMatrix& other) noexcept
{
    (void)other;
    throw std::runtime_error{"not implemented"};
}

template<Scalar T, std::size_t Rows, std::size_t Cols>
DenseMatrix<T, Rows, Cols>& DenseMatrix<T, Rows, Cols>::scale(T a)
{
    (void)a;
    throw std::runtime_error{"not implemented"};
}

template<Scalar T, std::size_t Rows, std::size_t Cols>
DenseMatrix<T, Rows, Cols>& DenseMatrix<T, Rows, Cols>::operator*=(T a)
{
    (void)a;
    throw std::runtime_error{"not implemented"};
}

template<Scalar T, std::size_t Rows, std::size_t Cols>
DenseMatrix<T, Rows, Cols>& DenseMatrix<T, Rows, Cols>::operator/=(T a)
{
    (void)a;
    throw std::runtime_error{"not implemented"};
}

template<Scalar T, std::size_t Rows, std::size_t Cols>
DenseMatrix<T, Rows, Cols>& DenseMatrix<T, Rows, Cols>::add_scaled(T a, const DenseMatrix& x)
{
    (void)a;
    (void)x;
    throw std::runtime_error{"not implemented"};
}

template<Scalar T, std::size_t Rows, std::size_t Cols>
DenseMatrix<T, Rows, Cols>& DenseMatrix<T, Rows, Cols>::operator+=(const DenseMatrix& x)
{
    (void)x;
    throw std::runtime_error{"not implemented"};
}

template<Scalar T, std::size_t Rows, std::size_t Cols>
DenseMatrix<T, Rows, Cols>& DenseMatrix<T, Rows, Cols>::operator-=(const DenseMatrix& x)
{
    (void)x;
    throw std::runtime_error{"not implemented"};
}

template<Scalar T, std::size_t Rows, std::size_t Cols>
DenseMatrix<T, Rows, Cols>& DenseMatrix<T, Rows, Cols>::multiply(const DenseMatrix& x)
{
    (void)x;
    throw std::runtime_error{"not implemented"};
}

template<Scalar T, std::size_t Rows, std::size_t Cols>
DenseMatrix<T, Rows, Cols>& DenseMatrix<T, Rows, Cols>::divide(const DenseMatrix& x)
{
    (void)x;
    throw std::runtime_error{"not implemented"};
}

// The componentwise transforms all share the same throwing stub body.
#define MISCIBILITY_MATRIX_TRANSFORM_STUB(name)                                                    \
    template<Scalar T, std::size_t Rows, std::size_t Cols>                                          \
    DenseMatrix<T, Rows, Cols>& DenseMatrix<T, Rows, Cols>::name()                                  \
    {                                                                                              \
        throw std::runtime_error{"not implemented"};                                               \
    }

MISCIBILITY_MATRIX_TRANSFORM_STUB(abs)
MISCIBILITY_MATRIX_TRANSFORM_STUB(sqrt)
MISCIBILITY_MATRIX_TRANSFORM_STUB(exp)
MISCIBILITY_MATRIX_TRANSFORM_STUB(exp2)
MISCIBILITY_MATRIX_TRANSFORM_STUB(expm1)
MISCIBILITY_MATRIX_TRANSFORM_STUB(log)
MISCIBILITY_MATRIX_TRANSFORM_STUB(log2)
MISCIBILITY_MATRIX_TRANSFORM_STUB(log10)
MISCIBILITY_MATRIX_TRANSFORM_STUB(log1p)
MISCIBILITY_MATRIX_TRANSFORM_STUB(sin)
MISCIBILITY_MATRIX_TRANSFORM_STUB(cos)
MISCIBILITY_MATRIX_TRANSFORM_STUB(sinh)
MISCIBILITY_MATRIX_TRANSFORM_STUB(tanh)
MISCIBILITY_MATRIX_TRANSFORM_STUB(asin)
MISCIBILITY_MATRIX_TRANSFORM_STUB(acos)
MISCIBILITY_MATRIX_TRANSFORM_STUB(asinh)
MISCIBILITY_MATRIX_TRANSFORM_STUB(acosh)
MISCIBILITY_MATRIX_TRANSFORM_STUB(atan)
MISCIBILITY_MATRIX_TRANSFORM_STUB(atanh)

#undef MISCIBILITY_MATRIX_TRANSFORM_STUB

} // namespace miscibility::instrument
