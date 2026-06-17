#pragma once

#include "instrument/matrix.hpp"
#include "instrument/vector.hpp"

#include <cstddef>
#include <initializer_list>
#include <stdexcept>

namespace miscibility::instrument {

template<Scalar T, std::size_t N = dynamic> class DiagonalMatrix {
public:
    using value_type = T;
    using size_type = std::size_t;

    static constexpr bool is_dynamic = (N == dynamic);

    // -- construction ---------------------------------------------------------

    DiagonalMatrix() { throw std::runtime_error{"not implemented"}; }

    explicit DiagonalMatrix(size_type n)
        requires is_dynamic
    {
        (void)n;
        throw std::runtime_error{"not implemented"};
    }

    DiagonalMatrix(std::initializer_list<T> diag)
    {
        (void)diag;
        throw std::runtime_error{"not implemented"};
    }

    explicit DiagonalMatrix(Vector<T, N> diag)
    {
        (void)diag;
        throw std::runtime_error{"not implemented"};
    }

    // -- dimensions / access --------------------------------------------------

    [[nodiscard]] size_type rows() const noexcept { return diag_.size(); }
    [[nodiscard]] size_type columns() const noexcept { return diag_.size(); }
    [[nodiscard]] size_type size() const noexcept { return diag_.size(); }
    [[nodiscard]] bool empty() const noexcept { return diag_.size() == 0; }

    [[nodiscard]] const T& operator[](size_type i) const noexcept { return diag_[i]; }

    [[nodiscard]] const T& at(size_type i) const
    {
        (void)i;
        throw std::runtime_error{"not implemented"};
    }

    [[nodiscard]] const Vector<T, N>& diagonal() const noexcept { return diag_; }
    [[nodiscard]] const T* data() const noexcept { return diag_.data(); }

    // -- mutation -------------------------------------------------------------

    void set(size_type i, T value)
    {
        (void)i;
        (void)value;
        throw std::runtime_error{"not implemented"};
    }

    // -- singularity ----------------------------------------------------------

    [[nodiscard]] bool singular() const noexcept { return singular_; }

    // -- matrix-vector product ------------------------------------------------

    void multiply_into(const Vector<T>& x, Vector<T>& y, T alpha = T(1), T beta = T(0),
                       Transpose op = Transpose::None) const
    {
        (void)x;
        (void)y;
        (void)alpha;
        (void)beta;
        (void)op;
        throw std::runtime_error{"not implemented"};
    }

    [[nodiscard]] Vector<T> multiply(const Vector<T>& x, Transpose op = Transpose::None) const
    {
        (void)x;
        (void)op;
        throw std::runtime_error{"not implemented"};
    }

    [[nodiscard]] Vector<T> operator*(const Vector<T>& x) const
    {
        (void)x;
        throw std::runtime_error{"not implemented"};
    }

    // -- matrix-matrix product (diagonal-times-dense row scaling) -------------

    void multiply_into(const DenseMatrix<T>& b, DenseMatrix<T>& c, T alpha = T(1), T beta = T(0),
                       Transpose opB = Transpose::None) const
    {
        (void)b;
        (void)c;
        (void)alpha;
        (void)beta;
        (void)opB;
        throw std::runtime_error{"not implemented"};
    }

    [[nodiscard]] DenseMatrix<T> multiply(const DenseMatrix<T>& b, Transpose opB = Transpose::None) const
    {
        (void)b;
        (void)opB;
        throw std::runtime_error{"not implemented"};
    }

    [[nodiscard]] DenseMatrix<T> operator*(const DenseMatrix<T>& b) const
    {
        (void)b;
        throw std::runtime_error{"not implemented"};
    }

    // -- solve ----------------------------------------------------------------

    void solve_into(const Vector<T>& b, Vector<T>& x) const
    {
        (void)b;
        (void)x;
        throw std::runtime_error{"not implemented"};
    }

    [[nodiscard]] Vector<T> solve(const Vector<T>& b) const
    {
        (void)b;
        throw std::runtime_error{"not implemented"};
    }

private:
    Vector<T, N> diag_{};
    bool singular_{false};
};

} // namespace miscibility::instrument
