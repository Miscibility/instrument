#pragma once

#include "instrument/matrix.hpp"
#include "instrument/vector.hpp"

#include <cstddef>
#include <initializer_list>
#include <stdexcept>
#include <utility>

namespace miscibility::instrument {

template<Scalar T, std::size_t N = dynamic> class DiagonalMatrix {
public:
    using value_type = T;
    using size_type = std::size_t;

    static constexpr bool is_dynamic = (N == dynamic);

    // -- construction ---------------------------------------------------------

    DiagonalMatrix() { recompute_singular(); }

    explicit DiagonalMatrix(size_type n)
        requires is_dynamic
        : diag_(n)
    {
        recompute_singular();
    }

    DiagonalMatrix(std::initializer_list<T> diag) : diag_(diag) { recompute_singular(); }

    explicit DiagonalMatrix(Vector<T, N> diag) : diag_(std::move(diag)) { recompute_singular(); }

    // -- dimensions / access --------------------------------------------------

    [[nodiscard]] size_type rows() const noexcept { return diag_.size(); }
    [[nodiscard]] size_type columns() const noexcept { return diag_.size(); }
    [[nodiscard]] size_type size() const noexcept { return diag_.size(); }
    [[nodiscard]] bool empty() const noexcept { return diag_.size() == 0; }

    [[nodiscard]] const T& operator[](size_type i) const noexcept { return diag_[i]; }

    [[nodiscard]] const T& at(size_type i) const { return diag_.at(i); }

    [[nodiscard]] const Vector<T, N>& diagonal() const noexcept { return diag_; }
    [[nodiscard]] const T* data() const noexcept { return diag_.data(); }

    // -- mutation -------------------------------------------------------------

    void set(size_type i, T value)
    {
        if (i >= diag_.size()) {
            throw std::out_of_range{"miscibility::instrument::DiagonalMatrix::set"};
        }
        const T old = diag_[i];
        if (old == T(0) && value != T(0)) {
            --zero_count_;
        }
        else if (old != T(0) && value == T(0)) {
            ++zero_count_;
        }
        diag_[i] = value;
        singular_ = (zero_count_ > 0);
    }

    // -- singularity ----------------------------------------------------------

    [[nodiscard]] bool singular() const noexcept { return singular_; }

    // -- matrix-vector product ------------------------------------------------

    void multiply_into(const Vector<T>& x, Vector<T>& y, T alpha = T(1), T beta = T(0),
                       Transpose op = Transpose::None) const
    {
        (void)op; // a diagonal is its own transpose; op only affects the (square) length check
        const size_type n = diag_.size();
        if (x.size() != n || y.size() != n) {
            throw std::invalid_argument{"miscibility::instrument::DiagonalMatrix matrix-vector size mismatch"};
        }
        // TODO: try and remove this allocation
        Vector<T> p(n);
        p.copy(diag_);
        p.elementwise_product(x); // p_i = d_i * x_i
        y.scale(beta);
        y.add_scaled(alpha, p); // y_i <- alpha*d_i*x_i + beta*y_i
    }

    [[nodiscard]] Vector<T> multiply(const Vector<T>& x, Transpose op = Transpose::None) const
    {
        Vector<T> y(diag_.size());
        multiply_into(x, y, T(1), T(0), op);
        return y;
    }

    [[nodiscard]] Vector<T> operator*(const Vector<T>& x) const { return multiply(x); }

    // -- matrix-matrix product (diagonal-times-dense row scaling) -------------

    void multiply_into(const DenseMatrix<T>& b, DenseMatrix<T>& c, T alpha = T(1), T beta = T(0),
                       Transpose opB = Transpose::None) const
    {
        const size_type n = diag_.size();
        const size_type brows = (opB == Transpose::None) ? b.rows() : b.columns();
        const size_type bcols = (opB == Transpose::None) ? b.columns() : b.rows();
        if (brows != n || c.rows() != n || c.columns() != bcols) {
            throw std::invalid_argument{"miscibility::instrument::DiagonalMatrix matrix-matrix size mismatch"};
        }
        // TODO: try and remove allocations
        // Each result column j is alpha*(d (.) op(B)_{:,j}) + beta*C_{:,j}.
        Vector<T> col(n);
        Vector<T> ccol(n);
        for (size_type j = 0; j < bcols; ++j) {
            for (size_type i = 0; i < n; ++i) {
                col[i] = (opB == Transpose::None) ? b(i, j) : b(j, i);
                ccol[i] = c(i, j);
            }
            col.elementwise_product(diag_); // col_i = d_i * op(B)_{i,j}
            ccol.scale(beta);
            ccol.add_scaled(alpha, col);
            for (size_type i = 0; i < n; ++i) {
                c(i, j) = ccol[i];
            }
        }
    }

    [[nodiscard]] DenseMatrix<T> multiply(const DenseMatrix<T>& b, Transpose opB = Transpose::None) const
    {
        const size_type bcols = (opB == Transpose::None) ? b.columns() : b.rows();
        DenseMatrix<T> c(diag_.size(), bcols);
        multiply_into(b, c, T(1), T(0), opB);
        return c;
    }

    [[nodiscard]] DenseMatrix<T> operator*(const DenseMatrix<T>& b) const { return multiply(b); }

    // -- solve ----------------------------------------------------------------

    void solve_into(const Vector<T>& b, Vector<T>& x) const
    {
        const size_type n = diag_.size();
        if (b.size() != n || x.size() != n) {
            throw std::invalid_argument{"miscibility::instrument::DiagonalMatrix solve size mismatch"};
        }
        if (singular_) {
            throw std::runtime_error{"miscibility::instrument::DiagonalMatrix: singular matrix"};
        }
        x.copy(b);
        x.elementwise_quotient(diag_); // x_i = b_i / d_i
    }

    [[nodiscard]] Vector<T> solve(const Vector<T>& b) const
    {
        Vector<T> x(diag_.size());
        solve_into(b, x);
        return x;
    }

private:
    void recompute_singular() noexcept
    {
        zero_count_ = 0;
        for (size_type i = 0; i < diag_.size(); ++i) {
            if (diag_[i] == T(0)) {
                ++zero_count_;
            }
        }
        singular_ = (zero_count_ > 0);
    }

    Vector<T, N> diag_{};
    size_type zero_count_{0};
    bool singular_{false};
};

} // namespace miscibility::instrument
