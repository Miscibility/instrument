#pragma once

#include "instrument/array.hpp"
#include "instrument/context.hpp"
#include "instrument/vector.hpp"

#include <cassert>
#include <ginkgo/ginkgo.hpp>
#include <initializer_list>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>

namespace miscibility::instrument {

/// A dense matrix wrapping ``gko::matrix::Dense<T>`` (more than one column).
///
/// ``DenseMatrix`` is the multi-column counterpart to :cpp:`Vector` over the same
/// underlying Ginkgo type; the two are kept separate purely so call sites read by
/// intent. It supports element access, the matrix–vector and matrix–matrix
/// products through :cpp:`apply`, and extraction of its values for LAPACK.
///
/// .. code-block:: cpp
///
///     Context ctx;
///     DenseMatrix<double> a{ctx, "A", {{1.0, 2.0}, {3.0, 4.0}}};
///     Vector<double> x{ctx, "x", {1.0, 1.0}};
///     Vector<double> y{ctx, "y", 2};
///     a.apply(x, y);                   // y = A*x = {3, 7}
///
/// Storage is row-major with a stride: element ``(i, j)`` lives at
/// ``data()[i*stride() + j]``, which is the layout to hand to LAPACKE as
/// ``LAPACK_ROW_MAJOR``. Element access and iteration are valid on host
/// executors only.
///
/// :tparam T: Floating-point element type.
template<Scalar T = double> class DenseMatrix : public OperatorHandle {
public:
    using value_type = T;
    using size_type = gko::size_type;

    /// Creates a matrix of the given shape, initialized to zero.
    ///
    /// :param ctx: Context the matrix lives on and registers with.
    /// :param name: Name reported for this matrix in timing output.
    /// :param shape: Number of rows and columns.
    DenseMatrix(Context& ctx, std::string name, gko::dim<2> shape);

    /// Creates a matrix from nested initializer lists, one inner list per row.
    ///
    /// :param ctx: Context the matrix lives on and registers with.
    /// :param name: Name reported for this matrix in timing output.
    /// :param rows: Outer list of rows; each inner list is one row's entries.
    /// :throws std::invalid_argument: if the rows are not all the same length.
    DenseMatrix(Context& ctx, std::string name, std::initializer_list<std::initializer_list<T>> rows);

    /// Creates a matrix from a Ginkgo ``matrix_data`` coordinate listing.
    ///
    /// :param ctx: Context the matrix lives on and registers with.
    /// :param name: Name reported for this matrix in timing output.
    /// :param data: The entries and shape to populate from.
    DenseMatrix(Context& ctx, std::string name, const gko::matrix_data<T, int>& data);

    /// Number of rows.
    [[nodiscard]] size_type rows() const noexcept;
    /// Number of columns.
    [[nodiscard]] size_type cols() const noexcept;
    /// The matrix shape as ``{rows, cols}``.
    [[nodiscard]] gko::dim<2> shape() const noexcept;
    /// Row stride: the distance in elements between the starts of consecutive rows in :cpp:`data`.
    [[nodiscard]] size_type stride() const noexcept;

    /// The entry at row ``i``, column ``j`` (host executors only).
    [[nodiscard]] T& at(size_type i, size_type j);
    /// The entry at row ``i``, column ``j`` (host executors only).
    [[nodiscard]] const T& at(size_type i, size_type j) const;
    /// Pointer to the row-major storage; entry ``(i, j)`` is at ``data()[i*stride() + j]``.
    [[nodiscard]] T* data() noexcept;
    /// Pointer to the row-major storage; entry ``(i, j)`` is at ``data()[i*stride() + j]``.
    [[nodiscard]] const T* data() const noexcept;

    /// Computes the matrix–vector product ``y = A*x``.
    ///
    /// :param x: Input vector, length :cpp:`cols`.
    /// :param y: Output vector, length :cpp:`rows`; overwritten.
    void apply(const Vector<T>& x, Vector<T>& y) const;

    /// Computes the matrix–matrix product ``result = A*other``.
    ///
    /// :param other: Right operand whose row count matches :cpp:`cols`.
    /// :param result: Output matrix, shaped ``{rows, other.cols}``; overwritten.
    void apply(const DenseMatrix<T>& other, DenseMatrix<T>& result) const;

    /// Computes the fused update ``y = alpha*A*x + beta*y``.
    ///
    /// With ``alpha = 1`` and ``beta = -1`` and ``y`` holding ``b``, this leaves
    /// the residual ``A*x - b`` in ``y``.
    ///
    /// :param alpha: Scalar on the product ``A*x``.
    /// :param x: Input vector, length :cpp:`cols`.
    /// :param beta: Scalar on the incoming contents of ``y``.
    /// :param y: Input/output vector, length :cpp:`rows`.
    void apply(T alpha, const Vector<T>& x, T beta, Vector<T>& y) const;

    /// Sets every entry to ``value``.
    DenseMatrix& fill(T value);
    /// Multiplies every entry by ``alpha``.
    DenseMatrix& scale(T alpha);
    /// Adds ``alpha*other`` entrywise.
    ///
    /// :param alpha: Scalar multiplier applied to ``other``.
    /// :param other: Matrix to add; must match this matrix in shape.
    /// :throws std::invalid_argument: if ``other`` and this matrix differ in shape.
    DenseMatrix& add_scaled(T alpha, const DenseMatrix& other);

    /// Copies the entries into a caller buffer in row-major order, compacting the stride.
    ///
    /// The buffer receives ``rows()*cols()`` contiguous values (leading dimension
    /// equal to :cpp:`cols`), ready to pass to a ``LAPACK_ROW_MAJOR`` routine.
    ///
    /// :param out: Destination buffer of at least ``rows()*cols()`` elements.
    void copy_values(T* out) const;

    /// Returns the entries as a Ginkgo ``matrix_data``, for inspection or format conversion.
    [[nodiscard]] gko::matrix_data<T, int> to_matrix_data() const;

private:
    using dense_type = gko::matrix::Dense<T>;

    DenseMatrix(Context& ctx, std::string name, std::shared_ptr<dense_type> dense);

    [[nodiscard]] dense_type* dense() noexcept;
    [[nodiscard]] const dense_type* dense() const noexcept;

    [[nodiscard]] std::unique_ptr<dense_type> make_scalar(T value) const;
    void check_same_shape(gko::dim<2> other) const;

    static std::shared_ptr<dense_type> make_zeros(Context& ctx, gko::dim<2> shape);
    static std::shared_ptr<dense_type> make_from_rows(Context& ctx,
                                                      std::initializer_list<std::initializer_list<T>> rows);
    static std::shared_ptr<dense_type> make_from_data(Context& ctx, const gko::matrix_data<T, int>& data);
};

template<Scalar T>
DenseMatrix<T>::DenseMatrix(Context& ctx, std::string name, std::shared_ptr<dense_type> dense) :
    OperatorHandle(ctx, std::move(name), std::move(dense))
{
}

template<Scalar T>
DenseMatrix<T>::DenseMatrix(Context& ctx, std::string name, gko::dim<2> shape) :
    DenseMatrix(ctx, std::move(name), make_zeros(ctx, shape))
{
}

template<Scalar T>
DenseMatrix<T>::DenseMatrix(Context& ctx, std::string name, std::initializer_list<std::initializer_list<T>> rows) :
    DenseMatrix(ctx, std::move(name), make_from_rows(ctx, rows))
{
}

template<Scalar T>
DenseMatrix<T>::DenseMatrix(Context& ctx, std::string name, const gko::matrix_data<T, int>& data) :
    DenseMatrix(ctx, std::move(name), make_from_data(ctx, data))
{
}

template<Scalar T> std::shared_ptr<gko::matrix::Dense<T>> DenseMatrix<T>::make_zeros(Context& ctx, gko::dim<2> shape)
{
    auto dense = dense_type::create(ctx.executor(), shape);
    dense->fill(T{0});
    return gko::share(std::move(dense));
}

template<Scalar T>
std::shared_ptr<gko::matrix::Dense<T>>
DenseMatrix<T>::make_from_rows(Context& ctx, std::initializer_list<std::initializer_list<T>> rows)
{
    const size_type row_count = rows.size();
    const size_type col_count = row_count == 0 ? 0 : rows.begin()->size();
    for (const auto& row : rows) {
        if (row.size() != col_count) {
            throw std::invalid_argument{"miscibility::instrument::DenseMatrix ragged initializer"};
        }
    }
    auto exec = ctx.executor();
    gko::array<T> host_values{exec->get_master(), row_count * col_count};
    T* write = host_values.get_data();
    for (const auto& row : rows) {
        for (T value : row) {
            *write++ = value;
        }
    }
    auto dense = dense_type::create(exec, gko::dim<2>{row_count, col_count},
                                    gko::array<T>{exec, std::move(host_values)}, col_count);
    return gko::share(std::move(dense));
}

template<Scalar T>
std::shared_ptr<gko::matrix::Dense<T>> DenseMatrix<T>::make_from_data(Context& ctx,
                                                                      const gko::matrix_data<T, int>& data)
{
    auto dense = dense_type::create(ctx.executor(), data.size);
    dense->read(data);
    return gko::share(std::move(dense));
}

template<Scalar T> typename DenseMatrix<T>::size_type DenseMatrix<T>::rows() const noexcept
{
    return dense()->get_size()[0];
}
template<Scalar T> typename DenseMatrix<T>::size_type DenseMatrix<T>::cols() const noexcept
{
    return dense()->get_size()[1];
}
template<Scalar T> gko::dim<2> DenseMatrix<T>::shape() const noexcept { return dense()->get_size(); }
template<Scalar T> typename DenseMatrix<T>::size_type DenseMatrix<T>::stride() const noexcept
{
    return dense()->get_stride();
}

template<Scalar T> T& DenseMatrix<T>::at(size_type i, size_type j)
{
    assert(context().executor()->get_master() == context().executor() && "element access requires a host executor");
    return dense()->at(i, j);
}
template<Scalar T> const T& DenseMatrix<T>::at(size_type i, size_type j) const
{
    assert(context().executor()->get_master() == context().executor() && "element access requires a host executor");
    return dense()->at(i, j);
}
template<Scalar T> T* DenseMatrix<T>::data() noexcept { return dense()->get_values(); }
template<Scalar T> const T* DenseMatrix<T>::data() const noexcept { return dense()->get_const_values(); }

template<Scalar T> void DenseMatrix<T>::apply(const Vector<T>& x, Vector<T>& y) const
{
    linop()->apply(x.linop(), y.linop());
}
template<Scalar T> void DenseMatrix<T>::apply(const DenseMatrix<T>& other, DenseMatrix<T>& result) const
{
    linop()->apply(other.linop(), result.linop());
}
template<Scalar T> void DenseMatrix<T>::apply(T alpha, const Vector<T>& x, T beta, Vector<T>& y) const
{
    linop()->apply(make_scalar(alpha), x.linop(), make_scalar(beta), y.linop());
}

template<Scalar T> DenseMatrix<T>& DenseMatrix<T>::fill(T value)
{
    dense()->fill(value);
    return *this;
}
template<Scalar T> DenseMatrix<T>& DenseMatrix<T>::scale(T alpha)
{
    dense()->scale(make_scalar(alpha));
    return *this;
}
template<Scalar T> DenseMatrix<T>& DenseMatrix<T>::add_scaled(T alpha, const DenseMatrix& other)
{
    check_same_shape(other.shape());
    dense()->add_scaled(make_scalar(alpha), other.dense());
    return *this;
}

template<Scalar T> void DenseMatrix<T>::copy_values(T* out) const
{
    const size_type row_count = rows();
    const size_type col_count = cols();
    const size_type row_stride = stride();
    const T* values = data();
    for (size_type i = 0; i < row_count; ++i) {
        for (size_type j = 0; j < col_count; ++j) {
            out[(i * col_count) + j] = values[(i * row_stride) + j];
        }
    }
}
template<Scalar T> gko::matrix_data<T, int> DenseMatrix<T>::to_matrix_data() const
{
    gko::matrix_data<T, int> data;
    dense()->write(data);
    return data;
}

template<Scalar T> gko::matrix::Dense<T>* DenseMatrix<T>::dense() noexcept
{
    return static_cast<dense_type*>(linop().get());
}
template<Scalar T> const gko::matrix::Dense<T>* DenseMatrix<T>::dense() const noexcept
{
    return static_cast<const dense_type*>(linop().get());
}

template<Scalar T> std::unique_ptr<gko::matrix::Dense<T>> DenseMatrix<T>::make_scalar(T value) const
{
    auto scalar = dense_type::create(context().executor(), gko::dim<2>{1, 1});
    scalar->fill(value);
    return scalar;
}

template<Scalar T> void DenseMatrix<T>::check_same_shape(gko::dim<2> other) const
{
    if (other != shape()) {
        throw std::invalid_argument{"miscibility::instrument::DenseMatrix size mismatch"};
    }
}

} // namespace miscibility::instrument
