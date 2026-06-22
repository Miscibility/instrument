#pragma once

#include "instrument/array.hpp"
#include "instrument/context.hpp"

#include <algorithm>
#include <ginkgo/ginkgo.hpp>
#include <initializer_list>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>

namespace miscibility::instrument {

/// A single-column linear-algebra vector wrapping ``gko::matrix::Dense<T>``.
///
/// ``Vector`` pairs the dense BLAS-1 operations (scaling, axpy, dot products,
/// norms) with a full STL-style container surface — iterators, ``operator[]``,
/// ``as_span`` — so the same object works with Ginkgo solvers and with
/// ``<algorithm>``/ranges. The math methods mutate in place and return ``*this``,
/// so they chain:
///
/// .. code-block:: cpp
///
///     Context ctx;
///     Vector<double> x{ctx, "x", {1.0, 2.0, 3.0}};
///     Vector<double> y{ctx, "y", 3, 1.0};
///     y.add_scaled(2.0, x);            // y := y + 2*x
///     double n = y.norm2();
///
/// Because it derives from :cpp:`OperatorHandle`, a vector registers itself with
/// the context (so its applies are timed) and converts implicitly to a
/// ``gko::LinOp`` for raw Ginkgo APIs.
///
/// Element access and iteration require a host executor; on a device executor they
/// throw ``std::logic_error`` (see :cpp:`is_host`). A ``Vector`` is deliberately a
/// single column — multiple right-hand sides belong to ``DenseMatrix``.
///
/// :tparam T: Floating-point element type.
template<Scalar T = double> class Vector : public OperatorHandle {
public:
    using value_type = T;
    using size_type = gko::size_type;
    using iterator = T*;
    using const_iterator = const T*;

    /// Creates an ``n``-element column initialized to zero.
    ///
    /// :param ctx: Context the vector lives on and registers with.
    /// :param name: Name reported for this vector in timing output.
    /// :param n: Number of elements.
    Vector(Context& ctx, std::string name, size_type n);

    /// Creates an ``n``-element column with every entry set to ``value``.
    ///
    /// :param ctx: Context the vector lives on and registers with.
    /// :param name: Name reported for this vector in timing output.
    /// :param n: Number of elements.
    /// :param value: Value written to every element.
    Vector(Context& ctx, std::string name, size_type n, T value);

    /// Creates a column holding a copy of the listed values, in order.
    ///
    /// :param ctx: Context the vector lives on and registers with.
    /// :param name: Name reported for this vector in timing output.
    /// :param values: The elements; the vector's length is their count.
    Vector(Context& ctx, std::string name, std::initializer_list<T> values);

    /// Creates a column by copying ``n`` values from a caller-owned buffer.
    ///
    /// The data is copied, not aliased: later changes to ``data`` do not affect
    /// the vector. Use :cpp:`view` for a zero-copy wrapper instead.
    ///
    /// :param ctx: Context the vector lives on and registers with.
    /// :param name: Name reported for this vector in timing output.
    /// :param data: Pointer to the first of ``n`` values to copy.
    /// :param n: Number of elements to copy.
    Vector(Context& ctx, std::string name, const T* data, size_type n);

    /// Wraps a caller-owned host buffer without copying it.
    ///
    /// The returned vector shares storage with ``data``: writes through the
    /// vector are visible in the buffer and vice versa. The buffer must outlive
    /// the vector, and this is host-executors only.
    ///
    /// :param ctx: Context the vector lives on and registers with.
    /// :param name: Name reported for this vector in timing output.
    /// :param data: Pointer to the first of ``n`` values to wrap in place.
    /// :param n: Number of elements the buffer holds.
    /// :returns: A vector aliasing ``data``.
    static Vector view(Context& ctx, std::string name, T* data, size_type n);

    /// Pointer to the underlying contiguous storage.
    ///
    /// :throws std::logic_error: if the executor is not a host executor.
    [[nodiscard]] T* data();
    /// Pointer to the underlying contiguous storage.
    ///
    /// :throws std::logic_error: if the executor is not a host executor.
    [[nodiscard]] const T* data() const;
    /// Number of elements.
    [[nodiscard]] size_type size() const noexcept;
    /// True when the vector has no elements.
    [[nodiscard]] bool empty() const noexcept;
    /// True when the executor is a host (Reference/OMP) executor, where element
    /// access and iteration are valid.
    [[nodiscard]] bool is_host() const noexcept;

    /// Unchecked element access (host executors only).
    [[nodiscard]] T& operator[](size_type i);
    /// Unchecked element access (host executors only).
    [[nodiscard]] const T& operator[](size_type i) const;

    /// Bounds-checked element access.
    ///
    /// :throws std::out_of_range: if ``i`` is not less than :cpp:`size`.
    [[nodiscard]] T& at(size_type i);
    /// Bounds-checked element access.
    ///
    /// :throws std::out_of_range: if ``i`` is not less than :cpp:`size`.
    [[nodiscard]] const T& at(size_type i) const;

    [[nodiscard]] iterator begin();
    /// Past-the-end iterator over the elements.
    [[nodiscard]] iterator end();
    [[nodiscard]] const_iterator begin() const;
    /// Past-the-end iterator over the elements.
    [[nodiscard]] const_iterator end() const;

    /// A span over the elements (length :cpp:`size`).
    [[nodiscard]] std::span<T> as_span();
    /// A span over the elements (length :cpp:`size`).
    [[nodiscard]] std::span<const T> as_span() const;

    /// Sets every element to ``value``.
    Vector& fill(T value);
    /// Multiplies every element by ``alpha`` (``x := alpha*x``).
    Vector& scale(T alpha);

    /// Adds a scaled vector in place (``y := y + alpha*x``), the axpy operation.
    ///
    /// :param alpha: Scalar multiplier applied to ``x``.
    /// :param x: Vector to add; must match this vector in length.
    /// :throws std::invalid_argument: if ``x`` and this vector differ in length.
    Vector& add_scaled(T alpha, const Vector& x);

    /// Subtracts a scaled vector in place (``y := y - alpha*x``).
    ///
    /// :param alpha: Scalar multiplier applied to ``x``.
    /// :param x: Vector to subtract; must match this vector in length.
    /// :throws std::invalid_argument: if ``x`` and this vector differ in length.
    Vector& sub_scaled(T alpha, const Vector& x);

    /// The inner product with ``x``.
    ///
    /// :throws std::invalid_argument: if ``x`` and this vector differ in length.
    [[nodiscard]] T dot(const Vector& x) const;
    /// The Euclidean (L2) norm.
    [[nodiscard]] T norm2() const;
    /// The sum of absolute values (the L1 norm).
    [[nodiscard]] T norm1() const;

    /// Scales the vector by ``alpha`` (see :cpp:`scale`).
    Vector& operator*=(T alpha);
    /// Divides every element by ``alpha``.
    Vector& operator/=(T alpha);
    /// Adds ``x`` element-wise (see :cpp:`add_scaled`).
    Vector& operator+=(const Vector& x);
    /// Subtracts ``x`` element-wise (see :cpp:`sub_scaled`).
    Vector& operator-=(const Vector& x);

private:
    using dense_type = gko::matrix::Dense<T>;
    using executor_ptr = std::shared_ptr<const gko::Executor>;

    Vector(Context& ctx, std::string name, std::shared_ptr<dense_type> dense);

    [[nodiscard]] dense_type* dense() noexcept;
    [[nodiscard]] const dense_type* dense() const noexcept;

    void require_host() const;
    void check_same_size(size_type other) const;

    // Reusable 1x1 scratch buffers, so the BLAS-1 methods don't allocate per call.
    [[nodiscard]] dense_type* scalar(T value) const;
    [[nodiscard]] dense_type& result_buffer() const;
    [[nodiscard]] T read_scalar(const dense_type& result) const;

    static std::shared_ptr<dense_type> make_zeros(const executor_ptr& exec, size_type n);
    static std::shared_ptr<dense_type> make_filled(const executor_ptr& exec, size_type n, T value);
    static std::shared_ptr<dense_type> make_copy(const executor_ptr& exec, const T* data, size_type n);
    static std::shared_ptr<dense_type> make_view(const executor_ptr& exec, T* data, size_type n);

    mutable std::shared_ptr<dense_type> scalar_scratch_;
    mutable std::shared_ptr<dense_type> result_scratch_;
};

template<Scalar T>
Vector<T>::Vector(Context& ctx, std::string name, std::shared_ptr<dense_type> dense) :
    OperatorHandle(ctx, std::move(name), std::move(dense), scalar_type_of<T>())
{
}

template<Scalar T>
Vector<T>::Vector(Context& ctx, std::string name, size_type n) :
    Vector(ctx, std::move(name), make_zeros(ctx.executor(), n))
{
}

template<Scalar T>
Vector<T>::Vector(Context& ctx, std::string name, size_type n, T value) :
    Vector(ctx, std::move(name), make_filled(ctx.executor(), n, value))
{
}

template<Scalar T>
Vector<T>::Vector(Context& ctx, std::string name, std::initializer_list<T> values) :
    Vector(ctx, std::move(name), make_copy(ctx.executor(), values.begin(), values.size()))
{
}

template<Scalar T>
Vector<T>::Vector(Context& ctx, std::string name, const T* data, size_type n) :
    Vector(ctx, std::move(name), make_copy(ctx.executor(), data, n))
{
}

template<Scalar T> Vector<T> Vector<T>::view(Context& ctx, std::string name, T* data, size_type n)
{
    return Vector{ctx, std::move(name), make_view(ctx.executor(), data, n)};
}

template<Scalar T> std::shared_ptr<gko::matrix::Dense<T>> Vector<T>::make_zeros(const executor_ptr& exec, size_type n)
{
    auto dense = dense_type::create(exec, gko::dim<2>{n, 1});
    dense->fill(T{0});
    return gko::share(std::move(dense));
}

template<Scalar T>
std::shared_ptr<gko::matrix::Dense<T>> Vector<T>::make_filled(const executor_ptr& exec, size_type n, T value)
{
    auto dense = dense_type::create(exec, gko::dim<2>{n, 1});
    dense->fill(value);
    return gko::share(std::move(dense));
}

template<Scalar T>
std::shared_ptr<gko::matrix::Dense<T>> Vector<T>::make_copy(const executor_ptr& exec, const T* data, size_type n)
{
    gko::array<T> host_values{exec->get_master(), n};
    std::copy_n(data, n, host_values.get_data());
    auto dense = dense_type::create(exec, gko::dim<2>{n, 1}, gko::array<T>{exec, std::move(host_values)}, 1);
    return gko::share(std::move(dense));
}

template<Scalar T>
std::shared_ptr<gko::matrix::Dense<T>> Vector<T>::make_view(const executor_ptr& exec, T* data, size_type n)
{
    auto dense = dense_type::create(exec, gko::dim<2>{n, 1}, gko::make_array_view(exec, n, data), 1);
    return gko::share(std::move(dense));
}

template<Scalar T> T* Vector<T>::data()
{
    require_host();
    return dense()->get_values();
}
template<Scalar T> const T* Vector<T>::data() const
{
    require_host();
    return dense()->get_const_values();
}
template<Scalar T> typename Vector<T>::size_type Vector<T>::size() const noexcept { return dense()->get_size()[0]; }
template<Scalar T> bool Vector<T>::empty() const noexcept { return size() == 0; }
template<Scalar T> bool Vector<T>::is_host() const noexcept { return context().is_host(); }
template<Scalar T> void Vector<T>::require_host() const
{
    if (!is_host()) {
        throw std::logic_error{"miscibility::instrument::Vector element access requires a host executor"};
    }
}

template<Scalar T> T& Vector<T>::operator[](size_type i) { return data()[i]; }
template<Scalar T> const T& Vector<T>::operator[](size_type i) const { return data()[i]; }
template<Scalar T> T& Vector<T>::at(size_type i)
{
    if (i >= size()) {
        throw std::out_of_range{"miscibility::instrument::Vector::at"};
    }
    return data()[i];
}
template<Scalar T> const T& Vector<T>::at(size_type i) const
{
    if (i >= size()) {
        throw std::out_of_range{"miscibility::instrument::Vector::at"};
    }
    return data()[i];
}

template<Scalar T> typename Vector<T>::iterator Vector<T>::begin() { return data(); }
template<Scalar T> typename Vector<T>::iterator Vector<T>::end() { return data() + size(); }
template<Scalar T> typename Vector<T>::const_iterator Vector<T>::begin() const { return data(); }
template<Scalar T> typename Vector<T>::const_iterator Vector<T>::end() const { return data() + size(); }

template<Scalar T> std::span<T> Vector<T>::as_span() { return {data(), size()}; }
template<Scalar T> std::span<const T> Vector<T>::as_span() const { return {data(), size()}; }

template<Scalar T> Vector<T>& Vector<T>::fill(T value)
{
    dense()->fill(value);
    return *this;
}
template<Scalar T> Vector<T>& Vector<T>::scale(T alpha)
{
    dense()->scale(scalar(alpha));
    return *this;
}
template<Scalar T> Vector<T>& Vector<T>::add_scaled(T alpha, const Vector& x)
{
    check_same_size(x.size());
    dense()->add_scaled(scalar(alpha), x.dense());
    return *this;
}
template<Scalar T> Vector<T>& Vector<T>::sub_scaled(T alpha, const Vector& x)
{
    check_same_size(x.size());
    dense()->sub_scaled(scalar(alpha), x.dense());
    return *this;
}

template<Scalar T> T Vector<T>::dot(const Vector& x) const
{
    check_same_size(x.size());
    dense_type& result = result_buffer();
    dense()->compute_dot(x.dense(), &result);
    return read_scalar(result);
}
template<Scalar T> T Vector<T>::norm2() const
{
    dense_type& result = result_buffer();
    dense()->compute_norm2(&result);
    return read_scalar(result);
}
template<Scalar T> T Vector<T>::norm1() const
{
    dense_type& result = result_buffer();
    dense()->compute_norm1(&result);
    return read_scalar(result);
}

template<Scalar T> Vector<T>& Vector<T>::operator*=(T alpha) { return scale(alpha); }
template<Scalar T> Vector<T>& Vector<T>::operator/=(T alpha) { return scale(T{1} / alpha); }
template<Scalar T> Vector<T>& Vector<T>::operator+=(const Vector& x) { return add_scaled(T{1}, x); }
template<Scalar T> Vector<T>& Vector<T>::operator-=(const Vector& x) { return sub_scaled(T{1}, x); }

template<Scalar T> gko::matrix::Dense<T>* Vector<T>::dense() noexcept
{
    return static_cast<dense_type*>(linop().get());
}
template<Scalar T> const gko::matrix::Dense<T>* Vector<T>::dense() const noexcept
{
    return static_cast<const dense_type*>(linop().get());
}

template<Scalar T> gko::matrix::Dense<T>* Vector<T>::scalar(T value) const
{
    if (!scalar_scratch_) {
        scalar_scratch_ = dense_type::create(context().executor(), gko::dim<2>{1, 1});
    }
    scalar_scratch_->fill(value);
    return scalar_scratch_.get();
}

template<Scalar T> gko::matrix::Dense<T>& Vector<T>::result_buffer() const
{
    if (!result_scratch_) {
        result_scratch_ = dense_type::create(context().executor(), gko::dim<2>{1, 1});
    }
    return *result_scratch_;
}

template<Scalar T> T Vector<T>::read_scalar(const dense_type& result) const
{
    return context().executor()->copy_val_to_host(result.get_const_values());
}

template<Scalar T> void Vector<T>::check_same_size(size_type other) const
{
    if (other != size()) {
        throw std::invalid_argument{"miscibility::instrument::Vector size mismatch"};
    }
}

} // namespace miscibility::instrument
