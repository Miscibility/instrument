#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <hwy/aligned_allocator.h>
#include <hwy/contrib/math/math-inl.h>
#include <hwy/highway.h>
#include <initializer_list>
#include <limits>
#include <new>
#include <span>
#include <stdexcept>
#include <type_traits>
#include <utility>

namespace miscibility::instrument {

/// Element types an :cpp:`Array` can hold: the standard floating-point types.
///
/// The arithmetic kernels are built on SIMD floating-point operations, so the
/// element type is restricted to ``float``, ``double``, and ``long double``.
template<class T>
concept Scalar = std::floating_point<T>;

/// Length tag selecting a runtime-sized (heap-allocated) :cpp:`Array`.
///
/// Pass this as the ``N`` template argument (it is the default) to get an array
/// whose length is fixed at construction; any other value gives a fixed-size
/// array whose storage lives inline.
inline constexpr std::size_t dynamic = std::dynamic_extent;

namespace detail {

namespace hn = hwy::HWY_NAMESPACE;

inline constexpr std::size_t alignment = HWY_ALIGNMENT;

template<Scalar T> inline constexpr std::size_t pad_step = (alignment / sizeof(T) == 0) ? 1 : alignment / sizeof(T);

template<Scalar T> [[nodiscard]] constexpr std::size_t padded_count(std::size_t n) noexcept
{
    const std::size_t s = pad_step<T>;
    return ((n + s - 1) / s) * s;
}

template<Scalar T> void scale(T* p, std::size_t cap, T a) noexcept
{
    const hn::ScalableTag<T> d;
    const std::size_t lanes = hn::Lanes(d);
    const auto va = hn::Set(d, a);
    for (std::size_t i = 0; i < cap; i += lanes) {
        hn::Store(hn::Mul(hn::Load(d, p + i), va), d, p + i);
    }
}

template<Scalar T> void add_scaled(T* y, const T* x, std::size_t cap, T a) noexcept
{
    const hn::ScalableTag<T> d;
    const std::size_t lanes = hn::Lanes(d);
    const auto va = hn::Set(d, a);
    for (std::size_t i = 0; i < cap; i += lanes) {
        const auto vy = hn::Load(d, y + i);
        const auto vx = hn::Load(d, x + i);
        hn::Store(hn::MulAdd(va, vx, vy), d, y + i);
    }
}

template<Scalar T> [[nodiscard]] T dot(const T* a, const T* b, std::size_t cap) noexcept
{
    const hn::ScalableTag<T> d;
    const std::size_t lanes = hn::Lanes(d);
    auto acc = hn::Zero(d);
    for (std::size_t i = 0; i < cap; i += lanes) {
        acc = hn::MulAdd(hn::Load(d, a + i), hn::Load(d, b + i), acc);
    }
    return hn::ReduceSum(d, acc);
}

template<Scalar T> [[nodiscard]] T sum(const T* p, std::size_t cap) noexcept
{
    const hn::ScalableTag<T> d;
    const std::size_t lanes = hn::Lanes(d);
    auto acc = hn::Zero(d);
    for (std::size_t i = 0; i < cap; i += lanes) {
        acc = hn::Add(acc, hn::Load(d, p + i));
    }
    return hn::ReduceSum(d, acc);
}

template<Scalar T> [[nodiscard]] T absolute_sum(const T* p, std::size_t cap) noexcept
{
    const hn::ScalableTag<T> d;
    const std::size_t lanes = hn::Lanes(d);
    auto acc = hn::Zero(d);
    for (std::size_t i = 0; i < cap; i += lanes) {
        acc = hn::Add(acc, hn::Abs(hn::Load(d, p + i)));
    }
    return hn::ReduceSum(d, acc);
}

template<Scalar T> [[nodiscard]] std::size_t index_of_max_magnitude(const T* p, std::size_t cap) noexcept
{
    const hn::ScalableTag<T> d;
    const hn::RebindToUnsigned<decltype(d)> du;
    using TU = hn::TFromD<decltype(du)>;
    const std::size_t lanes = hn::Lanes(d);

    auto best_mag = hn::Zero(d);
    auto best_idx = hn::Iota(du, 0);
    for (std::size_t i = 0; i < cap; i += lanes) {
        const auto mag = hn::Abs(hn::Load(d, p + i));
        const auto idx = hn::Iota(du, static_cast<TU>(i));
        const auto greater = hn::Gt(mag, best_mag);
        best_mag = hn::IfThenElse(greater, mag, best_mag);
        best_idx = hn::IfThenElse(hn::RebindMask(du, greater), idx, best_idx);
    }

    const auto max_mag = hn::Set(d, hn::ReduceMax(d, best_mag));
    const auto holds_max = hn::RebindMask(du, hn::Eq(best_mag, max_mag));
    const auto sentinel = hn::Set(du, std::numeric_limits<TU>::max());
    const auto candidates = hn::IfThenElse(holds_max, best_idx, sentinel);
    return static_cast<std::size_t>(hn::ReduceMin(du, candidates));
}

template<Scalar T, class F> void map(T* p, std::size_t cap, F f) noexcept
{
    const hn::ScalableTag<T> d;
    const std::size_t lanes = hn::Lanes(d);
    for (std::size_t i = 0; i < cap; i += lanes) {
        hn::Store(f(d, hn::Load(d, p + i)), d, p + i);
    }
}

template<Scalar T, class G> void zip(T* y, const T* x, std::size_t cap, G g) noexcept
{
    const hn::ScalableTag<T> d;
    const std::size_t lanes = hn::Lanes(d);
    for (std::size_t i = 0; i < cap; i += lanes) {
        hn::Store(g(d, hn::Load(d, y + i), hn::Load(d, x + i)), d, y + i);
    }
}

template<Scalar T> class dynamic_storage {
public:
    dynamic_storage() noexcept = default;

    explicit dynamic_storage(std::size_t n) : size_{n}, capacity_{padded_count<T>(n)}
    {
        if (capacity_ != 0) {
            data_ = allocate(capacity_);
            std::fill_n(data_, capacity_, T{});
        }
    }

    dynamic_storage(const dynamic_storage& o) : size_{o.size_}, capacity_{o.capacity_}
    {
        if (capacity_ != 0) {
            data_ = allocate(capacity_);
            std::copy_n(o.data_, capacity_, data_);
        }
    }

    dynamic_storage(dynamic_storage&& o) noexcept :
        data_{std::exchange(o.data_, nullptr)},
        size_{std::exchange(o.size_, std::size_t{0})},
        capacity_{std::exchange(o.capacity_, std::size_t{0})}
    {
    }

    dynamic_storage& operator=(const dynamic_storage& o)
    {
        if (this != &o) {
            if (data_ != nullptr && o.data_ != nullptr && capacity_ == o.capacity_) {
                std::copy_n(o.data_, capacity_, data_);
                size_ = o.size_;
            }
            else {
                dynamic_storage tmp{o};
                swap(tmp);
            }
        }
        return *this;
    }

    dynamic_storage& operator=(dynamic_storage&& o) noexcept
    {
        if (this != &o) {
            deallocate(data_, capacity_);
            data_ = std::exchange(o.data_, nullptr);
            size_ = std::exchange(o.size_, std::size_t{0});
            capacity_ = std::exchange(o.capacity_, std::size_t{0});
        }
        return *this;
    }

    ~dynamic_storage() { deallocate(data_, capacity_); }

    void swap(dynamic_storage& o) noexcept
    {
        using std::swap;
        swap(data_, o.data_);
        swap(size_, o.size_);
        swap(capacity_, o.capacity_);
    }

    [[nodiscard]] T* data() noexcept { return data_; }
    [[nodiscard]] const T* data() const noexcept { return data_; }
    [[nodiscard]] std::size_t size() const noexcept { return size_; }
    [[nodiscard]] std::size_t capacity() const noexcept { return capacity_; }

private:
    static T* allocate(std::size_t cap)
    {
        return static_cast<T*>(::operator new(cap * sizeof(T), std::align_val_t{alignment}));
    }
    static void deallocate(T* p, std::size_t) noexcept { ::operator delete(p, std::align_val_t{alignment}); }

    T* data_{nullptr};
    std::size_t size_{0};
    std::size_t capacity_{0};
};

template<Scalar T, std::size_t N> class static_storage {
    static_assert(N > 0, "use miscibility::instrument::dynamic for length 0 / runtime length");

public:
    [[nodiscard]] T* data() noexcept { return data_; }
    [[nodiscard]] const T* data() const noexcept { return data_; }
    [[nodiscard]] static constexpr std::size_t size() noexcept { return N; }
    [[nodiscard]] static constexpr std::size_t capacity() noexcept { return capacity_; }

private:
    static constexpr std::size_t capacity_ = padded_count<T>(N);
    alignas(alignment) T data_[capacity_]{};
};

template<Scalar T, std::size_t N>
using storage_t = std::conditional_t<N == dynamic, dynamic_storage<T>, static_storage<T, N>>;

} // namespace detail

/// A contiguous, SIMD-friendly numeric array with mutating math operations.
///
/// The array's storage is over-allocated and aligned so that the whole buffer
/// can be processed in full SIMD vectors without a scalar tail loop. The extra
/// trailing slots (between :cpp:`size` and :cpp:`capacity`) are kept zeroed, so
/// reductions and element-wise transforms run over the padded capacity yet
/// still produce the mathematically correct result for the logical length.
///
/// ``N`` chooses the storage strategy. The default, :cpp:`dynamic`, gives a
/// heap-allocated array whose length is set at construction. Any other value
/// fixes the length at compile time and stores the elements inline.
///
/// The arithmetic methods mutate the array in place and return ``*this``, so
/// they chain:
///
/// .. code-block:: cpp
///
///     Array<double> x{1.0, 2.0, 3.0};
///     x.scale(2.0).add_scaled(1.0, y);   // x = 2*x + y
///     double s = x.sum();
///
/// :tparam T: Floating-point element type.
/// :tparam N: Fixed length, or :cpp:`dynamic` for a runtime-sized array.
template<Scalar T, std::size_t N = dynamic> class Array {
    using storage = detail::storage_t<T, N>;

public:
    using value_type = T;
    using size_type = std::size_t;
    using iterator = T*;
    using const_iterator = const T*;

    /// True when ``N`` is :cpp:`dynamic`, i.e. the length is a runtime value.
    static constexpr bool is_dynamic = (N == dynamic);

    /// Constructs an empty array (length 0 when dynamic; zero-filled when fixed-size).
    Array() = default;

    /// Constructs a dynamic array of ``n`` zero-initialized elements.
    ///
    /// :param n: Number of logical elements.
    explicit Array(size_type n)
        requires is_dynamic
        : store_(n)
    {
    }

    /// Constructs a dynamic array of ``n`` elements, each set to ``value``.
    ///
    /// :param n: Number of logical elements.
    /// :param value: Value written to every element.
    Array(size_type n, T value)
        requires is_dynamic
        : store_(n)
    {
        fill(value);
    }

    /// Constructs a dynamic array holding a copy of ``init``.
    ///
    /// The array's length is taken from the initializer list.
    Array(std::initializer_list<T> init)
        requires is_dynamic
        : store_(init.size())
    {
        std::copy(init.begin(), init.end(), data());
    }

    /// Constructs a fixed-size array from ``init``.
    ///
    /// :throws std::invalid_argument: if ``init`` does not have exactly ``N`` elements.
    Array(std::initializer_list<T> init)
        requires(!is_dynamic)
    {
        if (init.size() != N) {
            throw std::invalid_argument{"miscibility::instrument::Array size mismatch"};
        }
        std::copy(init.begin(), init.end(), data());
    }

    /// Unchecked element access.
    [[nodiscard]] T& operator[](size_type i) noexcept { return data()[i]; }
    /// Unchecked element access.
    [[nodiscard]] const T& operator[](size_type i) const noexcept { return data()[i]; }

    /// Bounds-checked element access.
    ///
    /// :throws std::out_of_range: if ``i`` is not less than :cpp:`size`.
    [[nodiscard]] T& at(size_type i)
    {
        if (i >= size()) {
            throw std::out_of_range{"miscibility::instrument::Array::at"};
        }
        return data()[i];
    }

    /// Bounds-checked element access.
    ///
    /// :throws std::out_of_range: if ``i`` is not less than :cpp:`size`.
    [[nodiscard]] const T& at(size_type i) const
    {
        if (i >= size()) {
            throw std::out_of_range{"miscibility::instrument::Array::at"};
        }
        return data()[i];
    }

    /// Pointer to the first element of the contiguous buffer.
    [[nodiscard]] T* data() noexcept { return store_.data(); }
    /// Pointer to the first element of the contiguous buffer.
    [[nodiscard]] const T* data() const noexcept { return store_.data(); }
    /// Number of logical elements.
    [[nodiscard]] size_type size() const noexcept { return store_.size(); }
    /// Number of allocated slots, including the zero-padded SIMD tail (``>= size``).
    [[nodiscard]] size_type capacity() const noexcept { return store_.capacity(); }
    /// True when the array has no logical elements.
    [[nodiscard]] bool empty() const noexcept { return size() == 0; }

    [[nodiscard]] iterator begin() noexcept { return data(); }
    /// Past-the-end iterator over the logical elements (stops at :cpp:`size`).
    [[nodiscard]] iterator end() noexcept { return data() + size(); }
    [[nodiscard]] const_iterator begin() const noexcept { return data(); }
    /// Past-the-end iterator over the logical elements (stops at :cpp:`size`).
    [[nodiscard]] const_iterator end() const noexcept { return data() + size(); }

    /// A span over the logical elements (length :cpp:`size`).
    [[nodiscard]] std::span<T> as_span() noexcept { return {data(), size()}; }
    /// A span over the logical elements (length :cpp:`size`).
    [[nodiscard]] std::span<const T> as_span() const noexcept { return {data(), size()}; }

    /// A span over the whole buffer including the zero-padded tail (length :cpp:`capacity`).
    [[nodiscard]] std::span<T> padded_span() noexcept { return {data(), capacity()}; }
    /// A span over the whole buffer including the zero-padded tail (length :cpp:`capacity`).
    [[nodiscard]] std::span<const T> padded_span() const noexcept { return {data(), capacity()}; }

    /// Re-zeros the padding slots between :cpp:`size` and :cpp:`capacity`.
    ///
    /// Call this after writing directly through :cpp:`data` past the logical
    /// length, to restore the invariant the SIMD kernels rely on.
    void zero_pad() noexcept { std::fill(data() + size(), data() + capacity(), T{}); }

    /// Sets every logical element to ``value`` and re-zeros the padding.
    void fill(T value) noexcept
    {
        std::fill_n(data(), size(), value);
        zero_pad();
    }

    /// Swaps contents with ``other``.
    void swap(Array& other) noexcept { std::swap(store_, other.store_); }
    /// Swaps the contents of two arrays.
    friend void swap(Array& a, Array& b) noexcept { a.swap(b); }

    /// Copies the elements of ``src`` into this array.
    ///
    /// :param src: Source array; may have a different ``N`` but must match in length.
    /// :throws std::invalid_argument: if ``src`` and this array differ in :cpp:`size`.
    template<std::size_t M> Array& copy(const Array<T, M>& src)
    {
        check_same_size(src.size());
        std::copy_n(src.data(), capacity(), data());
        return *this;
    }

    /// Multiplies every element by ``a`` in place (``x := a*x``).
    Array& scale(T a) noexcept
    {
        detail::scale<T>(data(), capacity(), a);
        return *this;
    }

    /// Adds a scaled array in place (``y := y + a*x``), the AXPY operation.
    ///
    /// :param a: Scalar multiplier applied to ``x``.
    /// :param x: Array to add; must match this array in length.
    /// :throws std::invalid_argument: if ``x`` and this array differ in :cpp:`size`.
    template<std::size_t M> Array& add_scaled(T a, const Array<T, M>& x)
    {
        check_same_size(x.size());
        detail::add_scaled<T>(data(), x.data(), capacity(), a);
        return *this;
    }

    /// Sum of all elements. (The zeroed padding does not affect the result.)
    [[nodiscard]] T sum() const noexcept { return detail::sum<T>(data(), capacity()); }

    /// Sum of the absolute values of all elements (the L1 norm).
    [[nodiscard]] T absolute_sum() const noexcept { return detail::absolute_sum<T>(data(), capacity()); }

    /// Index of the element with the largest absolute value.
    ///
    /// On ties the lowest index wins. An empty array reports :cpp:`size`.
    ///
    /// :returns: The index of the largest-magnitude element, or :cpp:`size` if empty.
    [[nodiscard]] size_type index_of_max_magnitude() const noexcept
    {
        if (empty()) {
            return size();
        }
        return detail::index_of_max_magnitude<T>(data(), capacity());
    }

    /// Largest absolute element value (the infinity norm).
    [[nodiscard]] T max_magnitude() const noexcept { return std::abs(data()[index_of_max_magnitude()]); }

    /// Applies a SIMD lambda to every element in place and re-zeros the padding.
    ///
    /// This is the general transform the named math methods are built on. ``f``
    /// is invoked with the active SIMD descriptor and a vector of elements, and
    /// returns the transformed vector:
    ///
    /// .. code-block:: cpp
    ///
    ///     namespace hn = hwy::HWY_NAMESPACE;
    ///     x.apply([](auto d, auto v) { return hn::Mul(v, v); });   // square each element
    ///
    /// :param f: Callable ``(descriptor, vector) -> vector`` applied across the buffer.
    template<class F> Array& apply(F f) noexcept
    {
        detail::map<T>(data(), capacity(), f);
        zero_pad();
        return *this;
    }

    /// Replaces each element with its absolute value.
    Array& abs() noexcept
    {
        return apply([](auto, auto v) { return detail::hn::Abs(v); });
    }

    /// Replaces each element with its square root.
    Array& sqrt() noexcept
    {
        return apply([](auto, auto v) { return detail::hn::Sqrt(v); });
    }

    /// Replaces each element ``x`` with ``e**x``.
    Array& exp() noexcept
    {
        return apply([](auto d, auto v) { return detail::hn::Exp(d, v); });
    }

    /// Replaces each element ``x`` with ``2**x``.
    Array& exp2() noexcept
    {
        return apply([](auto d, auto v) { return detail::hn::Exp2(d, v); });
    }

    /// Replaces each element ``x`` with ``e**x - 1``, accurate for small ``x``.
    Array& expm1() noexcept
    {
        return apply([](auto d, auto v) { return detail::hn::Expm1(d, v); });
    }

    /// Replaces each element with its natural logarithm.
    Array& log() noexcept
    {
        return apply([](auto d, auto v) { return detail::hn::Log(d, v); });
    }

    /// Replaces each element with its base-2 logarithm.
    Array& log2() noexcept
    {
        return apply([](auto d, auto v) { return detail::hn::Log2(d, v); });
    }

    /// Replaces each element with its base-10 logarithm.
    Array& log10() noexcept
    {
        return apply([](auto d, auto v) { return detail::hn::Log10(d, v); });
    }

    /// Replaces each element ``x`` with ``log(1 + x)``, accurate for small ``x``.
    Array& log1p() noexcept
    {
        return apply([](auto d, auto v) { return detail::hn::Log1p(d, v); });
    }

    /// Replaces each element with its sine (argument in radians).
    Array& sin() noexcept
    {
        return apply([](auto d, auto v) { return detail::hn::Sin(d, v); });
    }

    /// Replaces each element with its cosine (argument in radians).
    Array& cos() noexcept
    {
        return apply([](auto d, auto v) { return detail::hn::Cos(d, v); });
    }

    /// Replaces each element with its hyperbolic sine.
    Array& sinh() noexcept
    {
        return apply([](auto d, auto v) { return detail::hn::Sinh(d, v); });
    }

    /// Replaces each element with its hyperbolic tangent.
    Array& tanh() noexcept
    {
        return apply([](auto d, auto v) { return detail::hn::Tanh(d, v); });
    }

    /// Replaces each element with its arc sine (result in radians).
    Array& asin() noexcept
    {
        return apply([](auto d, auto v) { return detail::hn::Asin(d, v); });
    }

    /// Replaces each element with its arc cosine (result in radians).
    Array& acos() noexcept
    {
        return apply([](auto d, auto v) { return detail::hn::Acos(d, v); });
    }

    /// Replaces each element with its inverse hyperbolic sine.
    Array& asinh() noexcept
    {
        return apply([](auto d, auto v) { return detail::hn::Asinh(d, v); });
    }

    /// Replaces each element with its inverse hyperbolic cosine.
    Array& acosh() noexcept
    {
        return apply([](auto d, auto v) { return detail::hn::Acosh(d, v); });
    }

    /// Replaces each element with its arc tangent (result in radians).
    Array& atan() noexcept
    {
        return apply([](auto d, auto v) { return detail::hn::Atan(d, v); });
    }

    /// Replaces each element with its inverse hyperbolic tangent.
    Array& atanh() noexcept
    {
        return apply([](auto d, auto v) { return detail::hn::Atanh(d, v); });
    }

    /// Multiplies this array element-by-element by ``x`` (the Hadamard product).
    ///
    /// :param x: Array of factors; must match this array in length.
    /// :throws std::invalid_argument: if ``x`` and this array differ in :cpp:`size`.
    template<std::size_t M> Array& elementwise_product(const Array<T, M>& x)
    {
        check_same_size(x.size());
        detail::zip<T>(data(), x.data(), capacity(), [](auto, auto y, auto v) { return detail::hn::Mul(y, v); });
        zero_pad();
        return *this;
    }

    /// Divides this array element-by-element by ``x``.
    ///
    /// :param x: Array of divisors; must match this array in length.
    /// :throws std::invalid_argument: if ``x`` and this array differ in :cpp:`size`.
    template<std::size_t M> Array& elementwise_quotient(const Array<T, M>& x)
    {
        check_same_size(x.size());
        detail::zip<T>(data(), x.data(), capacity(), [](auto, auto y, auto v) { return detail::hn::Div(y, v); });
        zero_pad();
        return *this;
    }

    /// Scales the array by ``a`` (see :cpp:`scale`).
    Array& operator*=(T a) noexcept { return scale(a); }
    /// Divides the array by ``a``.
    Array& operator/=(T a) noexcept { return scale(T(1) / a); }

    /// Adds ``x`` element-wise (see :cpp:`add_scaled`).
    template<std::size_t M> Array& operator+=(const Array<T, M>& x) { return add_scaled(T(1), x); }
    /// Subtracts ``x`` element-wise.
    template<std::size_t M> Array& operator-=(const Array<T, M>& x) { return add_scaled(T(-1), x); }

private:
    void check_same_size(size_type other) const
    {
        if (other != size()) {
            throw std::invalid_argument{"miscibility::instrument::Array size mismatch"};
        }
    }

    storage store_{};
};

} // namespace miscibility::instrument
