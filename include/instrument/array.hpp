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

template<class T>
concept Scalar = std::floating_point<T>;

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

template<Scalar T, std::size_t N = dynamic> class Array {
    using storage = detail::storage_t<T, N>;

public:
    using value_type = T;
    using size_type = std::size_t;
    using iterator = T*;
    using const_iterator = const T*;

    static constexpr bool is_dynamic = (N == dynamic);

    Array() = default;

    explicit Array(size_type n)
        requires is_dynamic
        : store_(n)
    {
    }

    Array(size_type n, T value)
        requires is_dynamic
        : store_(n)
    {
        fill(value);
    }

    Array(std::initializer_list<T> init)
        requires is_dynamic
        : store_(init.size())
    {
        std::copy(init.begin(), init.end(), data());
    }

    Array(std::initializer_list<T> init)
        requires(!is_dynamic)
    {
        if (init.size() != N) {
            throw std::invalid_argument{"miscibility::instrument::Array size mismatch"};
        }
        std::copy(init.begin(), init.end(), data());
    }

    [[nodiscard]] T& operator[](size_type i) noexcept { return data()[i]; }
    [[nodiscard]] const T& operator[](size_type i) const noexcept { return data()[i]; }

    [[nodiscard]] T& at(size_type i)
    {
        if (i >= size()) {
            throw std::out_of_range{"miscibility::instrument::Array::at"};
        }
        return data()[i];
    }

    [[nodiscard]] const T& at(size_type i) const
    {
        if (i >= size()) {
            throw std::out_of_range{"miscibility::instrument::Array::at"};
        }
        return data()[i];
    }

    [[nodiscard]] T* data() noexcept { return store_.data(); }
    [[nodiscard]] const T* data() const noexcept { return store_.data(); }
    [[nodiscard]] size_type size() const noexcept { return store_.size(); }
    [[nodiscard]] size_type capacity() const noexcept { return store_.capacity(); }
    [[nodiscard]] bool empty() const noexcept { return size() == 0; }

    [[nodiscard]] iterator begin() noexcept { return data(); }
    [[nodiscard]] iterator end() noexcept { return data() + size(); }
    [[nodiscard]] const_iterator begin() const noexcept { return data(); }
    [[nodiscard]] const_iterator end() const noexcept { return data() + size(); }

    [[nodiscard]] std::span<T> as_span() noexcept { return {data(), size()}; }
    [[nodiscard]] std::span<const T> as_span() const noexcept { return {data(), size()}; }

    [[nodiscard]] std::span<T> padded_span() noexcept { return {data(), capacity()}; }
    [[nodiscard]] std::span<const T> padded_span() const noexcept { return {data(), capacity()}; }

    void zero_pad() noexcept { std::fill(data() + size(), data() + capacity(), T{}); }

    void fill(T value) noexcept
    {
        std::fill_n(data(), size(), value);
        zero_pad();
    }

    void swap(Array& other) noexcept { std::swap(store_, other.store_); }
    friend void swap(Array& a, Array& b) noexcept { a.swap(b); }

    template<std::size_t M> Array& copy(const Array<T, M>& src)
    {
        check_same_size(src.size());
        std::copy_n(src.data(), capacity(), data());
        return *this;
    }

    Array& scale(T a) noexcept
    {
        detail::scale<T>(data(), capacity(), a);
        return *this;
    }

    template<std::size_t M> Array& add_scaled(T a, const Array<T, M>& x)
    {
        check_same_size(x.size());
        detail::add_scaled<T>(data(), x.data(), capacity(), a);
        return *this;
    }

    [[nodiscard]] T sum() const noexcept { return detail::sum<T>(data(), capacity()); }

    [[nodiscard]] T absolute_sum() const noexcept { return detail::absolute_sum<T>(data(), capacity()); }

    [[nodiscard]] size_type index_of_max_magnitude() const noexcept
    {
        if (empty()) {
            return size();
        }
        return detail::index_of_max_magnitude<T>(data(), capacity());
    }

    [[nodiscard]] T max_magnitude() const noexcept { return std::abs(data()[index_of_max_magnitude()]); }

    template<class F> Array& apply(F f) noexcept
    {
        detail::map<T>(data(), capacity(), f);
        zero_pad();
        return *this;
    }

    Array& abs() noexcept
    {
        return apply([](auto, auto v) { return detail::hn::Abs(v); });
    }

    Array& sqrt() noexcept
    {
        return apply([](auto, auto v) { return detail::hn::Sqrt(v); });
    }

    Array& exp() noexcept
    {
        return apply([](auto d, auto v) { return detail::hn::Exp(d, v); });
    }

    Array& exp2() noexcept
    {
        return apply([](auto d, auto v) { return detail::hn::Exp2(d, v); });
    }

    Array& expm1() noexcept
    {
        return apply([](auto d, auto v) { return detail::hn::Expm1(d, v); });
    }

    Array& log() noexcept
    {
        return apply([](auto d, auto v) { return detail::hn::Log(d, v); });
    }

    Array& log2() noexcept
    {
        return apply([](auto d, auto v) { return detail::hn::Log2(d, v); });
    }

    Array& log10() noexcept
    {
        return apply([](auto d, auto v) { return detail::hn::Log10(d, v); });
    }

    Array& log1p() noexcept
    {
        return apply([](auto d, auto v) { return detail::hn::Log1p(d, v); });
    }

    Array& sin() noexcept
    {
        return apply([](auto d, auto v) { return detail::hn::Sin(d, v); });
    }

    Array& cos() noexcept
    {
        return apply([](auto d, auto v) { return detail::hn::Cos(d, v); });
    }

    Array& sinh() noexcept
    {
        return apply([](auto d, auto v) { return detail::hn::Sinh(d, v); });
    }

    Array& tanh() noexcept
    {
        return apply([](auto d, auto v) { return detail::hn::Tanh(d, v); });
    }

    Array& asin() noexcept
    {
        return apply([](auto d, auto v) { return detail::hn::Asin(d, v); });
    }

    Array& acos() noexcept
    {
        return apply([](auto d, auto v) { return detail::hn::Acos(d, v); });
    }

    Array& asinh() noexcept
    {
        return apply([](auto d, auto v) { return detail::hn::Asinh(d, v); });
    }

    Array& acosh() noexcept
    {
        return apply([](auto d, auto v) { return detail::hn::Acosh(d, v); });
    }

    Array& atan() noexcept
    {
        return apply([](auto d, auto v) { return detail::hn::Atan(d, v); });
    }

    Array& atanh() noexcept
    {
        return apply([](auto d, auto v) { return detail::hn::Atanh(d, v); });
    }

    template<std::size_t M> Array& elementwise_product(const Array<T, M>& x)
    {
        check_same_size(x.size());
        detail::zip<T>(data(), x.data(), capacity(), [](auto, auto y, auto v) { return detail::hn::Mul(y, v); });
        zero_pad();
        return *this;
    }

    template<std::size_t M> Array& elementwise_quotient(const Array<T, M>& x)
    {
        check_same_size(x.size());
        detail::zip<T>(data(), x.data(), capacity(), [](auto, auto y, auto v) { return detail::hn::Div(y, v); });
        zero_pad();
        return *this;
    }

    Array& operator*=(T a) noexcept { return scale(a); }
    Array& operator/=(T a) noexcept { return scale(T(1) / a); }

    template<std::size_t M> Array& operator+=(const Array<T, M>& x) { return add_scaled(T(1), x); }
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
