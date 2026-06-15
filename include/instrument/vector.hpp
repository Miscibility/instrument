/**
 * @file vector.hpp
 * @brief An aligned, padded numeric vector with a compile-time *or* runtime length.
 *
 * @code{.cpp}
 * miscibility::instrument::Vector<double, 128>      // static length, no heap
 * miscibility::instrument::Vector<double>           // runtime length, aligned heap buffer
 * miscibility::instrument::Vector<float, dynamic>   // == Vector<float> (explicit sentinel)
 * @endcode
 *
 * @par Storage guarantees
 * The buffer is over-aligned to `HWY_ALIGNMENT` (64 B), so every SIMD load/store on it is
 * aligned. The element *count* is padded up to a whole number of widest-vectors and the pad
 * is held at zero. Because the padding is a neutral element for the numeric kernels built on
 * top of this container, and `capacity()` is an exact multiple of the SIMD lane count, each
 * such kernel can be a single counter-driven loop with no scalar remainder ("peeling") tail:
 * @code{.cpp}
 * for (std::size_t i = 0; i < v.capacity(); i += Lanes(d)) { ... }
 * @endcode
 *
 * @par The zero-pad invariant
 * Between operations the pad slots `[size(), capacity())` are held at zero. Construction and
 * fill() establish it; zero_pad() restores it after any operation that may write non-zero
 * values into the pad (e.g. a future componentwise `exp`, where `exp(0) == 1`).
 *
 * @par Static vs dynamic storage
 * `dynamic` (the default extent) keeps the buffer on the heap and is the recommended choice
 * for anything but small, fixed lengths. A static extent stores the whole padded array inline,
 * so a large one risks stack exhaustion when the Vector is a local (e.g.
 * `Vector<double, 100'000>` is ~800 KB). Prefer dynamic for large vectors; reserve static for
 * small fixed sizes (coordinates, small block dimensions).
 *
 * @par Ownership
 * The heap buffer lives in a small storage helper that implements the Rule of 5, so Vector
 * itself follows the Rule of Zero -- its defaulted special members are correct.
 *
 * @par SIMD dispatch
 * The lane-multiple guarantee relies on Highway *static dispatch* (`hwy::HWY_NAMESPACE`,
 * compiled for the build's `-march` target). Do not use `HWY_DYNAMIC_DISPATCH`, whose widest
 * target could exceed `HWY_ALIGNMENT` bytes.
 *
 * @par Header-only
 * C++23.
 */

#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <initializer_list>
#include <limits>
#include <new>
#include <span>
#include <stdexcept>
#include <type_traits>
#include <utility>

#include <hwy/aligned_allocator.h> // HWY_ALIGNMENT
#include <hwy/highway.h>           // static dispatch: ScalableTag / Lanes

/// @namespace miscibility::instrument
/// @brief Instrument library for the Miscibility project; this header adds an aligned vector.
namespace miscibility::instrument {

/**
 * @brief Element types permitted in a Vector: IEEE floating-point scalars.
 * @tparam T Candidate element type.
 *
 * Restricted to `std::floating_point` because Highway lanes are well-defined for IEEE
 * float/double, which also keeps the numeric reductions built on this container (e.g.
 * euclidean norm, absolute sum) meaningful -- they rely on `Abs` and `sqrt`.
 */
template<class T>
concept Scalar = std::floating_point<T>;

/// @brief Sentinel extent selecting a runtime (heap-backed) length; alias for std::dynamic_extent.
inline constexpr std::size_t dynamic = std::dynamic_extent;

/// @namespace miscibility::instrument::detail
/// @brief Implementation details. Not part of the public API; documented for maintainers.
namespace detail {

/// @internal @brief Short alias for the active Highway static-dispatch namespace.
namespace hn = hwy::HWY_NAMESPACE;

/// @internal @brief Buffer alignment in bytes; every SIMD load/store on the buffer is aligned.
inline constexpr std::size_t alignment = HWY_ALIGNMENT; // 64 bytes

/**
 * @internal
 * @brief Element granularity the logical count is padded up to.
 * @tparam T Element type.
 *
 * Equals `alignment / sizeof(T)` (min 1). Because a full SIMD vector is at most `alignment`
 * bytes and both quantities are powers of two, a multiple of `pad_step<T>` elements is also a
 * whole multiple of `Lanes(d)` on every static-dispatch target.
 */
template<Scalar T>
inline constexpr std::size_t pad_step = (alignment / sizeof(T) == 0) ? 1 : alignment / sizeof(T);

/**
 * @internal
 * @brief Round a logical element count up to a whole multiple of pad_step<T>.
 * @tparam T Element type.
 * @param n Logical element count.
 * @return The padded capacity (a multiple of pad_step<T>, hence of the lane count).
 */
template<Scalar T> [[nodiscard]] constexpr std::size_t padded_count(std::size_t n) noexcept
{
    const std::size_t s = pad_step<T>;
    return ((n + s - 1) / s) * s;
}

// ---- BLAS-1 kernels --------------------------------------------------------
//
// Explicitly-vectorized numeric kernels operating over the padded capacity
// `cap`. Each is a single counter-driven loop over `[0, cap)` with no scalar
// remainder tail, relying on the zero-pad invariant: scale, add_scaled and dot
// all leave the pad at zero (0*a = 0, 0 + a*0 = 0, 0*0 = 0), so the invariant is
// self-maintaining for these operations. All are noexcept; the throwing size
// check lives in the Vector methods, before the kernel call.

/// @internal @brief In place scaling: `p <- a*p` over `[0, cap)`.
template<Scalar T> void scale(T* p, std::size_t cap, T a) noexcept
{
    const hn::ScalableTag<T> d;
    const std::size_t lanes = hn::Lanes(d);
    const auto va = hn::Set(d, a);
    for (std::size_t i = 0; i < cap; i += lanes) {
        hn::Store(hn::Mul(hn::Load(d, p + i), va), d, p + i);
    }
}

/// @internal @brief Scaled accumulate (axpy): `y <- y + a*x` over `[0, cap)`.
template<Scalar T> void add_scaled(T* y, const T* x, std::size_t cap, T a) noexcept
{
    const hn::ScalableTag<T> d;
    const std::size_t lanes = hn::Lanes(d);
    const auto va = hn::Set(d, a);
    for (std::size_t i = 0; i < cap; i += lanes) {
        const auto vy = hn::Load(d, y + i);
        const auto vx = hn::Load(d, x + i);
        hn::Store(hn::MulAdd(va, vx, vy), d, y + i); // a*x + y
    }
}

/// @internal @brief Dot product: `Sum a_i*b_i` over `[0, cap)` (MulAdd + ReduceSum).
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

/// @internal @brief Sum of squares: `Sum p_i^2` over `[0, cap)`.
template<Scalar T> [[nodiscard]] T sum_squares(const T* p, std::size_t cap) noexcept
{
    const hn::ScalableTag<T> d;
    const std::size_t lanes = hn::Lanes(d);
    auto acc = hn::Zero(d);
    for (std::size_t i = 0; i < cap; i += lanes) {
        const auto v = hn::Load(d, p + i);
        acc = hn::MulAdd(v, v, acc);
    }
    return hn::ReduceSum(d, acc);
}

/// @internal @brief Sum of magnitudes: `Sum |p_i|` over `[0, cap)`.
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

/// @internal @brief Index of the largest magnitude element, smallest index on ties.
///
/// Per lane, the running max is updated only on a strict `>` and indices are scanned in
/// increasing order, so the smallest index wins a tie within a column. The horizontal step
/// then takes the smallest index among the lanes holding the overall maximum magnitude. Pad
/// lanes are zero and the update is strict, so a pad slot can never displace a real element;
/// for an all-zero input every lane keeps its initial `Iota` index and the result is 0.
template<Scalar T>
[[nodiscard]] std::size_t index_of_max_magnitude(const T* p, std::size_t cap) noexcept
{
    const hn::ScalableTag<T> d;
    const hn::RebindToUnsigned<decltype(d)> du;
    using TU = hn::TFromD<decltype(du)>;
    const std::size_t lanes = hn::Lanes(d);

    auto best_mag = hn::Zero(d);
    auto best_idx = hn::Iota(du, 0); // {0, 1, ..., lanes-1}
    for (std::size_t i = 0; i < cap; i += lanes) {
        const auto mag = hn::Abs(hn::Load(d, p + i));
        const auto idx = hn::Iota(du, static_cast<TU>(i)); // {i, i+1, ...}
        const auto greater = hn::Gt(mag, best_mag);        // strict: ties keep the lower index
        best_mag = hn::IfThenElse(greater, mag, best_mag);
        best_idx = hn::IfThenElse(hn::RebindMask(du, greater), idx, best_idx);
    }

    // Horizontal: smallest index among lanes that hold the overall max magnitude.
    const auto max_mag = hn::Set(d, hn::ReduceMax(d, best_mag));
    const auto holds_max = hn::RebindMask(du, hn::Eq(best_mag, max_mag));
    const auto sentinel = hn::Set(du, std::numeric_limits<TU>::max());
    const auto candidates = hn::IfThenElse(holds_max, best_idx, sentinel);
    return static_cast<std::size_t>(hn::ReduceMin(du, candidates));
}

// ---- Storage helpers -------------------------------------------------------
//
// Two storage strategies, selected by extent. Each exposes the same surface
// (data(), size(), capacity()) so Vector's layer is written once.

/**
 * @internal
 * @brief Runtime-length storage: owns an aligned heap buffer (Rule of 5).
 * @tparam T Element type.
 *
 * The buffer is allocated with aligned `operator new`/`delete` (`std::align_val_t{alignment}`)
 * and the whole capacity is zero-initialized on construction so the pad starts neutral.
 */
template<Scalar T> class dynamic_storage {
public:
    /// @brief Construct an empty buffer (no allocation).
    dynamic_storage() noexcept = default;

    /// @brief Allocate a zero-filled, padded buffer for @p n logical elements.
    explicit dynamic_storage(std::size_t n) : size_{n}, capacity_{padded_count<T>(n)}
    {
        if (capacity_ != 0) {
            data_ = allocate(capacity_);
            std::fill_n(data_, capacity_, T{}); // begins lifetimes; zeros the pad
        }
    }

    /// @brief Deep-copy: allocate an independent buffer and copy every slot (incl. the pad).
    dynamic_storage(const dynamic_storage& o) : size_{o.size_}, capacity_{o.capacity_}
    {
        if (capacity_ != 0) {
            data_ = allocate(capacity_);
            std::copy_n(o.data_, capacity_, data_);
        }
    }

    /// @brief Steal @p o's buffer, leaving it empty.
    dynamic_storage(dynamic_storage&& o) noexcept
        : data_{std::exchange(o.data_, nullptr)}, size_{std::exchange(o.size_, std::size_t{0})},
          capacity_{std::exchange(o.capacity_, std::size_t{0})}
    {
    }

    /// @brief Copy-assign via copy-and-swap: strong exception guarantee, self-assignment safe.
    dynamic_storage& operator=(const dynamic_storage& o)
    {
        if (this != &o) {
            dynamic_storage tmp{o};
            swap(tmp);
        }
        return *this;
    }

    /// @brief Move-assign: release the current buffer and steal @p o's.
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

    /// @brief Release the buffer.
    ~dynamic_storage() { deallocate(data_, capacity_); }

    /// @brief Exchange buffers (and sizes) with @p o.
    void swap(dynamic_storage& o) noexcept
    {
        using std::swap;
        swap(data_, o.data_);
        swap(size_, o.size_);
        swap(capacity_, o.capacity_);
    }

    [[nodiscard]] T* data() noexcept { return data_; }              ///< Mutable buffer pointer.
    [[nodiscard]] const T* data() const noexcept { return data_; }  ///< Const buffer pointer.
    [[nodiscard]] std::size_t size() const noexcept { return size_; }         ///< Logical length.
    [[nodiscard]] std::size_t capacity() const noexcept { return capacity_; } ///< Padded length.

private:
    /// @brief Allocate @p cap aligned, uninitialized elements.
    static T* allocate(std::size_t cap)
    {
        return static_cast<T*>(::operator new(cap * sizeof(T), std::align_val_t{alignment}));
    }
    /// @brief Free an aligned buffer (no-op on nullptr).
    static void deallocate(T* p, std::size_t /*cap*/) noexcept
    {
        ::operator delete(p, std::align_val_t{alignment});
    }

    T* data_{nullptr};        ///< Owned aligned buffer, or nullptr when empty.
    std::size_t size_{0};     ///< Logical element count.
    std::size_t capacity_{0}; ///< Padded element count (a multiple of the lane count).
};

/**
 * @internal
 * @brief Compile-time-length storage: an aligned, padded, value-initialized array (Rule of Zero).
 * @tparam T Element type.
 * @tparam N Logical length (must be > 0; use dynamic for length 0 or a runtime length).
 *
 * Trivially copyable, so the defaulted special members are correct. The array is value-
 * initialized, so both the logical elements and the pad start at zero. Lives inline (on the
 * stack when the owning Vector is a local) -- keep @p N small.
 */
template<Scalar T, std::size_t N> class static_storage {
    static_assert(N > 0, "use miscibility::instrument::dynamic for length 0 / runtime length");

public:
    [[nodiscard]] T* data() noexcept { return data_; }             ///< Mutable buffer pointer.
    [[nodiscard]] const T* data() const noexcept { return data_; } ///< Const buffer pointer.
    [[nodiscard]] static constexpr std::size_t size() noexcept { return N; } ///< Logical length.
    [[nodiscard]] static constexpr std::size_t capacity() noexcept { return capacity_; } ///< Padded length.

private:
    static constexpr std::size_t capacity_ = padded_count<T>(N);
    alignas(alignment) T data_[capacity_]{}; // value-initialized: logical + pad are zero
};

/// @internal @brief Storage strategy selected by extent: heap for @c dynamic, inline otherwise.
template<Scalar T, std::size_t N>
using storage_t = std::conditional_t<N == dynamic, dynamic_storage<T>, static_storage<T, N>>;

} // namespace detail

// ----------------------------------------------------------------------------

/**
 * @brief An aligned, padded numeric vector with a compile-time or runtime length.
 * @tparam T Element type (an IEEE floating-point @ref Scalar).
 * @tparam N Logical length, or @c dynamic (the default) for a runtime, heap-backed length.
 *
 * The buffer is over-aligned to `detail::alignment` and `capacity()` is padded to a whole
 * multiple of the SIMD lane count, with the pad slots `[size(), capacity())` held at zero --
 * the invariant the library's numeric kernels rely on. See the file overview for details.
 *
 * Copy, move, and destruction follow the Rule of Zero: the storage member handles them, so a
 * dynamic Vector deep-copies and a static Vector copies its inline array.
 *
 * @code{.cpp}
 * miscibility::instrument::Vector<double> v(1000);   // 1000 zeros, heap-backed
 * v.fill(2.0);                                        // logical elements = 2, pad stays 0
 * miscibility::instrument::Vector<double, 3> p{1, 2, 3};   // fixed length, inline
 * @endcode
 */
template<Scalar T, std::size_t N = dynamic> class Vector {
    using storage = detail::storage_t<T, N>;

public:
    using value_type = T;                ///< Element type.
    using size_type = std::size_t;       ///< Length / index type.
    using iterator = T*;                 ///< Mutable contiguous iterator.
    using const_iterator = const T*;     ///< Const contiguous iterator.

    /// @brief True iff this is the runtime-length (heap-backed) specialization.
    static constexpr bool is_dynamic = (N == dynamic);

    // -- construction (Rule of Zero: storage handles copy/move/destroy) -------

    /// @brief Default: an empty dynamic vector, or a zero-filled fixed-length static vector.
    Vector() = default;

    /**
     * @brief Construct a runtime-length, zero-filled vector. Dynamic extent only.
     * @param n Logical length.
     */
    explicit Vector(size_type n)
        requires is_dynamic
        : store_(n)
    {
    }

    /**
     * @brief Construct a runtime-length vector with every logical element set to @p value.
     * @param n     Logical length.
     * @param value Value written to each logical element (pad stays zero). Dynamic extent only.
     */
    Vector(size_type n, T value)
        requires is_dynamic
        : store_(n)
    {
        fill(value);
    }

    /**
     * @brief Construct a runtime-length vector from a braced list. Dynamic extent only.
     * @param init Initial elements; the length becomes `init.size()`.
     */
    Vector(std::initializer_list<T> init)
        requires is_dynamic
        : store_(init.size())
    {
        std::copy(init.begin(), init.end(), data());
    }

    /**
     * @brief Construct a fixed-length vector from a braced list. Static extent only.
     * @param init Exactly @p N elements.
     * @throws std::invalid_argument if `init.size() != N`.
     */
    Vector(std::initializer_list<T> init)
        requires(!is_dynamic)
    {
        if (init.size() != N) {
            throw std::invalid_argument{"miscibility::instrument::Vector size mismatch"};
        }
        std::copy(init.begin(), init.end(), data());
    }

    // -- element access -------------------------------------------------------

    /// @brief Unchecked element access. @param i Index in `[0, size())`. @return Reference to element @p i.
    [[nodiscard]] T& operator[](size_type i) noexcept { return data()[i]; }
    /// @brief Unchecked const element access. @param i Index in `[0, size())`. @return Const reference to element @p i.
    [[nodiscard]] const T& operator[](size_type i) const noexcept { return data()[i]; }

    /**
     * @brief Bounds-checked element access.
     * @param i Index.
     * @return Reference to element @p i.
     * @throws std::out_of_range if `i >= size()`.
     */
    [[nodiscard]] T& at(size_type i)
    {
        if (i >= size()) {
            throw std::out_of_range{"miscibility::instrument::Vector::at"};
        }
        return data()[i];
    }
    /**
     * @brief Bounds-checked const element access.
     * @param i Index.
     * @return Const reference to element @p i.
     * @throws std::out_of_range if `i >= size()`.
     */
    [[nodiscard]] const T& at(size_type i) const
    {
        if (i >= size()) {
            throw std::out_of_range{"miscibility::instrument::Vector::at"};
        }
        return data()[i];
    }

    [[nodiscard]] T* data() noexcept { return store_.data(); }             ///< Pointer to the aligned buffer.
    [[nodiscard]] const T* data() const noexcept { return store_.data(); } ///< Const pointer to the aligned buffer.
    [[nodiscard]] size_type size() const noexcept { return store_.size(); } ///< Logical element count.
    /// @brief Padded element count: a whole multiple of the SIMD lane count, `>= size()`.
    [[nodiscard]] size_type capacity() const noexcept { return store_.capacity(); }
    [[nodiscard]] bool empty() const noexcept { return size() == 0; } ///< True iff `size() == 0`.

    [[nodiscard]] iterator begin() noexcept { return data(); }                  ///< Begin of the logical range.
    [[nodiscard]] iterator end() noexcept { return data() + size(); }           ///< End of the logical range.
    [[nodiscard]] const_iterator begin() const noexcept { return data(); }        ///< Const begin of the logical range.
    [[nodiscard]] const_iterator end() const noexcept { return data() + size(); } ///< Const end of the logical range.

    /// @brief View over the logical range `[0, size())` (excludes the pad). @return A span of the elements.
    [[nodiscard]] std::span<T> as_span() noexcept { return {data(), size()}; }
    /// @brief Const view over the logical range `[0, size())`. @return A const span of the elements.
    [[nodiscard]] std::span<const T> as_span() const noexcept { return {data(), size()}; }

    /// @brief View over the full padded buffer `[0, capacity())`, including the pad. @return A span of all slots.
    [[nodiscard]] std::span<T> padded_span() noexcept { return {data(), capacity()}; }
    /// @brief Const view over the full padded buffer `[0, capacity())`. @return A const span of all slots.
    [[nodiscard]] std::span<const T> padded_span() const noexcept { return {data(), capacity()}; }

    /**
     * @brief Reset the pad `[size(), capacity())` to zero, restoring the zero-pad invariant.
     *
     * Call this after any operation that may have written non-zero values into the pad (e.g. a
     * componentwise transform such as `exp`) and before a reduction that assumes a neutral pad.
     * Cheap: touches at most `detail::pad_step<T> - 1` elements.
     */
    void zero_pad() noexcept { std::fill(data() + size(), data() + capacity(), T{}); }

    /**
     * @brief Set every logical element to @p value while keeping the pad at zero.
     * @param value Value written to each element of `[0, size())`.
     */
    void fill(T value) noexcept
    {
        std::fill_n(data(), size(), value);
        zero_pad();
    }

    /// @brief Exchange contents with @p other. @param other Vector to swap with.
    void swap(Vector& other) noexcept { std::swap(store_, other.store_); }
    /// @brief ADL swap: exchange the contents of @p a and @p b.
    friend void swap(Vector& a, Vector& b) noexcept { a.swap(b); }

    // -- BLAS-1 numeric operations --------------------------------------------

    /// @brief In place scaling `x <- a*x` (scal). @param a Scale factor. @return *this.
    Vector& scale(T a) noexcept
    {
        detail::scale<T>(data(), capacity(), a);
        return *this;
    }

    /**
     * @brief Scaled accumulate `this <- this + a*x` (axpy).
     * @tparam M Extent of @p x (any matching-size extent is accepted).
     * @param a Scale factor applied to @p x.
     * @param x Operand of the same logical size as @c *this.
     * @return *this.
     * @throws std::invalid_argument if `x.size() != size()`.
     */
    template<std::size_t M> Vector& add_scaled(T a, const Vector<T, M>& x)
    {
        check_same_size(x.size());
        detail::add_scaled<T>(data(), x.data(), capacity(), a);
        return *this;
    }

    /**
     * @brief Dot product `Sum this_i * other_i` (dot).
     * @tparam M Extent of @p other (any matching-size extent is accepted).
     * @param other Operand of the same logical size as @c *this.
     * @return The dot product.
     * @throws std::invalid_argument if `other.size() != size()`.
     */
    template<std::size_t M> [[nodiscard]] T dot(const Vector<T, M>& other) const
    {
        check_same_size(other.size());
        return detail::dot<T>(data(), other.data(), capacity());
    }

    /// @brief Euclidean (L2) norm `sqrt(Sum this_i^2)` (nrm2). @return The norm.
    [[nodiscard]] T euclidean_norm() const noexcept
    {
        return std::sqrt(detail::sum_squares<T>(data(), capacity()));
    }

    /// @brief Sum of magnitudes `Sum |this_i|` (asum). @return The absolute sum.
    [[nodiscard]] T absolute_sum() const noexcept
    {
        return detail::absolute_sum<T>(data(), capacity());
    }

    /// @brief Index of the largest magnitude element, smallest index on ties; `size()` when empty (iamax).
    [[nodiscard]] size_type index_of_max_magnitude() const noexcept
    {
        if (empty()) {
            return size(); // no element
        }
        return detail::index_of_max_magnitude<T>(data(), capacity());
    }

    /// @brief Largest element magnitude `|at(index_of_max_magnitude())|`. @return The max magnitude.
    [[nodiscard]] T max_magnitude() const noexcept
    {
        return std::abs(data()[index_of_max_magnitude()]);
    }

    // -- convenience operators (built on the kernels above) -------------------

    /// @brief `x <- a*x`. @param a Scale factor. @return *this.
    Vector& operator*=(T a) noexcept { return scale(a); }
    /// @brief `x <- (1/a)*x`. @param a Divisor. @return *this.
    Vector& operator/=(T a) noexcept { return scale(T(1) / a); }

    /// @brief `this <- this + x`. @tparam M Extent of @p x. @param x Operand. @return *this.
    template<std::size_t M> Vector& operator+=(const Vector<T, M>& x) { return add_scaled(T(1), x); }
    /// @brief `this <- this - x`. @tparam M Extent of @p x. @param x Operand. @return *this.
    template<std::size_t M> Vector& operator-=(const Vector<T, M>& x) { return add_scaled(T(-1), x); }

private:
    /// @brief Throw if @p other differs from this vector's logical size.
    void check_same_size(size_type other) const
    {
        if (other != size()) {
            throw std::invalid_argument{"miscibility::instrument::Vector size mismatch"};
        }
    }

    storage store_{}; ///< Storage strategy (heap for dynamic, inline array for static).
};

} // namespace miscibility::instrument
