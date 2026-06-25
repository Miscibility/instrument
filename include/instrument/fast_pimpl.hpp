#ifndef INSTRUMENT_FAST_PIMPL_HPP
#define INSTRUMENT_FAST_PIMPL_HPP

#include <cstddef>
#include <new>
#include <type_traits>
#include <utility>

namespace instrument {
/// A pimpl that stores its implementation inline, with no heap allocation.
///
/// ``FastPimple<T, Size, Align>`` holds the ``T`` directly in an aligned byte
/// buffer inside the object, so access costs no pointer indirection and
/// construction costs no allocation. The price is that you must size the buffer
/// up front: because ``T`` is incomplete in this header, you supply ``Size`` (and
/// optionally ``Align``) yourself, and they are checked against the real
/// ``sizeof(T)`` / ``alignof(T)`` by static assertions that fire in the
/// translation unit where ``T`` is complete. If ``Size`` or ``Align`` is too
/// small, the build fails there with a clear message.
///
/// Unlike :cpp:`Pimpl` and :cpp:`ValuePimpl`, a ``FastPimple`` is always engaged:
/// there is no empty or moved-from-null state, so it has no ``operator bool``. A
/// move leaves the source holding a moved-from ``T`` rather than emptying it.
/// Copy, move, and assignment simply mirror the corresponding operations on
/// ``T``, including their ``noexcept`` status.
///
/// .. code-block:: cpp
///
///     // wrapper.hpp — Impl is incomplete; you promise it fits in 64 bytes
///     class Wrapper {
///         struct Impl;
///         instrument::FastPimple<Impl, 64> impl_;
///     };
///
///     // wrapper.cpp — the size/alignment promise is verified here
///     struct Wrapper::Impl { /* ... */ };
///
/// :tparam T: Implementation type stored inline; may be incomplete in this header.
/// :tparam Size: Size in bytes reserved for ``T``; must be at least ``sizeof(T)``.
/// :tparam Align: Alignment reserved for ``T``; must be at least ``alignof(T)``.
template<class T, std::size_t Size, std::size_t Align = alignof(std::max_align_t)> class FastPimple {
public:
    /// The wrapped type.
    using element_type = T;

    /// Default-constructs a ``T`` in the inline buffer.
    FastPimple();

    /// Constructs the held ``T`` in place from ``args``.
    ///
    /// The ``std::in_place`` tag disambiguates this from the copy and move
    /// constructors. Arguments are perfect-forwarded to ``T``'s constructor.
    ///
    /// :param args: Arguments forwarded to the constructor of ``T``.
    /// :tparam Args: Types of the forwarded constructor arguments.
    template<class... Args, class = std::enable_if_t<std::is_constructible_v<T, Args...>>>
    explicit FastPimple(std::in_place_t /*unused*/, Args&&... args);

    /// Copy-constructs the held ``T`` from ``o``'s into this object's buffer.
    FastPimple(const FastPimple& o) noexcept(std::is_nothrow_copy_constructible_v<T>);
    /// Move-constructs the held ``T`` from ``o``'s; ``o`` keeps a moved-from ``T``.
    FastPimple(FastPimple&& o) noexcept(std::is_nothrow_move_constructible_v<T>);
    /// Copy-assigns the held ``T`` from ``o``'s.
    FastPimple& operator=(const FastPimple& o) noexcept(std::is_nothrow_copy_assignable_v<T>);
    /// Move-assigns the held ``T`` from ``o``'s; ``o`` keeps a moved-from ``T``.
    FastPimple& operator=(FastPimple&& o) noexcept(std::is_nothrow_move_assignable_v<T>);
    ~FastPimple();

    /// Returns a pointer to the held object.
    T* get() noexcept { return ptr(); }
    /// Returns a ``const`` pointer to the held object.
    const T* get() const noexcept { return ptr(); }
    /// Returns a reference to the held object.
    T& operator*() noexcept { return *ptr(); }
    /// Returns a ``const`` reference to the held object.
    const T& operator*() const noexcept { return *ptr(); }
    /// Accesses members of the held object.
    T* operator->() noexcept { return ptr(); }
    /// Accesses members of the held ``const`` object.
    const T* operator->() const noexcept { return ptr(); }

    /// Swaps the held objects by swapping the underlying ``T`` values.
    void swap(FastPimple& other) noexcept(std::is_nothrow_swappable_v<T>)
    {
        using std::swap;
        swap(*ptr(), *other.ptr());
    }
    /// Swaps two ``FastPimple`` objects by swapping their held ``T`` values.
    ///
    /// :relates: FastPimple
    friend void swap(FastPimple& a, FastPimple& b) noexcept(noexcept(a.swap(b))) { a.swap(b); }

private:
    T* ptr() noexcept { return std::launder(reinterpret_cast<T*>(&storage_)); }
    const T* ptr() const noexcept { return std::launder(reinterpret_cast<const T*>(&storage_)); }

    // Instantiated only inside the member bodies below, i.e. only where T is complete.
    static void validate() noexcept
    {
        static_assert(Size >= sizeof(T), "FastPimple: declared Size is too small for T");
        static_assert(Align >= alignof(T), "FastPimple: declared Align is too small for T");
        // To discover the real numbers on a failure, momentarily uncomment:
        //   reveal<sizeof(T), alignof(T)> _; // error prints reveal<ACTUAL_SIZE, ACTUAL_ALIGN>
    }

    alignas(Align) std::byte storage_[Size]; // NOLINT
};

/// Diagnostic helper that surfaces a type's real size and alignment in an error message.
///
/// It is declared but never defined, so naming it in a context that requires a
/// complete type produces an error spelling out the template arguments. To learn
/// the actual ``sizeof(T)`` and ``alignof(T)`` behind a :cpp:`FastPimple` sizing
/// failure, momentarily instantiate ``reveal<sizeof(T), alignof(T)>`` where ``T``
/// is complete and read the numbers out of the compiler's diagnostic.
///
/// :tparam Size: Size value to report.
/// :tparam Align: Alignment value to report.
template<std::size_t Size, std::size_t Align> struct reveal;

// NOLINTBEGIN(cppcoreguidelines-pro-type-member-init)
template<class T, std::size_t S, std::size_t A> FastPimple<T, S, A>::FastPimple()
{
    validate();
    ::new (static_cast<void*>(&storage_)) T();
}

template<class T, std::size_t S, std::size_t A>
template<class... Args, class>
FastPimple<T, S, A>::FastPimple(std::in_place_t /*unused*/, Args&&... args)
{
    validate();
    ::new (static_cast<void*>(&storage_)) T(std::forward<Args>(args)...);
}

template<class T, std::size_t S, std::size_t A>
FastPimple<T, S, A>::FastPimple(const FastPimple& o) noexcept(std::is_nothrow_copy_constructible_v<T>)
{
    validate();
    ::new (static_cast<void*>(&storage_)) T(*o.ptr());
}

template<class T, std::size_t S, std::size_t A>
FastPimple<T, S, A>::FastPimple(FastPimple&& o) noexcept(std::is_nothrow_move_constructible_v<T>)
{
    validate();
    ::new (static_cast<void*>(&storage_)) T(std::move(*o.ptr()));
}
// NOLINTEND(cppcoreguidelines-pro-type-member-init)

template<class T, std::size_t S, std::size_t A>
FastPimple<T, S, A>& FastPimple<T, S, A>::operator=(const FastPimple& o) noexcept(std::is_nothrow_copy_assignable_v<T>)
{
    if (this != &o) {
        *ptr() = *o.ptr();
    }
    return *this;
}

template<class T, std::size_t S, std::size_t A>
FastPimple<T, S, A>& FastPimple<T, S, A>::operator=(FastPimple&& o) noexcept(std::is_nothrow_move_assignable_v<T>)
{
    if (this != &o) {
        *ptr() = std::move(*o.ptr());
    }
    return *this;
}

template<class T, std::size_t S, std::size_t A> FastPimple<T, S, A>::~FastPimple()
{
    validate();
    ptr()->~T();
}

} // namespace instrument

#endif
