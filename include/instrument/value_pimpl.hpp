#ifndef INSTRUMENT_VALUE_PIMPL_HPP
#define INSTRUMENT_VALUE_PIMPL_HPP

#include <type_traits>
#include <utility>

namespace instrument {
namespace detail {
struct vp_vtable {
    void (*destroy)(void*) noexcept;
    void* (*clone)(const void*);
};
// NOLINTBEGIN(cppcoreguidelines-owning-memory)
template<class T>
inline constexpr vp_vtable vp_vtable_for{[](void* p) noexcept { delete static_cast<T*>(p); },
                                         [](const void* p) -> void* { return new T(*static_cast<const T*>(p)); }};
// NOLINTEND(cppcoreguidelines-owning-memory)
} // namespace detail

/// A rule-of-zero heap pimpl pointer with value semantics.
///
/// Like :cpp:`Pimpl`, ``ValuePimpl<T>`` owns a ``T`` on the heap and copies it
/// deeply, but it routes destruction, cloning, and move through a per-``T`` table
/// of function pointers captured at construction. Because that table does the
/// type-specific work, the wrapper's own destructor and move never need ``T`` to
/// be complete — and so the *owning* class needs no hand-written special members
/// at all. It is genuinely rule-of-zero, and other translation units may default,
/// destroy, and move it while ``T`` is still incomplete. (Copying the owning class
/// still needs ``T`` complete in that translation unit, as with any deep-copying
/// pimpl.)
///
/// The trade-off against :cpp:`Pimpl` is size: a ``ValuePimpl`` carries both the
/// object pointer and the table pointer, so ``sizeof(ValuePimpl<T>) == 2 *
/// sizeof(void*)``. In exchange the owning class can be a pure aggregate:
///
/// .. code-block:: cpp
///
///     // wrapper.hpp — Impl is incomplete, yet Wrapper needs no special members
///     class Wrapper {
///         struct Impl;
///         instrument::ValuePimpl<Impl> impl_;
///     };
///
///     // wrapper.cpp
///     struct Wrapper::Impl { /* ... */ };
///
/// This is the spiritual ancestor of C++26's ``std::indirect<T>``; prefer that
/// when it is available to you.
///
/// :tparam T: Implementation type held on the heap; may be incomplete in this header.
template<class T> class ValuePimpl {
public:
    /// The wrapped type.
    using element_type = T;

    /// Default-constructs a ``T`` on the heap.
    ValuePimpl() : obj_(new T()), vt_(&detail::vp_vtable_for<T>) {}

    /// Constructs the held ``T`` in place from ``args``.
    ///
    /// The ``std::in_place`` tag disambiguates this from the copy and move
    /// constructors. Arguments are perfect-forwarded to ``T``'s constructor.
    ///
    /// :param args: Arguments forwarded to the constructor of ``T``.
    /// :tparam Args: Types of the forwarded constructor arguments.
    template<class... Args>
    explicit ValuePimpl(std::in_place_t /*unused*/, Args&&... args)
        requires(std::is_constructible_v<T, Args...>)
        : obj_(new T(std::forward<Args>(args)...)), vt_(&detail::vp_vtable_for<T>)
    {
    }

    /// Deep-copies ``o`` by cloning its held object; copying an empty wrapper yields an empty one.
    ValuePimpl(const ValuePimpl& o) : obj_(o.obj_ ? o.vt_->clone(o.obj_) : nullptr), vt_(o.vt_) {}

    /// Replaces the held object with a clone of ``o``'s.
    ///
    /// Offers the strong exception guarantee: the clone is built before the old
    /// object is destroyed, so a throwing copy leaves ``*this`` unchanged.
    ValuePimpl& operator=(const ValuePimpl& o)
    {
        if (this != &o) {
            void* n = o.obj_ ? o.vt_->clone(o.obj_) : nullptr; // construct first (strong)
            if (obj_) {
                vt_->destroy(obj_);
            }
            obj_ = n;
            vt_ = o.vt_;
        }
        return *this;
    }

    /// Transfers ownership from ``o``, which is left empty.
    ValuePimpl(ValuePimpl&& o) noexcept : obj_(o.obj_), vt_(o.vt_) { o.obj_ = nullptr; }

    /// Transfers ownership from ``o``, releasing the current object; ``o`` is left empty.
    ValuePimpl& operator=(ValuePimpl&& o) noexcept
    {
        if (this != &o) {
            if (obj_) {
                vt_->destroy(obj_);
            }
            obj_ = o.obj_;
            vt_ = o.vt_;
            o.obj_ = nullptr;
        }
        return *this;
    }

    ~ValuePimpl()
    {
        if (obj_) {
            vt_->destroy(obj_);
        }
    }

    /// Returns a pointer to the held object, or ``nullptr`` if empty.
    T* get() noexcept { return static_cast<T*>(obj_); }
    /// Returns a ``const`` pointer to the held object, or ``nullptr`` if empty.
    const T* get() const noexcept { return static_cast<const T*>(obj_); }
    /// Returns a reference to the held object; the wrapper must be non-empty.
    T& operator*() noexcept { return *get(); }
    /// Returns a ``const`` reference to the held object; the wrapper must be non-empty.
    const T& operator*() const noexcept { return *get(); }
    /// Accesses members of the held object; the wrapper must be non-empty.
    T* operator->() noexcept { return get(); }
    /// Accesses members of the held ``const`` object; the wrapper must be non-empty.
    const T* operator->() const noexcept { return get(); }
    /// Tests whether the wrapper holds an object; it is empty only after being moved from.
    explicit operator bool() const noexcept { return obj_ != nullptr; }

    /// Swaps the held object and table pointers with ``o`` in constant time.
    void swap(ValuePimpl& o) noexcept
    {
        std::swap(obj_, o.obj_);
        std::swap(vt_, o.vt_);
    }
    /// Swaps two ``ValuePimpl`` objects in constant time.
    ///
    /// :relates: ValuePimpl
    friend void swap(ValuePimpl& a, ValuePimpl& b) noexcept { a.swap(b); }

private:
    void* obj_;
    const detail::vp_vtable* vt_;
};
} // namespace instrument

#endif