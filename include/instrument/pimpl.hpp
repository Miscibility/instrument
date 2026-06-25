#ifndef INSTRUMENT_PIMPL_HPP
#define INSTRUMENT_PIMPL_HPP

#include <concepts>
#include <memory>
#include <utility>

namespace instrument {

/// A heap-allocated pimpl pointer with value semantics and const propagation.
///
/// ``Pimpl<T>`` keeps a ``T`` on the heap behind a single owning pointer, so
/// ``sizeof(Pimpl<T>) == sizeof(T*)``, while behaving like a value: copies are
/// deep, moves transfer ownership and leave the source empty, and ``const``
/// access to the wrapper yields ``const`` access to the pointee (the behavior of
/// ``std::experimental::propagate_const``).
///
/// It exists to serve the `pimpl <https://en.cppreference.com/cpp/language/pimpl>`__ idiom.
/// ``T`` may be *incomplete* everywhere in
/// this header, so a class can declare a ``Pimpl<T>`` member knowing only a
/// forward declaration of its implementation. The wrapper's special members are
/// defined out of line and instantiated only where they are used, which leaves
/// the owning class with the usual obligation: declare its own special members
/// and define them (``= default`` suffices) in the translation unit where ``T``
/// is complete.
///
/// .. code-block:: cpp
///
///     // wrapper.hpp — Impl is incomplete here
///     class Wrapper {
///     public:
///         Wrapper();
///         ~Wrapper();
///         Wrapper(Wrapper&&) noexcept;
///         Wrapper& operator=(Wrapper&&) noexcept;
///     private:
///         struct Impl;
///         instrument::Pimpl<Impl> impl_;
///     };
///
///     // wrapper.cpp — Impl is complete here, so the defaults can be emitted
///     struct Wrapper::Impl { /* ... */ };
///     Wrapper::Wrapper() = default;
///     Wrapper::~Wrapper() = default;
///     Wrapper::Wrapper(Wrapper&&) noexcept = default;
///     Wrapper& Wrapper::operator=(Wrapper&&) noexcept = default;
///
/// :tparam T: Implementation type held on the heap; may be incomplete in this header.
template<class T> class Pimpl {
public:
    /// The wrapped type.
    using element_type = T;
    /// Pointer to the wrapped type.
    using pointer = T*;
    /// Pointer to the ``const`` wrapped type.
    using const_pointer = const T*;

    /// Default-constructs a ``T`` on the heap.
    ///
    /// Participates only when ``T`` is default-constructible.
    Pimpl();

    /// Constructs the held ``T`` in place from ``args``.
    ///
    /// The ``std::in_place`` tag disambiguates this from the copy and move
    /// constructors. Arguments are perfect-forwarded to ``T``'s constructor.
    ///
    /// .. code-block:: cpp
    ///
    ///     instrument::Pimpl<std::string> p{std::in_place, 3, 'x'};  // "xxx"
    ///
    /// :param args: Arguments forwarded to the constructor of ``T``.
    /// :tparam Args: Types of the forwarded constructor arguments.
    template<class... Args>
        requires std::constructible_from<T, Args...> // SFINAE-correct, lazy
    explicit Pimpl(std::in_place_t /*unused*/, Args&&... args);

    /// Deep-copies ``other``, allocating a fresh ``T``.
    ///
    /// Copying an empty (moved-from) ``Pimpl`` yields an empty one. Requires
    /// ``T`` to be copy-constructible.
    Pimpl(const Pimpl& other);

    /// Replaces the held object with a deep copy of ``other``'s.
    ///
    /// Offers the strong exception guarantee: the new object is constructed
    /// before the old one is released, so a throwing copy leaves ``*this``
    /// unchanged. Requires ``T`` to be copy-constructible.
    Pimpl& operator=(const Pimpl& other);

    /// Transfers ownership from ``other``, which is left empty.
    Pimpl(Pimpl&& other) noexcept;

    /// Transfers ownership from ``other``, releasing the current object; ``other`` is left empty.
    Pimpl& operator=(Pimpl&& other) noexcept;

    ~Pimpl();

    /// Returns a pointer to the held object, or ``nullptr`` if empty.
    T* get() noexcept { return ptr_.get(); }
    /// Returns a ``const`` pointer to the held object, or ``nullptr`` if empty.
    const T* get() const noexcept { return ptr_.get(); }
    /// Returns a reference to the held object; the wrapper must be non-empty.
    T& operator*() noexcept { return *ptr_; }
    /// Returns a ``const`` reference to the held object; the wrapper must be non-empty.
    const T& operator*() const noexcept { return *ptr_; }
    /// Accesses members of the held object; the wrapper must be non-empty.
    T* operator->() noexcept { return ptr_.get(); }
    /// Accesses members of the held ``const`` object; the wrapper must be non-empty.
    const T* operator->() const noexcept { return ptr_.get(); }
    /// Tests whether the wrapper holds an object; it is empty only after being moved from.
    explicit operator bool() const noexcept { return static_cast<bool>(ptr_); }

    /// Swaps the held pointers with ``other`` in constant time.
    void swap(Pimpl& other) noexcept { ptr_.swap(other.ptr_); }
    /// Swaps two ``Pimpl`` objects in constant time.
    ///
    /// :relates: Pimpl
    friend void swap(Pimpl& a, Pimpl& b) noexcept { a.swap(b); }

private:
    std::unique_ptr<T> ptr_;
};

template<class T> Pimpl<T>::Pimpl()
{
    static_assert(std::default_initializable<T>,
                  "Pimpl<T>: T must be default-constructible to default-construct Pimpl<T>");
    ptr_ = std::make_unique<T>(); // NOLINT
}

template<class T>
template<class... Args>
    requires std::constructible_from<T, Args...>
Pimpl<T>::Pimpl(std::in_place_t /*unused*/, Args&&... args) : ptr_(std::make_unique<T>(std::forward<Args>(args)...))
{
}

template<class T> Pimpl<T>::Pimpl(const Pimpl& other)
{
    static_assert(std::copy_constructible<T>, "Pimpl<T>: T must be copy-constructible to copy Pimpl<T>");
    if (other.ptr_) {
        ptr_ = std::make_unique<T>(*other.ptr_);
    }
}

template<class T> Pimpl<T>& Pimpl<T>::operator=(const Pimpl& other)
{
    static_assert(std::copy_constructible<T>, "Pimpl<T>: T must be copy-constructible to copy-assign Pimpl<T>");
    if (this != &other) {
        ptr_ = other.ptr_ ? std::make_unique<T>(*other.ptr_) : nullptr; // strong guarantee
    }
    return *this;
}

template<class T> Pimpl<T>::Pimpl(Pimpl&&) noexcept = default;
template<class T> Pimpl<T>& Pimpl<T>::operator=(Pimpl&&) noexcept = default;
template<class T> Pimpl<T>::~Pimpl() = default;

} // namespace instrument
#endif