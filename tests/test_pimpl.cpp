// test_pimpl.cpp -- exhaustive unit tests for instrument::Pimpl<T> (boost/ut).
//
// Pimpl<T> is the classic heap pimpl: sizeof(Pimpl<T>) == sizeof(T*), deep copy,
// pointer-stealing move (which leaves the source empty), const-propagating access,
// and — the whole point of the idiom — usability with an *incomplete* T in a class
// definition. The tests below cover every special member, every accessor in both
// const and non-const form, swap, the empty/moved-from state, and the
// incomplete-type discipline.

#include <boost/ut.hpp>
#include <type_traits>
#include <utility>

#include "instrument/pimpl.hpp"

namespace ip = instrument;

namespace {

// ---------------------------------------------------------------------------
// Instrumented payload: every special member bumps a global counter so a test
// can prove which of T's operations a Pimpl<T> operation actually triggered.
// ---------------------------------------------------------------------------
struct Stats {
    int ctor = 0;
    int dtor = 0;
    int copy = 0;
    int move = 0;
    int copy_assign = 0;
    int move_assign = 0;
    int alive() const { return ctor + copy + move - dtor; }
};
Stats g_stats;

struct Tracked {
    int value;
    explicit Tracked(int v = 0) : value(v) { ++g_stats.ctor; }
    Tracked(int a, int b) : value(a + b) { ++g_stats.ctor; } // multi-arg for in_place
    Tracked(const Tracked& o) : value(o.value) { ++g_stats.copy; }
    Tracked(Tracked&& o) noexcept : value(o.value)
    {
        o.value = -1;
        ++g_stats.move;
    }
    Tracked& operator=(const Tracked& o)
    {
        value = o.value;
        ++g_stats.copy_assign;
        return *this;
    }
    Tracked& operator=(Tracked&& o) noexcept
    {
        value = o.value;
        o.value = -1;
        ++g_stats.move_assign;
        return *this;
    }
    ~Tracked() { ++g_stats.dtor; }
};

// Distinguishes const from non-const access: proves const-propagation.
struct Probe {
    int f() { return 1; }       // chosen through a non-const Pimpl
    int f() const { return 2; } // chosen through a const Pimpl
};

// Throws from its constructors on demand, to exercise exception-safety paths.
struct Boom {
    static inline bool armed = false;
    int value;
    explicit Boom(int v = 0) : value(v)
    {
        if (armed) {
            throw 42;
        }
    }
    Boom(const Boom& o) : value(o.value)
    {
        if (armed) {
            throw 42;
        }
    }
};

// ---------------------------------------------------------------------------
// Incomplete-type discipline. OwnerImpl is incomplete at the point Owner is
// defined; this only compiles because Pimpl<OwnerImpl>'s special members are
// not instantiated there (sizeof == pointer). Owner declares its own special
// members and defines them out of line once OwnerImpl is complete.
// ---------------------------------------------------------------------------
struct OwnerImpl; // incomplete here

class Owner {
public:
    Owner();
    Owner(const Owner&);
    Owner& operator=(const Owner&);
    Owner(Owner&&) noexcept;
    Owner& operator=(Owner&&) noexcept;
    ~Owner();
    int value() const; // reads through const operator->
    void set(int v);   // writes through non-const operator->

private:
    ip::Pimpl<OwnerImpl> impl_;
};

struct OwnerImpl {
    int x = 41;
};

Owner::Owner() = default;
Owner::Owner(const Owner&) = default;
Owner& Owner::operator=(const Owner&) = default;
Owner::Owner(Owner&&) noexcept = default;
Owner& Owner::operator=(Owner&&) noexcept = default;
Owner::~Owner() = default;
int Owner::value() const { return impl_->x; }
void Owner::set(int v) { impl_->x = v; }

} // namespace

int main()
{
    using namespace boost::ut;

    // The size invariant is the defining property of a heap pimpl.
    static_assert(sizeof(ip::Pimpl<Tracked>) == sizeof(Tracked*));
    static_assert(std::is_same_v<ip::Pimpl<Tracked>::element_type, Tracked>);
    static_assert(std::is_same_v<ip::Pimpl<Tracked>::pointer, Tracked*>);
    static_assert(std::is_same_v<ip::Pimpl<Tracked>::const_pointer, const Tracked*>);

    suite<"Pimpl.construct"> construct = [] {
        test("default construction makes an engaged, default-valued object") = [] {
            g_stats = Stats{};
            {
                ip::Pimpl<Tracked> p;
                expect(static_cast<bool>(p));
                expect(p.get() != nullptr);
                expect(p->value == 0_i);
                expect(g_stats.ctor == 1_i);
                expect(g_stats.alive() == 1_i);
            }
            expect(g_stats.alive() == 0_i) << "destructor ran T's destructor";
            expect(g_stats.dtor == 1_i);
        };

        test("in_place forwards a single argument") = [] {
            g_stats = Stats{};
            ip::Pimpl<Tracked> p{std::in_place, 7};
            expect(p->value == 7_i);
            expect(g_stats.ctor == 1_i);
            expect(g_stats.copy == 0_i);
            expect(g_stats.move == 0_i);
        };

        test("in_place forwards multiple arguments") = [] {
            g_stats = Stats{};
            ip::Pimpl<Tracked> p{std::in_place, 20, 22};
            expect(p->value == 42_i);
            expect(g_stats.ctor == 1_i);
        };

        test("in_place perfect-forwards lvalues (copy) and rvalues (move)") = [] {
            g_stats = Stats{};
            Tracked lv{5};
            ip::Pimpl<Tracked> from_lvalue{std::in_place, lv};
            expect(from_lvalue->value == 5_i);
            expect(g_stats.copy == 1_i) << "lvalue selects T's copy ctor";
            ip::Pimpl<Tracked> from_rvalue{std::in_place, std::move(lv)};
            expect(from_rvalue->value == 5_i);
            expect(g_stats.move == 1_i) << "rvalue selects T's move ctor";
        };
    };

    suite<"Pimpl.access"> access = [] {
        test("get, operator* and operator-> reach the same object") = [] {
            ip::Pimpl<Tracked> p{std::in_place, 99};
            expect(p.get() == &*p);
            expect(&p->value == &(*p).value);
            (*p).value = 3;
            expect(p->value == 3_i);
        };

        test("const access is available and propagates const") = [] {
            const ip::Pimpl<Probe> cp{std::in_place};
            ip::Pimpl<Probe> mp{std::in_place};
            // Same object, but constness of the wrapper selects the overload.
            expect(mp->f() == 1_i) << "non-const Pimpl -> non-const member";
            expect(cp->f() == 2_i) << "const Pimpl -> const member (propagate_const)";
            expect((*cp).f() == 2_i);
            const Probe* cptr = cp.get();
            expect(cptr->f() == 2_i);
            static_assert(std::is_same_v<decltype(cp.get()), const Probe*>);
            static_assert(std::is_same_v<decltype(mp.get()), Probe*>);
        };

        test("operator bool reflects engaged state") = [] {
            ip::Pimpl<Tracked> p;
            expect(static_cast<bool>(p));
            ip::Pimpl<Tracked> moved{std::move(p)};
            expect(not static_cast<bool>(p)) << "moved-from is empty";
            expect(static_cast<bool>(moved));
        };
    };

    suite<"Pimpl.copy"> copy = [] {
        test("copy construction is a deep, independent copy") = [] {
            g_stats = Stats{};
            ip::Pimpl<Tracked> a{std::in_place, 10};
            ip::Pimpl<Tracked> b{a};
            expect(g_stats.copy == 1_i) << "deep copy uses T's copy ctor";
            expect(a.get() != b.get()) << "distinct storage";
            a->value = 999;
            expect(b->value == 10_i) << "mutating the source does not touch the copy";
        };

        test("copy assignment is deep and independent") = [] {
            ip::Pimpl<Tracked> a{std::in_place, 1};
            ip::Pimpl<Tracked> b{std::in_place, 2};
            b = a;
            expect(b->value == 1_i);
            expect(a.get() != b.get());
            a->value = 7;
            expect(b->value == 1_i);
        };

        test("copy assignment is self-assignment safe") = [] {
            ip::Pimpl<Tracked> a{std::in_place, 55};
            const auto& alias = a; // dodge -Wself-assign-overloaded
            a = alias;
            expect(static_cast<bool>(a));
            expect(a->value == 55_i);
        };

        test("copying an empty Pimpl yields an empty Pimpl") = [] {
            ip::Pimpl<Tracked> src;
            { ip::Pimpl<Tracked> sink{std::move(src)}; } // src is now the empty one
            expect(not static_cast<bool>(src));
            ip::Pimpl<Tracked> copy_of_empty{src};
            expect(not static_cast<bool>(copy_of_empty));
            expect(copy_of_empty.get() == nullptr);
        };

        test("copy-assigning from an empty source empties the target") = [] {
            ip::Pimpl<Tracked> empty;
            { ip::Pimpl<Tracked> sink{std::move(empty)}; } // empty is now empty
            ip::Pimpl<Tracked> target{std::in_place, 3};
            expect(static_cast<bool>(target));
            target = empty;
            expect(not static_cast<bool>(target)) << "deep-copy of null is null";
        };
    };

    suite<"Pimpl.move"> move = [] {
        test("move construction steals the pointer and empties the source") = [] {
            g_stats = Stats{};
            ip::Pimpl<Tracked> src{std::in_place, 8};
            Tracked* raw = src.get();
            ip::Pimpl<Tracked> dst{std::move(src)};
            expect(dst.get() == raw) << "pointer stolen, not copied";
            expect(not static_cast<bool>(src));
            expect(dst->value == 8_i);
            expect(g_stats.copy == 0_i) << "move must not copy the payload";
            expect(g_stats.move == 0_i) << "the payload itself is not moved, only the pointer";
        };

        test("move assignment steals the pointer and empties the source") = [] {
            g_stats = Stats{};
            ip::Pimpl<Tracked> src{std::in_place, 11};
            Tracked* raw = src.get();
            ip::Pimpl<Tracked> dst{std::in_place, 22};
            dst = std::move(src);
            expect(dst.get() == raw);
            expect(dst->value == 11_i);
            expect(not static_cast<bool>(src));
            expect(g_stats.copy == 0_i);
        };

        test("move assignment is self-assignment safe") = [] {
            ip::Pimpl<Tracked> a{std::in_place, 77};
            auto& alias = a; // dodge -Wself-move
            a = std::move(alias);
            expect(static_cast<bool>(a)) << "self-move must not destroy the object";
            expect(a->value == 77_i);
        };
    };

    suite<"Pimpl.exceptions"> exceptions = [] {
        test("a throwing constructor propagates out of every constructor") = [] {
            Boom::armed = false;
            ip::Pimpl<Boom> a{std::in_place, 1}; // built while disarmed
            Boom::armed = true;
            expect(throws([] { ip::Pimpl<Boom> def; })) << "default constructor";
            expect(throws([] { ip::Pimpl<Boom> ip{std::in_place, 5}; })) << "in_place constructor";
            expect(throws([&] { ip::Pimpl<Boom> cp{a}; })) << "copy constructor";
            Boom::armed = false;
        };

        test("copy assignment offers the strong guarantee") = [] {
            Boom::armed = false;
            ip::Pimpl<Boom> a{std::in_place, 1};
            ip::Pimpl<Boom> b{std::in_place, 2};
            Boom::armed = true;
            expect(throws([&] { b = a; })) << "cloning the source throws";
            Boom::armed = false;
            expect(b->value == 2_i) << "target is unchanged after the throw";
        };
    };

    suite<"Pimpl.swap"> swap = [] {
        test("member swap exchanges the managed objects") = [] {
            ip::Pimpl<Tracked> a{std::in_place, 1};
            ip::Pimpl<Tracked> b{std::in_place, 2};
            Tracked* pa = a.get();
            Tracked* pb = b.get();
            a.swap(b);
            expect(a.get() == pb);
            expect(b.get() == pa);
            expect(a->value == 2_i);
            expect(b->value == 1_i);
        };

        test("ADL free swap works and swaps with an empty Pimpl") = [] {
            ip::Pimpl<Tracked> a{std::in_place, 5};
            ip::Pimpl<Tracked> b;
            { ip::Pimpl<Tracked> sink{std::move(b)}; } // b empty
            using std::swap;
            swap(a, b);
            expect(not static_cast<bool>(a));
            expect(static_cast<bool>(b));
            expect(b->value == 5_i);
        };
    };

    suite<"Pimpl.incomplete_type"> incomplete = [] {
        test("Pimpl supports an owning class over an incomplete impl") = [] {
            Owner o;
            expect(o.value() == 41_i) << "reads through const operator->";
            o.set(100);
            expect(o.value() == 100_i);
        };

        test("the owning class deep-copies and moves correctly") = [] {
            Owner a;
            a.set(7);
            Owner b{a}; // deep copy
            a.set(8);
            expect(b.value() == 7_i) << "copy is independent";
            Owner c{std::move(b)}; // move
            expect(c.value() == 7_i);
            Owner d;
            d = c; // copy assign
            expect(d.value() == 7_i);
            d.set(9);
            expect(c.value() == 7_i);
            d = std::move(c); // move assign
            expect(d.value() == 7_i);
        };
    };

    return ::boost::ut::cfg<>.run();
}
