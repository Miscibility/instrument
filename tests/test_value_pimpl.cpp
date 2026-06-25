// test_value_pimpl.cpp -- exhaustive unit tests for instrument::value_pimpl<T>.
//
// value_pimpl<T> is the rule-of-zero heap pimpl: destruction and move dispatch
// through a per-T static vtable of function pointers, so the *owning* class needs
// no hand-written special members and can be destroyed/moved with T incomplete.
// sizeof == 2 pointers (object + vtable). Move leaves the source empty; copy is a
// deep clone via the vtable. These tests cover every special member, both the
// const and non-const accessors, swap, the empty/moved-from state, and the
// rule-of-zero discipline.

#include <boost/ut.hpp>
#include <type_traits>
#include <utility>

#include "instrument/value_pimpl.hpp"

namespace ip = instrument;

namespace {

// Instrumented payload (same shape as the other pimpl test files).
struct Stats {
    int ctor = 0;
    int dtor = 0;
    int copy = 0;
    int move = 0;
    int alive() const { return ctor + copy + move - dtor; }
};
Stats g_stats;

struct Tracked {
    int value;
    explicit Tracked(int v = 0) : value(v) { ++g_stats.ctor; }
    Tracked(int a, int b) : value(a + b) { ++g_stats.ctor; }
    Tracked(const Tracked& o) : value(o.value) { ++g_stats.copy; }
    Tracked(Tracked&& o) noexcept : value(o.value)
    {
        o.value = -1;
        ++g_stats.move;
    }
    Tracked& operator=(const Tracked&) = default;
    Tracked& operator=(Tracked&&) = default;
    ~Tracked() { ++g_stats.dtor; }
};

struct Probe {
    int f() { return 1; }
    int f() const { return 2; }
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

// Rule-of-zero owner: declares *no* special members, yet is fully copyable,
// movable and destructible purely because value_pimpl is. VImpl is complete by
// the time any VOwner is actually constructed.
struct VImpl {
    int x = 7;
};
struct VOwner {
    ip::value_pimpl<VImpl> impl;
};
static_assert(std::is_copy_constructible_v<VOwner>);
static_assert(std::is_move_constructible_v<VOwner>);
static_assert(std::is_copy_assignable_v<VOwner>);
static_assert(std::is_move_assignable_v<VOwner>);

} // namespace

int main()
{
    using namespace boost::ut;

    static_assert(sizeof(ip::value_pimpl<Tracked>) == 2 * sizeof(void*),
                  "value_pimpl carries an object pointer plus a vtable pointer");
    static_assert(std::is_same_v<ip::value_pimpl<Tracked>::element_type, Tracked>);

    suite<"value_pimpl.construct"> construct = [] {
        test("default construction makes an engaged, default-valued object") = [] {
            g_stats = Stats{};
            {
                ip::value_pimpl<Tracked> p;
                expect(static_cast<bool>(p));
                expect(p.get() != nullptr);
                expect(p->value == 0_i);
                expect(g_stats.ctor == 1_i);
            }
            expect(g_stats.alive() == 0_i) << "vtable destroy ran T's destructor";
            expect(g_stats.dtor == 1_i);
        };

        test("in_place forwards single and multiple arguments") = [] {
            g_stats = Stats{};
            ip::value_pimpl<Tracked> one{std::in_place, 7};
            ip::value_pimpl<Tracked> two{std::in_place, 20, 22};
            expect(one->value == 7_i);
            expect(two->value == 42_i);
            expect(g_stats.copy == 0_i);
            expect(g_stats.move == 0_i);
        };

        test("in_place perfect-forwards lvalues (copy) and rvalues (move)") = [] {
            g_stats = Stats{};
            Tracked lv{5};
            ip::value_pimpl<Tracked> from_lvalue{std::in_place, lv};
            expect(g_stats.copy == 1_i);
            ip::value_pimpl<Tracked> from_rvalue{std::in_place, std::move(lv)};
            expect(g_stats.move == 1_i);
            expect(from_lvalue->value == 5_i);
            expect(from_rvalue->value == 5_i);
        };
    };

    suite<"value_pimpl.access"> access = [] {
        test("get, operator* and operator-> reach the same object") = [] {
            ip::value_pimpl<Tracked> p{std::in_place, 99};
            expect(p.get() == &*p);
            expect(&p->value == &(*p).value);
            (*p).value = 3;
            expect(p->value == 3_i);
        };

        test("const access propagates const through the accessors") = [] {
            const ip::value_pimpl<Probe> cp{std::in_place};
            ip::value_pimpl<Probe> mp{std::in_place};
            expect(mp->f() == 1_i);
            expect(cp->f() == 2_i);
            expect((*cp).f() == 2_i);
            static_assert(std::is_same_v<decltype(cp.get()), const Probe*>);
            static_assert(std::is_same_v<decltype(mp.get()), Probe*>);
            // Clone Probe too so its vtable's clone slot is exercised, not just Tracked's.
            ip::value_pimpl<Probe> clone{cp};
            expect(clone->f() == 1_i) << "the clone is a fresh, non-const object";
        };

        test("operator bool reflects engaged state") = [] {
            ip::value_pimpl<Tracked> p;
            expect(static_cast<bool>(p));
            ip::value_pimpl<Tracked> moved{std::move(p)};
            expect(not static_cast<bool>(p));
            expect(static_cast<bool>(moved));
        };
    };

    suite<"value_pimpl.copy"> copy = [] {
        test("copy construction deep-clones through the vtable") = [] {
            g_stats = Stats{};
            ip::value_pimpl<Tracked> a{std::in_place, 10};
            ip::value_pimpl<Tracked> b{a};
            expect(g_stats.copy == 1_i) << "clone uses T's copy ctor";
            expect(a.get() != b.get());
            a->value = 999;
            expect(b->value == 10_i) << "clone is independent";
        };

        test("copy assignment deep-clones and is independent") = [] {
            ip::value_pimpl<Tracked> a{std::in_place, 1};
            ip::value_pimpl<Tracked> b{std::in_place, 2};
            b = a;
            expect(b->value == 1_i);
            expect(a.get() != b.get());
            a->value = 7;
            expect(b->value == 1_i);
        };

        test("copy assignment is self-assignment safe (this != &o guard)") = [] {
            g_stats = Stats{};
            ip::value_pimpl<Tracked> a{std::in_place, 55};
            const auto& alias = a;
            a = alias;
            expect(static_cast<bool>(a));
            expect(a->value == 55_i);
            expect(g_stats.copy == 0_i) << "guard skips the clone entirely";
            expect(g_stats.dtor == 0_i) << "guard skips destroying the live object";
        };

        test("copying and copy-assigning an empty source stays empty") = [] {
            ip::value_pimpl<Tracked> empty;
            { ip::value_pimpl<Tracked> sink{std::move(empty)}; } // empty now empty
            expect(not static_cast<bool>(empty));
            ip::value_pimpl<Tracked> copy_of_empty{empty};
            expect(not static_cast<bool>(copy_of_empty)) << "clone of null is null";
            ip::value_pimpl<Tracked> target{std::in_place, 3};
            target = empty;
            expect(not static_cast<bool>(target));
        };

        test("copy-assigning a value into an empty target engages it") = [] {
            ip::value_pimpl<Tracked> empty;
            { ip::value_pimpl<Tracked> sink{std::move(empty)}; } // empty now empty
            expect(not static_cast<bool>(empty));
            ip::value_pimpl<Tracked> src{std::in_place, 9};
            empty = src; // exercises the "no live object to destroy" branch
            expect(static_cast<bool>(empty));
            expect(empty->value == 9_i);
            expect(empty.get() != src.get()) << "still a deep clone";
        };

        test("copy assignment offers the strong guarantee") = [] {
            Boom::armed = false;
            ip::value_pimpl<Boom> a{std::in_place, 1};
            ip::value_pimpl<Boom> b{std::in_place, 2};
            Boom::armed = true;
            expect(throws([&] { b = a; })) << "cloning the source throws";
            Boom::armed = false;
            expect(b->value == 2_i) << "target is unchanged: clone built before old is destroyed";
        };

        test("a throwing constructor propagates out of every constructor") = [] {
            Boom::armed = false;
            ip::value_pimpl<Boom> a{std::in_place, 1};
            Boom::armed = true;
            expect(throws([] { ip::value_pimpl<Boom> def; })) << "default constructor";
            expect(throws([] { ip::value_pimpl<Boom> ip{std::in_place, 5}; })) << "in_place constructor";
            expect(throws([&] { ip::value_pimpl<Boom> cp{a}; })) << "copy constructor";
            Boom::armed = false;
        };
    };

    suite<"value_pimpl.move"> move = [] {
        test("move construction steals the object and empties the source") = [] {
            g_stats = Stats{};
            ip::value_pimpl<Tracked> src{std::in_place, 8};
            Tracked* raw = src.get();
            ip::value_pimpl<Tracked> dst{std::move(src)};
            expect(dst.get() == raw) << "object pointer stolen";
            expect(not static_cast<bool>(src));
            expect(dst->value == 8_i);
            expect(g_stats.copy == 0_i);
            expect(g_stats.move == 0_i) << "no payload copy or move, just pointers";
        };

        test("move assignment steals the object, empties source, frees old target") = [] {
            g_stats = Stats{};
            ip::value_pimpl<Tracked> src{std::in_place, 11};
            ip::value_pimpl<Tracked> dst{std::in_place, 22};
            Tracked* raw = src.get();
            dst = std::move(src);
            expect(dst.get() == raw);
            expect(dst->value == 11_i);
            expect(not static_cast<bool>(src));
            expect(g_stats.dtor == 1_i) << "the overwritten target object was destroyed";
            expect(g_stats.copy == 0_i);
        };

        test("move assignment is self-assignment safe (this != &o guard)") = [] {
            ip::value_pimpl<Tracked> a{std::in_place, 77};
            auto& alias = a;
            a = std::move(alias);
            expect(static_cast<bool>(a)) << "self-move must not destroy the object";
            expect(a->value == 77_i);
        };

        test("move-assigning into an empty target does not double free") = [] {
            ip::value_pimpl<Tracked> empty;
            { ip::value_pimpl<Tracked> sink{std::move(empty)}; } // empty now empty
            ip::value_pimpl<Tracked> src{std::in_place, 4};
            empty = std::move(src);
            expect(static_cast<bool>(empty));
            expect(empty->value == 4_i);
            expect(not static_cast<bool>(src));
        };
    };

    suite<"value_pimpl.swap"> swap = [] {
        test("member swap exchanges objects and vtables") = [] {
            ip::value_pimpl<Tracked> a{std::in_place, 1};
            ip::value_pimpl<Tracked> b{std::in_place, 2};
            Tracked* pa = a.get();
            Tracked* pb = b.get();
            a.swap(b);
            expect(a.get() == pb);
            expect(b.get() == pa);
            expect(a->value == 2_i);
            expect(b->value == 1_i);
        };

        test("ADL free swap works, including against an empty pimpl") = [] {
            ip::value_pimpl<Tracked> a{std::in_place, 5};
            ip::value_pimpl<Tracked> b;
            { ip::value_pimpl<Tracked> sink{std::move(b)}; } // b empty
            using std::swap;
            swap(a, b);
            expect(not static_cast<bool>(a));
            expect(static_cast<bool>(b));
            expect(b->value == 5_i);
        };
    };

    suite<"value_pimpl.rule_of_zero"> roz = [] {
        test("an owner with no declared special members is fully regular") = [] {
            VOwner a;
            expect(a.impl->x == 7_i);
            a.impl->x = 50;
            VOwner b{a}; // implicit deep copy via value_pimpl's copy
            a.impl->x = 51;
            expect(b.impl->x == 50_i) << "copy is independent";
            VOwner c{std::move(b)}; // implicit move
            expect(c.impl->x == 50_i);
            VOwner d;
            d = c; // implicit copy assign
            expect(d.impl->x == 50_i);
            d = std::move(c); // implicit move assign
            expect(d.impl->x == 50_i);
        };
    };

    return ::boost::ut::cfg<>.run();
}
