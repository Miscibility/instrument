// test_fast_pimpl.cpp -- exhaustive unit tests for instrument::FastPimple<T,Size,Align>.
//
// FastPimple stores the impl in an aligned byte buffer *inside* the object: no
// heap allocation and no indirection. It is always "engaged" — there is no empty
// or moved-from-null state — so move construction leaves the source as a valid
// moved-from T rather than emptying it, and copy/move/assign mirror T's own
// operations exactly. These tests cover the storage/alignment invariants, every
// special member, both const and non-const accessors, swap, and the always-engaged
// property.

#include "instrument/fast_pimpl.hpp"

#include <boost/ut.hpp>
#include <cstddef>
#include <type_traits>
#include <utility>

namespace ip = instrument;

namespace {

struct Stats {
    int ctor = 0;
    int dtor = 0;
    int copy = 0;
    int move = 0;
    int copy_assign = 0;
    int move_assign = 0;
    [[nodiscard]] int alive() const { return ctor + copy + move - dtor; }
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

struct Probe {
    int f() { return 1; }
    int f() const { return 2; }
};

using FP = ip::FastPimple<Tracked, sizeof(Tracked)>;
// A deliberately over-aligned instantiation to exercise the Align parameter.
using FPAligned = ip::FastPimple<Tracked, sizeof(Tracked), 64>;

} // namespace

int main()
{
    using namespace boost::ut;

    static_assert(std::is_same_v<FP::element_type, Tracked>);
    // No heap, no pointer indirection: the wrapper is exactly its buffer.
    static_assert(sizeof(FP) >= sizeof(Tracked));
    // The Align parameter drives the wrapper's alignment and (rounded) size.
    static_assert(alignof(FPAligned) == 64);
    static_assert(sizeof(FPAligned) == 64);
    // Always engaged: there is intentionally no operator bool / empty state.
    static_assert(not std::is_constructible_v<bool, FP>);

    suite<"FastPimple.storage"> storage = [] {
        test("the managed object lives inside the wrapper (no heap)") = [] {
            FP fp{std::in_place, 5};
            const auto* base = reinterpret_cast<const std::byte*>(&fp);
            const auto* obj = reinterpret_cast<const std::byte*>(fp.get());
            expect(obj >= base) << "object is not before the wrapper";
            expect(obj < base + sizeof(fp)) << "object is within the wrapper's footprint";
        };

        test("the managed object honours the requested alignment") = [] {
            FPAligned fp{std::in_place, 1};
            const auto addr = reinterpret_cast<std::uintptr_t>(fp.get());
            expect((addr % 64) == 0_u) << "object is 64-byte aligned";
        };
    };

    suite<"FastPimple.construct"> construct = [] {
        test("default construction builds an engaged, default-valued object") = [] {
            g_stats = Stats{};
            {
                FP fp;
                expect(fp.get() != nullptr);
                expect(fp->value == 0_i);
                expect(g_stats.ctor == 1_i);
            }
            expect(g_stats.alive() == 0_i) << "destructor ran T's destructor in place";
            expect(g_stats.dtor == 1_i);
        };

        test("in_place forwards single and multiple arguments") = [] {
            g_stats = Stats{};
            FP one{std::in_place, 7};
            FP two{std::in_place, 20, 22};
            expect(one->value == 7_i);
            expect(two->value == 42_i);
            expect(g_stats.copy == 0_i);
            expect(g_stats.move == 0_i);
        };

        test("in_place perfect-forwards lvalues (copy) and rvalues (move)") = [] {
            g_stats = Stats{};
            Tracked lv{5};
            FP from_lvalue{std::in_place, lv};
            expect(g_stats.copy == 1_i);
            FP from_rvalue{std::in_place, std::move(lv)};
            expect(g_stats.move == 1_i);
            expect(from_lvalue->value == 5_i);
            expect(from_rvalue->value == 5_i);
        };
    };

    suite<"FastPimple.access"> access = [] {
        test("get, operator* and operator-> reach the same object") = [] {
            FP fp{std::in_place, 99};
            expect(fp.get() == &*fp);
            expect(&fp->value == &(*fp).value);
            (*fp).value = 3;
            expect(fp->value == 3_i);
        };

        test("const access propagates const through the accessors") = [] {
            const ip::FastPimple<Probe, sizeof(Probe)> cp{std::in_place};
            ip::FastPimple<Probe, sizeof(Probe)> mp{std::in_place};
            expect(mp->f() == 1_i);
            expect(cp->f() == 2_i);
            expect((*cp).f() == 2_i);
            static_assert(std::is_same_v<decltype(cp.get()), const Probe*>);
            static_assert(std::is_same_v<decltype(mp.get()), Probe*>);
        };
    };

    suite<"FastPimple.copy"> copy = [] {
        test("copy construction copies the payload into its own buffer") = [] {
            g_stats = Stats{};
            FP a{std::in_place, 10};
            FP b{a};
            expect(g_stats.copy == 1_i) << "uses T's copy ctor";
            expect(a.get() != b.get()) << "separate in-object buffers";
            a->value = 999;
            expect(b->value == 10_i) << "copy is independent";
        };

        test("copy assignment mirrors T's copy assignment") = [] {
            g_stats = Stats{};
            FP a{std::in_place, 1};
            FP b{std::in_place, 2};
            b = a;
            expect(g_stats.copy_assign == 1_i) << "assigns onto the existing object";
            expect(b->value == 1_i);
            a->value = 7;
            expect(b->value == 1_i);
        };

        test("copy assignment is self-assignment safe (this != &o guard)") = [] {
            g_stats = Stats{};
            FP a{std::in_place, 55};
            const auto& alias = a;
            a = alias;
            expect(a->value == 55_i);
            expect(g_stats.copy_assign == 0_i) << "guard skips the assignment";
        };
    };

    suite<"FastPimple.move"> move = [] {
        test("move construction move-constructs T and leaves source engaged") = [] {
            g_stats = Stats{};
            FP src{std::in_place, 8};
            FP dst{std::move(src)};
            expect(dst->value == 8_i);
            expect(g_stats.move == 1_i) << "uses T's move ctor";
            // No null state: the source is still a valid (moved-from) object.
            expect(src.get() != nullptr) << "always engaged, never null";
            expect(src->value == -1_i) << "source is T's moved-from value";
        };

        test("move assignment mirrors T's move assignment") = [] {
            g_stats = Stats{};
            FP src{std::in_place, 11};
            FP dst{std::in_place, 22};
            dst = std::move(src);
            expect(dst->value == 11_i);
            expect(g_stats.move_assign == 1_i);
            expect(src.get() != nullptr) << "source remains engaged";
            expect(src->value == -1_i);
        };

        test("move assignment is self-assignment safe (this != &o guard)") = [] {
            g_stats = Stats{};
            FP a{std::in_place, 77};
            auto& alias = a;
            a = std::move(alias);
            expect(a->value == 77_i) << "self-move leaves the object intact";
            expect(g_stats.move_assign == 0_i) << "guard skips the assignment";
        };
    };

    suite<"FastPimple.swap"> swap = [] {
        test("member swap exchanges the payloads in place") = [] {
            FP a{std::in_place, 1};
            FP b{std::in_place, 2};
            Tracked* pa = a.get();
            Tracked* pb = b.get();
            a.swap(b);
            // Buffers stay put (no pointer steal); only the contents are swapped.
            expect(a.get() == pa) << "in-object storage does not move";
            expect(b.get() == pb);
            expect(a->value == 2_i);
            expect(b->value == 1_i);
        };

        test("ADL free swap exchanges the payloads") = [] {
            FP a{std::in_place, 10};
            FP b{std::in_place, 20};
            using std::swap;
            swap(a, b);
            expect(a->value == 20_i);
            expect(b->value == 10_i);
        };
    };

    return ::boost::ut::cfg<>.run();
}
