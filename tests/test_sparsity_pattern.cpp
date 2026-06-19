// test_sparsity_pattern.cpp -- unit tests for miscibility::instrument SparsityPattern.

#include "instrument/sparsity_pattern.hpp"

#include <boost/ut.hpp>
#include <ginkgo/ginkgo.hpp>

namespace mi = miscibility::instrument;

int main()
{
    using namespace boost::ut;

    suite<"SparsityPattern"> pattern_suite = [] {
        test("add_value accumulates repeated entries") = [] {
            mi::SparsityPattern<double> pattern{gko::dim<2>{3, 3}};
            pattern.add_value(0, 0, 1.0);
            pattern.add_value(0, 0, 4.0);
            expect(pattern.get_value(0, 0) == 5.0_d);
        };

        test("set_value overwrites an accumulated entry") = [] {
            mi::SparsityPattern<double> pattern{gko::dim<2>{3, 3}};
            pattern.add_value(1, 1, 7.0);
            pattern.set_value(1, 1, 2.0);
            expect(pattern.get_value(1, 1) == 2.0_d);
        };

        test("get_value of an unset entry returns zero") = [] {
            mi::SparsityPattern<double> pattern{gko::dim<2>{3, 3}};
            expect(pattern.get_value(2, 2) == 0.0_d);
        };

        test("num_nonzeros counts distinct stored entries") = [] {
            mi::SparsityPattern<double> pattern{gko::dim<2>{3, 3}};
            pattern.add_value(0, 0, 1.0);
            pattern.add_value(0, 0, 1.0); // same entry, accumulates
            pattern.add_value(1, 2, 3.0);
            expect(pattern.num_nonzeros() == 2_u);
        };

        test("size reports the constructed shape") = [] {
            mi::SparsityPattern<double> pattern{gko::dim<2>{4, 5}};
            expect(pattern.size() == gko::dim<2>{4, 5});
        };

        test("to_matrix_data is sorted row-major with unique, summed entries") = [] {
            mi::SparsityPattern<double> pattern{gko::dim<2>{3, 3}};
            pattern.add_value(2, 0, 1.0);
            pattern.add_value(0, 1, 2.0);
            pattern.add_value(0, 1, 3.0); // duplicate of (0,1) -> summed to 5
            pattern.add_value(1, 2, 4.0);
            auto data = pattern.to_matrix_data();

            expect(data.nonzeros.size() == 3_u) << "duplicate (0,1) collapses to one entry";
            // row-major ordering: (0,1), (1,2), (2,0)
            expect(data.nonzeros[0].row == 0_i);
            expect(data.nonzeros[0].column == 1_i);
            expect(data.nonzeros[0].value == 5.0_d);
            expect(data.nonzeros[1].row == 1_i);
            expect(data.nonzeros[1].column == 2_i);
            expect(data.nonzeros[2].row == 2_i);
            expect(data.nonzeros[2].column == 0_i);
        };

        test("explicitly inserted zero is retained") = [] {
            mi::SparsityPattern<double> pattern{gko::dim<2>{3, 3}};
            pattern.add_value(0, 1, 0.0);
            auto data = pattern.to_matrix_data();
            expect(data.nonzeros.size() == 1_u) << "structural zeros are not pruned";
            expect(data.nonzeros[0].value == 0.0_d);
        };

        test("out-of-range add_value throws std::out_of_range") = [] {
            mi::SparsityPattern<double> pattern{gko::dim<2>{3, 3}};
            expect(throws<std::out_of_range>([&] { pattern.add_value(3, 0, 1.0); }));
            expect(throws<std::out_of_range>([&] { pattern.add_value(0, 3, 1.0); }));
        };
    };

    return 0;
}
