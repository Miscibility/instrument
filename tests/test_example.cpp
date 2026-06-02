#include <boost/ut.hpp>

int main(){
using namespace boost::ut;

"1 + 1 = 2"_test = [] {
    expect(1 + 1 == 2_i);
};
}
