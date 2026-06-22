#include <boost/ut.hpp>
#include <ginkgo/ginkgo.hpp>

int main()
{
    using namespace boost::ut;

    "ginkgo reference executor constructs"_test = [] {
        auto executor = gko::ReferenceExecutor::create();
        expect(executor != nullptr);
    };
}
