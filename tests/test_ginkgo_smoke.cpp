#include <ginkgo/ginkgo.hpp>

#include <boost/ut.hpp>

int main()
{
    using namespace boost::ut;

    "ginkgo reference executor constructs"_test = [] {
        auto executor = gko::ReferenceExecutor::create();
        expect(executor != nullptr);
    };
}
