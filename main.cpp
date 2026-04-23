#include <benchmark/benchmark.h>

namespace
{

    void Test(benchmark::State& state)
    {
        for (auto idx : state)
        {
        }
    }

} // namespace

BENCHMARK(Test)->Threads(1)->Iterations(100000)->Name("Test name");

auto main(int argc, char** argv) -> int
{
    benchmark::MaybeReenterWithoutASLR(argc, argv);
    benchmark::Initialize(&argc, argv);
    benchmark::RunSpecifiedBenchmarks();
    benchmark::Shutdown();
    return 0;

    return 0;
}
