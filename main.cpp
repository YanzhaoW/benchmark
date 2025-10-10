#include "RootModel.hpp"
#include "TorchModel.hpp"
#include <benchmark/benchmark.h>
#include <torch/torch.h>

using std::make_unique;

namespace
{
    void TorchBM(benchmark::State& state)
    {
        Net model{ 100 };
        // std::println("creating model");
        auto x_vals = std::to_array<float>({ 67.5, 67.5, 47.5, 47.5, 57.5 });
        auto y_vals = std::to_array<float>({ 107.5, 102.5, -127.5, 117.5, 112.5 });
        for (auto idx : state)
        {
            const auto size = x_vals.size();
            const auto x_tensor = torch::from_blob(x_vals.data(), { size, 1 });
            const auto y_tensor = torch::from_blob(y_vals.data(), { size, 1 });
            model.clear();
            auto n_iter = model.train_from_data(x_tensor, y_tensor);
        }
        model.print();
    }

    void TorchAdam(benchmark::State& state)
    {
        Net model{ 100 };
        // std::println("creating model");
        auto x_vals = std::to_array<float>({ 67.5, 67.5, 47.5, 47.5, 57.5 });
        auto y_vals = std::to_array<float>({ 107.5, 102.5, -127.5, 117.5, 112.5 });
        for (auto idx : state)
        {
            const auto size = x_vals.size();
            const auto x_tensor = torch::from_blob(x_vals.data(), { size, 1 });
            const auto y_tensor = torch::from_blob(y_vals.data(), { size, 1 });
            model.clear();
            auto n_iter = model.train_from_data_adam(x_tensor, y_tensor);
        }
        model.print();
    }

    void RootBM(benchmark::State& state)
    {
        auto root_model = RootModel(1.35);
        std::vector<double> x_vals{ 47.5, 47.5, 57.5, 67.5, 67.5 };
        std::vector<double> y_vals{ -127.5, 117.5, 112.5, 107.5, 102.5 };

        for (auto idx : state)
        {
            root_model.clear();
            root_model.train_from_data(x_vals, y_vals);
        }
        root_model.print();
    }

} // namespace

// BENCHMARK(benchmark_function)->Name("testing");
BENCHMARK(TorchBM)->Threads(1)->Iterations(20)->Name("Using libtorch")->Unit(benchmark::kMillisecond);
BENCHMARK(TorchAdam)->Threads(1)->Iterations(20)->Name("Using adam")->Unit(benchmark::kMillisecond);
BENCHMARK(RootBM)->Threads(1)->Iterations(20)->Name("Using ROOT")->Unit(benchmark::kMillisecond);

// BENCHMARK_MAIN();

auto main(int argc, char** argv) -> int
{
    benchmark::MaybeReenterWithoutASLR(argc, argv);
    // auto mm = make_unique<benchmark::MemoryManager>();
    // benchmark::RegisterMemoryManager();
    benchmark::Initialize(&argc, argv);
    benchmark::RunSpecifiedBenchmarks();
    benchmark::Shutdown();
}
