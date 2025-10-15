#include <benchmark/benchmark.h>
#include <print>
#include <torch/torch.h>

namespace
{
    void TorchBM(benchmark::State& state)
    {
        auto x_vals = std::to_array<float>({ 67.5, 67.5, 47.5, 47.5, 57.5 });
        auto y_vals = std::to_array<float>({ 107.5, 102.5, -127.5, 117.5, 112.5 });
        const auto size = x_vals.size();
        const auto x_tensor = torch::from_blob(x_vals.data(), { size, 1 });
        const auto y_tensor = torch::from_blob(y_vals.data(), { size, 1 });
        auto sum = 0.F;
        for (auto idx : state)
        {
            auto z_vals = x_tensor * y_tensor;
            sum += z_vals.sum().item<float>();
        }
        std::println("sum: {}", sum);
    }
    void TorchBM2(benchmark::State& state)
    {
        auto x_vals = std::to_array<float>({ 67.5, 67.5, 47.5, 47.5, 57.5 });
        auto y_vals = std::to_array<float>({ 107.5, 102.5, -127.5, 117.5, 112.5 });
        const auto size = x_vals.size();
        const auto x_tensor = torch::from_blob(x_vals.data(), { size, 1 });
        const auto y_tensor = torch::from_blob(y_vals.data(), { size, 1 });
        auto z_vals = torch::Tensor{};
        auto sum = 0.F;
        for (auto idx : state)
        {
            z_vals = x_tensor * y_tensor;
            sum += z_vals.sum().item<float>();
        }
        std::println("sum: {}", sum);
    }

    void StlBM(benchmark::State& state)
    {
        auto x_vals = std::to_array<float>({ 67.5, 67.5, 47.5, 47.5, 57.5 });
        auto y_vals = std::to_array<float>({ 107.5, 102.5, -127.5, 117.5, 112.5 });
        auto sum = 0.F;
        for (auto idx : state)
        {
            sum += std::ranges::fold_left(
                std::views::zip_transform([](auto val1, auto val2) { return val1 + val2; }, x_vals, y_vals),
                0.F,
                std::plus{});
        }
        std::println("sum: {}", sum);
    }

} // namespace

// BENCHMARK(benchmark_function)->Name("testing");
BENCHMARK(TorchBM)->Threads(1)->Iterations(100)->Name("Using libtorch")->Unit(benchmark::kMillisecond);
BENCHMARK(TorchBM2)->Threads(1)->Iterations(100)->Name("Using libtorch2")->Unit(benchmark::kMillisecond);
BENCHMARK(StlBM)->Threads(1)->Iterations(100)->Name("Using stl")->Unit(benchmark::kMillisecond);

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
