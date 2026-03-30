#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <algorithm>
#include <benchmark/benchmark.h>
#include <print>
#include <random>
#include <ranges>

namespace
{
    constexpr auto n_rows_max = 200;
    constexpr auto n_rows_min = 30;
    constexpr auto n_cols_max = 200;
    constexpr auto n_cols_min = 30;
    constexpr auto enable_hist = false;

    void dense_randomize(auto& matrix, std::mt19937& engine)
    {
        auto dist = std::uniform_real_distribution{ 10.F, 20.F };
        for (auto col : matrix.colwise())
        {
            std::ranges::generate(col, [&engine, &dist]() -> float { return dist(engine); });
        }
    }

    void TestDenseMatrix(benchmark::State& state)
    {
        auto rand_dev = std::random_device{};
        auto engine = std::mt19937{ rand_dev() };
        auto row_dist = std::uniform_int_distribution{ n_rows_min, n_rows_max };
        auto col_dist = std::uniform_int_distribution{ n_cols_min, n_cols_max };
        // auto matrix1 = Eigen::MatrixXf{ n_rows, n_cols };
        // auto matrix2 = Eigen::MatrixXf{ n_rows, n_cols };
        auto matrix = Eigen::MatrixXf::Zero(n_rows_max, n_cols_max).eval();

        auto hist_n_rows = std::views::iota(n_rows_min, n_rows_max + 1) |
                           std::views::transform([](auto num) { return std::pair{ num, 0U }; }) |
                           std::ranges::to<std::unordered_map>();
        auto hist_n_cols = std::views::iota(n_cols_min, n_cols_max + 1) |
                           std::views::transform([](auto num) { return std::pair{ num, 0U }; }) |
                           std::ranges::to<std::unordered_map>();
        auto sum = 0.F;
        for (auto idx : state)
        {
            const auto n_row = row_dist(engine);
            const auto n_col = col_dist(engine);
            if constexpr (enable_hist)
            {
                ++hist_n_rows[n_row];
                ++hist_n_cols[n_col];
            }
            matrix.resize(n_row, n_col);
            dense_randomize(matrix, engine);
            sum += matrix.sum() / (n_row * n_col);
        }
        if constexpr (enable_hist)
        {
            std::println("row hist: {}", hist_n_rows);
            std::println("col hist: {}", hist_n_rows);
        }
    }

    void TestVector(benchmark::State& state)
    {
        auto rand_dev = std::random_device{};
        auto engine = std::mt19937{ rand_dev() };
        auto row_dist = std::uniform_int_distribution{ n_rows_min, n_rows_max };
        auto col_dist = std::uniform_int_distribution{ n_cols_min, n_cols_max };
        // auto matrix1 = Eigen::MatrixXf{ n_rows, n_cols };
        // auto matrix2 = Eigen::MatrixXf{ n_rows, n_cols };
        auto vec = std::vector<float>{};
        vec.resize(n_rows_max * n_cols_max, 0.F);

        auto hist_n_rows = std::views::iota(n_rows_min, n_rows_max + 1) |
                           std::views::transform([](auto num) { return std::pair{ num, 0U }; }) |
                           std::ranges::to<std::unordered_map>();
        auto hist_n_cols = std::views::iota(n_cols_min, n_cols_max + 1) |
                           std::views::transform([](auto num) { return std::pair{ num, 0U }; }) |
                           std::ranges::to<std::unordered_map>();

        auto sum = 0.F;
        for (auto idx : state)
        {
            const auto n_row = row_dist(engine);
            const auto n_col = col_dist(engine);
            if constexpr (enable_hist)
            {
                ++hist_n_rows[n_row];
                ++hist_n_cols[n_col];
            }
            vec.resize(n_row * n_col, 0.F);
            auto matrix =
                Eigen::Map<Eigen::Matrix<float, -1, -1, Eigen::ColMajor>, Eigen::Aligned>(vec.data(), n_row, n_col);
            dense_randomize(matrix, engine);
            sum += matrix.sum() / (n_row * n_col);
        }
        if constexpr (enable_hist)
        {
            std::println("row hist: {}", hist_n_rows);
            std::println("col hist: {}", hist_n_rows);
        }
    }

} // namespace

// BENCHMARK(TestMatrix)->Threads(1)->Iterations(10000000)->Name("Using Eigen")->Unit(benchmark::kMillisecond);
BENCHMARK(TestDenseMatrix)->Threads(1)->Iterations(100000)->Name("Using dense matrix resize");
BENCHMARK(TestVector)->Threads(1)->Iterations(100000)->Name("Using vector");

auto main(int argc, char** argv) -> int
{
    benchmark::MaybeReenterWithoutASLR(argc, argv);
    // auto mm = make_unique<benchmark::MemoryManager>();
    // benchmark::RegisterMemoryManager();
    benchmark::Initialize(&argc, argv);
    benchmark::RunSpecifiedBenchmarks();
    benchmark::Shutdown();
    return 0;
}
