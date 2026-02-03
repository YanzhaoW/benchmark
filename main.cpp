#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <algorithm>
#include <benchmark/benchmark.h>
#include <print>
#include <random>
#include <ranges>

namespace
{
    constexpr auto n_rows = 3000;
    constexpr auto n_cols = 3000;
    constexpr auto nnz = 10;
    constexpr auto nnz_col = 20;

    void dense_randomize(Eigen::MatrixXf& matrix, std::mt19937& engine)
    {
        auto dist = std::uniform_real_distribution{ 10.F, 20.F };
        for (auto col : matrix.colwise())
        {
            std::ranges::generate(col, [&engine, &dist]() -> float { return dist(engine); });
        }
    }

    void sparse_randomize(Eigen::SparseMatrix<float>& matrix, std::mt19937& engine)
    {
        auto value_dist = std::uniform_real_distribution{ 10.F, 20.F };
        matrix.setZero();
        auto nnz_array_row_idx = std::array<int, nnz>{};
        auto nnz_array_col_idx = std::array<int, nnz>{};
        auto nnz_array_value = std::array<float, nnz>{};

        std::ranges::sample(std::views::iota(0, n_rows), nnz_array_row_idx.begin(), nnz_array_row_idx.size(), engine);
        std::ranges::sample(std::views::iota(0, nnz_col), nnz_array_col_idx.begin(), nnz_array_col_idx.size(), engine);
        std::ranges::generate(nnz_array_value, [&engine, &value_dist]() -> float { return value_dist(engine); });

        auto reserve_index_vec = Eigen::VectorXi{ Eigen::VectorXi::Zero(n_cols) };
        for (auto col_idx : nnz_array_col_idx)
        {
            reserve_index_vec[col_idx] = nnz;
        }

        matrix.reserve(reserve_index_vec);
        for (const auto& [row_idx, col_idx, value] :
             std::views::zip(nnz_array_row_idx, nnz_array_col_idx, nnz_array_value))
        {
            matrix.insert(row_idx, col_idx) = value;
        }
        matrix.makeCompressed();
    }

    void TestDenseMatrix(benchmark::State& state)
    {
        auto rand_dev = std::random_device{};
        auto engine = std::mt19937{ rand_dev() };
        // auto matrix1 = Eigen::MatrixXf{ n_rows, n_cols };
        // auto matrix2 = Eigen::MatrixXf{ n_rows, n_cols };
        auto matrix1 = Eigen::MatrixXf{ Eigen::MatrixXf::Constant(n_rows, n_cols, 1.7F) };
        auto matrix2 = Eigen::MatrixXf{ Eigen::MatrixXf::Constant(n_rows, n_cols, 2.7F) };
        auto matrix3 = Eigen::MatrixXf{ Eigen::MatrixXf::Zero(n_rows, n_cols) };

        auto sum = 0.F;
        for (auto idx : state)
        {
            dense_randomize(matrix1, engine);
            dense_randomize(matrix2, engine);
            matrix3 = matrix1 * matrix2;
            sum += matrix3.sum();
        }
    }

    void TestSparseMatrix(benchmark::State& state, bool only_rand = true)
    {
        auto rand_dev = std::random_device{};
        auto engine = std::mt19937{ rand_dev() };
        // auto matrix1 = Eigen::MatrixXf{ n_rows, n_cols };
        // auto matrix2 = Eigen::MatrixXf{ n_rows, n_cols };
        auto matrix1 = Eigen::SparseMatrix<float>{ n_rows, n_cols };
        auto matrix2 = Eigen::SparseMatrix<float>{ n_rows, n_cols };

        auto sum = 0.F;
        for (auto idx : state)
        {
            sparse_randomize(matrix1, engine);
            sparse_randomize(matrix2, engine);
            if (not only_rand)
            {
                sum += (matrix1 * matrix2).sum();
            }
        }
    }
} // namespace

// BENCHMARK(TestMatrix)->Threads(1)->Iterations(10000000)->Name("Using Eigen")->Unit(benchmark::kMillisecond);
BENCHMARK_CAPTURE(TestSparseMatrix, all, false)->Threads(1)->Iterations(10)->Name("Using sparse matrix");
BENCHMARK_CAPTURE(TestSparseMatrix, only_rand, true)->Threads(1)->Iterations(10)->Name("Using sparse matrix (only randomization)");
BENCHMARK(TestDenseMatrix)->Threads(1)->Iterations(10)->Name("Using dense matrix");

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
