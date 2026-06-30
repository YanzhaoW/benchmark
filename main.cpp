#include <Eigen/Dense>
#include <Eigen/Eigenvalues>
#include <Eigen/LU>
#include <Eigen/Sparse>
#include <benchmark/benchmark.h>
#include <iostream>
#include <print>
#include <ranges>

namespace
{
    void Test(benchmark::State& state)
    {

        auto matrix_a = Eigen::MatrixX<double>{}.eval();
        auto matrix_b = Eigen::MatrixX<double>{}.eval();
        auto matrix_c = Eigen::MatrixX<double>{}.eval();

        auto matrix_int = Eigen::MatrixX<double>{}.eval();
        auto matrix_res = Eigen::MatrixX<double>{}.eval();

        matrix_a.resize(200, 200);
        matrix_b.resize(200, 200);
        matrix_c.resize(200, 200);

        matrix_int.resize(200, 200);
        matrix_res.resize(200, 200);

        matrix_int.setZero();
        matrix_a.setRandom();
        matrix_b.setRandom();
        matrix_c.setRandom();
        // auto matrix_res0 = matrix_a * matrix_b;
        // std::println("row: {}, col: {}", matrix_res0.rows(), matrix_res0.cols());
        Eigen::internal::set_is_malloc_allowed(false);
        auto sum = 0.;
        for (auto idx : state)
        {
            matrix_int.noalias() = matrix_a.lazyProduct(matrix_b);
            matrix_int.noalias() = matrix_res;
            matrix_res.noalias() = matrix_int + matrix_c;
            sum += matrix_res.sum();
        }
        Eigen::internal::set_is_malloc_allowed(true);
        std::println("sum: {}", sum);
    }

    void Test2(benchmark::State& state)
    {

        auto matrix_a = Eigen::MatrixX<double>{}.eval();
        auto matrix_b = Eigen::MatrixX<double>{}.eval();
        auto matrix_c = Eigen::MatrixX<double>{}.eval();

        auto matrix_int = Eigen::MatrixX<double>{}.eval();
        auto matrix_res = Eigen::MatrixX<double>{}.eval();

        auto metric = Eigen::Matrix<double, Eigen::Dynamic, 1>{}.eval(); //!< Sigma values

        matrix_a.resize(200, 200);
        matrix_b.resize(200, 200);
        matrix_c.resize(200, 200);
        metric.resize(200);

        matrix_int.resize(200, 200);
        matrix_res.resize(200, 200);

        matrix_int.setZero();
        matrix_a.setRandom();
        matrix_b.setRandom();
        matrix_c.setRandom();
        // auto matrix_res0 = matrix_a * matrix_b;
        // std::println("row: {}, col: {}", matrix_res0.rows(), matrix_res0.cols());
        Eigen::internal::set_is_malloc_allowed(false);
        auto sum = 0.;
        for (auto idx : state)
        {
            matrix_res.noalias() =
                matrix_a.lazyProduct(metric.asDiagonal() * matrix_b) + matrix_c.lazyProduct(matrix_b);
            sum += matrix_res.sum();
        }
        Eigen::internal::set_is_malloc_allowed(true);
        std::println("sum: {}", sum);
    }

    constexpr auto dim = 10;
    constexpr auto max_dim = 13;
    using MatrixFixSize = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor, max_dim, max_dim>;

    void TestInverse(benchmark::State& state)
    {

        auto matrix_a = Eigen::MatrixX<double>{}.eval();
        auto matrix_res = Eigen::MatrixX<double>{}.eval();

        matrix_a.resize(dim, dim);
        matrix_res.resize(dim, dim);
        matrix_a.setRandom();
        matrix_a = (matrix_a * matrix_a.transpose()).eval();

        auto sum = 0.;
        auto is_invertible = true;
        for (auto idx : state)
        {
            matrix_res.noalias() = matrix_a.inverse();
            sum += matrix_res.sum();
        }
        // std::println("sum: {}", sum);
    }

    void TestLU(benchmark::State& state)
    {
        auto matrix_a = MatrixFixSize{}.eval();
        auto matrix_res = MatrixFixSize{}.eval();
        auto plu = Eigen::PartialPivLU<MatrixFixSize>{ max_dim };

        Eigen::internal::set_is_malloc_allowed(false);
        matrix_a.resize(dim, dim);
        matrix_res.resize(dim, dim);
        matrix_a.setRandom();
        matrix_a = (matrix_a * matrix_a.transpose()).eval();

        auto sum = 0.;
        auto is_invertible = true;
        // std::cout << "matrix_a: " << matrix_a << "\n";
        for (auto idx : state)
        {
            matrix_res.setIdentity();

            plu.compute(matrix_a);
            const auto& lu_mat = plu.matrixLU();
            lu_mat.triangularView<Eigen::UnitLower>().solveInPlace(matrix_res);
            lu_mat.triangularView<Eigen::Upper>().solveInPlace(matrix_res);

            sum += matrix_res.sum();
        }
        Eigen::internal::set_is_malloc_allowed(true);
        // std::cout << "res: " << matrix_res << "\n";
        // std::println("sum: {}", sum);
    }

    void TestCholesky(benchmark::State& state)
    {
        auto matrix_a = MatrixFixSize{}.eval();
        auto matrix_res = MatrixFixSize{}.eval();
        auto llt = Eigen::LLT<MatrixFixSize>{ max_dim };

        Eigen::internal::set_is_malloc_allowed(false);
        matrix_a.resize(dim, dim);
        matrix_res.resize(dim, dim);
        matrix_a.setRandom();
        matrix_a = (matrix_a * matrix_a.transpose()).eval();

        auto sum = 0.;
        auto is_invertible = true;
        // std::cout << "matrix_a: " << matrix_a << "\n";
        for (auto idx : state)
        {
            matrix_res.setIdentity();

            llt.compute(matrix_a);
            llt.solveInPlace(matrix_res);

            sum += matrix_res.sum();
        }
        Eigen::internal::set_is_malloc_allowed(true);
        // std::cout << "res: " << matrix_res << "\n";
        // std::println("sum: {}", sum);
    }

    void print(auto info)
    {
        switch (info)
        {
            case Eigen::Success:
                std::println("success");
                break;
            case Eigen::NumericalIssue:
                std::println("NumericalIssue");
                break;
            case Eigen::NoConvergence:
                std::println("NoConvergence");
                break;
            case Eigen::InvalidInput:
                std::println("InvalidInput");
                break;
        }
    }

    void TestSparse(benchmark::State& state)
    {
        // auto matrix_a = MatrixFixSize{}.eval();

        auto matrix_a = Eigen::Matrix<double, 4, 4>{}.eval();

        matrix_a(0, 0) = 1;
        // matrix_a(1, 0) = 1;
        matrix_a(1, 1) = 1;
        // matrix_a(1, 1) = 1;
        matrix_a(2, 2) = 1;
        matrix_a(0, 2) = 1;
        matrix_a(2, 0) = 1;
        matrix_a(3, 3) = 2;
        std::cout << matrix_a << "\n";

        auto matrix_b = matrix_a.llt();
        auto info = matrix_b.info();
        print(info);

        // std::cout << Eigen::Matrix<double, 4, 4>{ matrix_b.matrixL() } << "\n";
        std::cout << "inverse: \n" << matrix_b.solve(Eigen::Matrix4d::Identity()) << "\n";
        print(matrix_b.info());
        std::cout << "prod: \n" << matrix_b.solve(Eigen::Matrix4d::Identity()) * matrix_a << "\n";

        auto eigen_solver = Eigen::SelfAdjointEigenSolver<Eigen::Matrix<double, 4, 4>>{ matrix_a };

        const auto& eigen_values = eigen_solver.eigenvalues();
        std::cout << "vec: \n" << eigen_values << "\n";
        std::cout << "transform: \n" << eigen_solver.eigenvectors() << "\n";

        auto matrix_c = Eigen::Matrix<double, 4, 4>{}.eval();
        for (const auto [idx, val] : std::views::zip(std::views::iota(0), eigen_values))
        {
            if (val == 0)
            {
                matrix_c(idx, idx) = 1;
                std::println("index {} is zero", idx);
            }
        }
        std::cout << "check: \n"
                  << eigen_solver.eigenvectors() * matrix_c * eigen_solver.eigenvectors().transpose() << "\n";

        for (auto idx : state)
        {
        }
        // std::cout << "res: " << matrix_res << "\n";
        // std::println("sum: {}", sum);
    }
} // namespace

// BENCHMARK(Test)->Threads(1)->Iterations(50)->Name("With intermediate");
// BENCHMARK(Test2)->Threads(1)->Iterations(50)->Name("Without intermediate");
// BENCHMARK(TestLU)->Threads(1)->Iterations(50)->Name("LU");
// BENCHMARK(TestInverse)->Threads(1)->Iterations(50)->Name("Inverse");
// BENCHMARK(TestCholesky)->Threads(1)->Iterations(50)->Name("TestCholesky");
BENCHMARK(TestSparse)->Threads(1)->Iterations(1)->Name("TestCholesky");

auto main(int argc, char** argv) -> int
{
    benchmark::MaybeReenterWithoutASLR(argc, argv);
    benchmark::Initialize(&argc, argv);
    benchmark::RunSpecifiedBenchmarks();
    benchmark::Shutdown();
    return 0;

    return 0;
}
