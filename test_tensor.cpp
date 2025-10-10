
// #include <torch/torch.h>

#include <algorithm>
#include <print>
#include <vector>

__attribute__((noinline))
auto run_tensor() -> int
{
    auto sum = 0;
    for (auto idx = 0; idx < 1000; ++idx)
    {
        // auto tensor = torch::tensor(0.);

        auto vec = std::vector{ 1, 2, 3, 4 };
        sum += std::ranges::fold_left(vec, 0, std::plus{});
    }
    return sum;
}

auto main() -> int
{
    auto sum = run_tensor();
    return 0;
}
