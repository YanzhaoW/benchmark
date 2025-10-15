#include <map>
#include <string>
#include <torch/torch.h>
#include <vector>

auto myfun() -> std::size_t
{
    auto sum = 0;
    for (auto idx = int{}; idx < 10; ++idx)
    {
        // auto vec = std::vector<double>{};
        auto torch_vec = torch::randn({ 1000000 });
        // auto torch_vec2 = torch::randn({ 1000000 });
        auto torch_vec2 = torch_vec;
        auto torch2 = torch_vec + torch_vec2;
        // vec.reserve(800000);
        sum += torch_vec.size(0) + torch2.size(0);
    }
    return sum;
}
