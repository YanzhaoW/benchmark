#include "TorchModel.hpp"

auto main() -> int
{
    auto x_vals = std::to_array<float>({ 127.5, 117.5, 107.5, 97.5, 87.5, 77.5, 67.5, 7.5, 17.5, 27.5, 37.5, 47.5, 57.5 });
    auto y_vals = std::to_array<float>({-117.5, -117.5, -117.5, -117.5, -117.5, -117.5, -117.5, -122.5, -122.5, -122.5, -122.5, -117.5, -117.5});

    const auto size = x_vals.size();
    const auto x_tensor = torch::from_blob(x_vals.data(), { size, 1 });
    const auto y_tensor = torch::from_blob(y_vals.data(), { size, 1 });

    constexpr auto max_iter = 100;
    auto model = Net{ max_iter };

    auto n_iter = model.train_from_data(x_tensor, y_tensor);
    // auto n_iter = model.train_from_data_adam(x_tensor, y_tensor);
    std::println("Number of iterations: {}", n_iter);
    model.print();

    return 0;
}
