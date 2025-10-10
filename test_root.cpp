#include "RootModel.hpp"

auto main() -> int
{
    auto root_model = RootModel(1.35);
    std::vector<double> x_vals{ 127.5, 117.5, 107.5, 97.5, 87.5, 77.5, 67.5, 7.5, 17.5, 27.5, 37.5, 47.5, 57.5 };
    std::vector<double> y_vals{-117.5, -117.5, -117.5, -117.5, -117.5, -117.5, -117.5, -122.5, -122.5, -122.5, -122.5, -117.5, -117.5};
    std::vector<double> y_errs{ 5., 5., 5., 5., 5. };

    root_model.train_from_data(x_vals, y_vals);
    root_model.print();
    std::println("p value: {}", root_model.calculate_p_value(y_errs));

    return 0;
}
