#pragma once

#include <Math/Factory.h>
#include <Math/Functor.h>
// #include <Math/GSLMinimizer.h>
#include <Math/Minimizer.h>
#include <Math/ProbFunc.h>
#include <Minuit2/Minuit2Minimizer.h>
#include <array>
#include <memory>
#include <print>
#include <ranges>

class RootModel
{
  public:
    explicit RootModel(double epsilon)
        : epsilon_{ epsilon }
        , minimizer_{ ROOT::Math::Factory::CreateMinimizer("Minuit2") }
    {
        minimizer_->SetTolerance(0.01);
        clear();
    }

    static constexpr auto n_pars = 3;

    auto calculate_loss(const double* raw_pars) -> double
    {
        std::ranges::copy(std::span(raw_pars, n_pars), pars_.begin());
        const auto [slope, offset, sigma] = pars_;

        const auto n_data = static_cast<int>(x_vals_.size());
        auto n_outliers = 0;

        auto loss = std::ranges::fold_left(std::views::zip_transform(
                                               [&](auto x_val, auto y_val)
                                               {
                                                   const auto residual = std::abs(y_val - (x_val * slope) - offset);
                                                   const auto thresh = std::abs(sigma * epsilon_);
                                                   if (residual > thresh)
                                                   {
                                                       ++n_outliers;
                                                       return 2. * epsilon_ * residual;
                                                   }
                                                   return residual * residual / sigma;
                                               },
                                               x_vals_,
                                               y_vals_),
                                           (0.0001 * slope * slope),
                                           std::plus{});
        loss += n_data * sigma - sigma * n_outliers * epsilon_ * epsilon_;
        // std::println("loss: {}", loss);
        return loss;
    }

    void train_from_data(const std::vector<double>& x_vals, const std::vector<double>& y_vals)
    {
        // auto loss_functor =
        auto functor = ROOT::Math::Functor{ [&](const double* raw_pars) -> double { return calculate_loss(raw_pars); },
                                            RootModel::n_pars };
        minimizer_->SetFunction(functor);
        x_vals_ = std::span{ x_vals };
        y_vals_ = std::span{ y_vals };
        auto is_ok = minimizer_->Minimize();

        if (not is_ok)
        {
            std::println("minimization failed!");
        }
    }
    [[nodiscard]] auto check_outlier(double x_val,
                                     double y_val,
                                     const std::pair<double, double>& weight_bias) const -> std::pair<bool, double>
    {
        const auto residual = std::abs(y_val - (x_val * weight_bias.first) - weight_bias.second);
        const auto thresh = std::abs(pars_[2] * epsilon_);
        return std::make_pair(residual > thresh, residual);
    }

    auto calculate_p_value(const std::vector<double>& y_errs) -> double
    {
        const auto chi_square = std::ranges::fold_left(std::views::zip_transform(
                                                           [&](auto x_val, auto y_val, auto y_err)
                                                           {
                                                               auto [is_outlier, residual] = check_outlier(
                                                                   x_val, y_val, std::make_pair(pars_[0], pars_[1]));

                                                               if (is_outlier)
                                                               {
                                                                   return 0.;
                                                               }
                                                               return residual * residual / y_err;
                                                           },
                                                           x_vals_,
                                                           y_vals_,
                                                           y_errs),
                                                       0.,
                                                       std::plus{});
        return ROOT::Math::chisquared_cdf_c(chi_square, static_cast<double>(y_errs.size() - 1));
    }

    void clear()
    {
        minimizer_->SetVariable(0, "slope", 1., .1);
        minimizer_->SetVariable(1, "offset", 0., 1);
        minimizer_->SetVariable(2, "sigma", 1., .1);
    }

    void print()
    {
        const auto pars = std::span(minimizer_->X(), RootModel::n_pars);
        const auto errors = std::span(minimizer_->Errors(), RootModel::n_pars);
        // const auto errors = std::vector<double>(RootModel::n_pars);
        // std::println("errors: {}", errors);
        std::println("iterations: {}", minimizer_->NIterations());
        static constexpr auto names = std::to_array<std::string_view>({ "slope", "offset", "sigma" });
        for (const auto [name, par, error] : std::views::zip(names, pars, errors))
        {
            std::println("{}: {} +/- {}", name, par, error);
        }
    }
    // void set_x_vals(const std::vector<double>& vals) { x_vals_ = std::span{ vals }; }
    // void set_y_vals(const std::vector<double>& vals) { y_vals_ = std::span{ vals }; }

  private:
    double epsilon_ = 0.;
    std::span<const double> x_vals_;
    std::span<const double> y_vals_;
    std::array<double, n_pars> pars_{};
    std::unique_ptr<ROOT::Math::Minimizer> minimizer_;
};
