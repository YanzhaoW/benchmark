#pragma once

#include <format>
#include <print>
#include <sstream>
#include <torch/torch.h>

template <>
struct std::formatter<torch::Tensor>
{
    static constexpr auto parse(std::format_parse_context& ctx) { return ctx.begin(); }

    static auto format(const torch::Tensor& tensor, std::format_context& ctx)
    {
        return std::format_to(ctx.out(), "{}", (std::stringstream{} << tensor).str());
    }
};

class Net : public torch::nn::Module
{
  public:
    explicit Net(int max_iter)
        : weight_{ register_parameter("weight", torch::tensor({ 1.F }), true) }
        , bias_{ register_parameter("bias", torch::tensor({ 0.F }), true) }
        , sigma_{ register_parameter("scale", torch::tensor({ 1.F }), true) }
        , optimizer_{ torch::optim::LBFGS{
              parameters(),
              torch::optim::LBFGSOptions{}.max_iter(max_iter).line_search_fn("strong_wolfe") } }
    {
        auto fast_lr = std::make_unique<torch::optim::AdamOptions>();
        fast_lr->lr(10);
        auto offset_group = torch::optim::OptimizerParamGroup(std::vector{ bias_ }, std::move(fast_lr));
        auto other_group = torch::optim::OptimizerParamGroup(std::vector{ weight_, sigma_ }, std::move(fast_lr));
        adam_optimizer_ = std::make_unique<torch::optim::Adam>(std::vector{ offset_group, other_group },
                                                               torch::optim::AdamOptions{}.lr(0.1));
    }
    auto forward(const torch::Tensor& val) -> torch::Tensor { return weight_ * val + bias_; }

    auto calculate_loss(const torch::Tensor& y_vals, const torch::Tensor& y_preds)
    {
        auto n_outliers = 0;
        const auto n_data = y_vals.size(0);

        auto loss = std::ranges::fold_left(std::views::zip_transform(
                                               [&](const torch::Tensor& y_val, const torch::Tensor& y_pred)
                                               {
                                                   auto residual = torch::abs(y_val - y_pred);
                                                   if ((residual > epsilon_ * torch::abs(sigma_)).item<bool>())
                                                   {
                                                       ++n_outliers;
                                                       return residual * 2.0 * epsilon_;
                                                   }
                                                   return residual * residual / torch::abs(sigma_);
                                               },
                                               y_vals.unbind(),
                                               y_preds.unbind()),
                                           0.0001 * weight_ * weight_,
                                           std::plus{});
        loss += n_data * torch::abs(sigma_);
        loss -= n_outliers * torch::abs(sigma_) * epsilon_ * epsilon_;
        return loss;
    }

    auto train_from_data(const torch::Tensor& x_vals, const torch::Tensor& y_vals) -> int
    {
        auto n_iter = 0;

        auto loss_fun = [&]()
        {
            optimizer_.zero_grad();
            auto predict = forward(x_vals);
            auto loss = calculate_loss(y_vals, predict);
            loss.backward({}, true);
            ++n_iter;
            return loss;
        };

        auto loss_val = optimizer_.step(loss_fun);
        n_iter_ = n_iter;
        return n_iter;
    }

    auto train_from_data_adam(const torch::Tensor& x_vals, const torch::Tensor& y_vals) -> int
    {
        auto n_iter = 0;
        auto tolerance = 0.001F;
        auto max_grad = 1.F;
        const auto max_iter = 500;

        for (auto idx : std::views::iota(0, 500))
        {
            adam_optimizer_->zero_grad();
            auto predict = forward(x_vals);
            auto loss = calculate_loss(y_vals, predict);
            loss.backward({}, true);
            ++n_iter;
            auto loss_val = adam_optimizer_->step();
            max_grad = std::max({ std::abs(weight_.grad().item<float>()),
                                  std::abs(bias_.grad().item<float>()),
                                  std::abs(sigma_.grad().item<float>()) });
            if (max_grad < tolerance)
            {
                break;
            }
        }

        // adam_optimizer_->zero_grad();
        // auto predict = forward(x_vals);
        // auto loss = calculate_loss(y_vals, predict);
        // loss.backward({}, true);
        // ++n_iter;

        n_iter_ = n_iter;
        return n_iter;
    }

    void clear()
    {
        optimizer_.zero_grad();
        torch::NoGradGuard no_grad;
        weight_.fill_(torch::tensor(1.F));
        bias_.fill_(torch::tensor(0.F));
        sigma_.fill_(torch::tensor(1.F));
    }

    void print() const
    {
        std::println("Iterations: {}", n_iter_);
        for (auto par : named_parameters())
        {
            if (par.value().grad().numel() == 0)
            {
                std::println("{}: {}", par.key(), par.value().item<float>());
            }
            else
            {
                std::println("{}: {}, grad:{}", par.key(), par.value().item<float>(), par.value().grad().item<float>());
            }
        }
    }

  private:
    float epsilon_ = 1.35;
    int n_iter_ = 0;
    torch::Tensor weight_;
    torch::Tensor bias_;
    torch::Tensor sigma_;
    torch::optim::LBFGS optimizer_;
    std::unique_ptr<torch::optim::Adam> adam_optimizer_;
};
