
// #include <torch/torch.h>

#include <algorithm>
#include <array>
#include <malloc.h>
#include <map>
#include <print>
#include <vector>

// auto print_info(const struct mallinfo2& info)
// {
//     std::println("size: {}", sizeof(info));
//     auto info_array = std::array<std::size_t, sizeof(info) / sizeof(size_t)>{};
//     std::memcpy(info_array.data(), &info, sizeof(info));
//     std::println("array: {}", info_array);
//     std::println("arena: {}, ordbls: {}, uordblks: {}", info.arena, info.ordblks, info.uordblks);
// }

class MallocInfo
{
  public:
    MallocInfo()
    {
        first_info_ = mallinfo2();
        // print_info(first_info_);
    }
    ~MallocInfo()
    {
        auto last_info = mallinfo2();
        // print_info(last_info);
        std::println("allocation difference: {}", last_info.uordblks - first_info_.uordblks);
    }

  private:
    using mallinfo_struct = struct mallinfo2;
    mallinfo_struct first_info_{};
};

// auto run_tensor()
// {
//      auto val =  std::map<int, std::string>{ { 1, "adsfagadf" }, { 2, "adsfasdfasg" } };
//     // malloc_stats();
//     // auto info = MallocInfo{};
//     // auto sum = 0;
//     // // auto tensor = torch::tensor(1.0);
//     // // auto tensor2 = tensor;
//     // // auto tensor3 = torch::tensor(1.0);
//     // // return (tensor2 + tensor3).item<float>();
//     // auto vec = std::vector<int>{};
//     // vec.reserve(6);
//     // malloc_stats();
//     // return vec.capacity();
//      return val.size();
// }


auto myfun()-> std::size_t;

auto main() -> int
{
    auto sum = myfun();
    std::println("{}", sum);
    return 0;
}
