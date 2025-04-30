#include <hpx/experimental/run_on_all.hpp>
#include <hpx/execution.hpp>
#include <hpx/init.hpp>

int hpx_main()
{
    hpx::experimental::run_on_all(hpx::execution::par, []() {
        // Test code here
    });
    return hpx::finalize();
}

int main(int argc, char* argv[])
{
    return hpx::init(argc, argv);
} 