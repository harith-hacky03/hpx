//  Copyright (c) 2025 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file run_on_all.hpp
/// \page hpx::experimental::run_on_all
/// \headerfile hpx/experimental/run_on_all.hpp

#pragma once

#include <hpx/config.hpp>
#include <hpx/async_combinators/wait_all.hpp>
#include <hpx/execution/detail/execution_parameter_callbacks.hpp>
#include <hpx/execution/execution.hpp>
#include <hpx/execution_base/execution.hpp>
#include <hpx/executors/parallel_executor.hpp>
#include <hpx/functional/experimental/scope_exit.hpp>
#include <hpx/parallel/algorithms/for_loop_reduction.hpp>

#include <cstddef>

namespace hpx::experimental {

    template <typename T, typename Op, typename F, typename... Ts>
    void run_on_all(std::size_t num_tasks,
        hpx::parallel::detail::reduction_helper<T, Op>&& r, F&& f, Ts&&... ts)
    {
        // force using index_queue scheduler with given amount of threads
        auto exec = hpx::execution::experimental::with_processing_units_count(
            hpx::execution::parallel_executor(
                hpx::threads::thread_priority::bound),
            num_tasks);
        exec.set_hierarchical_threshold(0);

        r.init_iteration(0, 0);
        auto on_exit =
            hpx::experimental::scope_exit([&] { r.exit_iteration(0); });

        hpx::wait_all(hpx::parallel::execution::bulk_async_execute(
            exec, [&](auto i) { f(r.iteration_value(i), ts...); }, num_tasks,
            HPX_FORWARD(Ts, ts)...));
    }

    template <typename T, typename Op, typename F, typename... Ts>
    void run_on_all(
        hpx::parallel::detail::reduction_helper<T, Op>&& r, F&& f, Ts&&... ts)
    {
        std::size_t cores =
            hpx::parallel::execution::detail::get_os_thread_count();
        run_on_all(
            cores, HPX_MOVE(r), HPX_FORWARD(F, f), HPX_FORWARD(Ts, ts)...);
    }
}    // namespace hpx::experimental
