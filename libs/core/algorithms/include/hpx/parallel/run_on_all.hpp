//  Copyright (c) 2025 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file run_on_all.hpp
/// \page hpx::experimental::run_on_all
/// \headerfile hpx/experimental/run_on_all.hpp
///
/// This file provides the implementation of the run_on_all functionality, which
/// allows running a function on all available worker threads with support for
/// different execution policies and reductions.

#pragma once

#include <hpx/config.hpp>
#include <hpx/async_combinators/wait_all.hpp>
#include <hpx/concepts/concepts.hpp>
#include <hpx/execution/detail/execution_parameter_callbacks.hpp>
#include <hpx/execution/execution.hpp>
#include <hpx/execution_base/execution.hpp>
#include <hpx/executors/parallel_executor.hpp>
#include <hpx/functional/experimental/scope_exit.hpp>
#include <hpx/parallel/algorithms/for_loop_reduction.hpp>
#include <hpx/parallel/execution.hpp>

#include <cstddef>
#include <type_traits>

namespace hpx::experimental {

    /// \brief Run a function on all available worker threads with reduction support
    /// \tparam ExPolicy The execution policy type
    /// \tparam T The reduction value type
    /// \tparam Op The reduction operation type
    /// \tparam F The function type to execute
    /// \tparam Ts Additional argument types
    /// \param policy The execution policy to use
    /// \param num_tasks The number of tasks to create
    /// \param r The reduction helper
    /// \param f The function to execute
    /// \param ts Additional arguments to pass to the function
    template <typename ExPolicy, typename T, typename Op, typename F,
        typename... Ts>
    decltype(auto) run_on_all(ExPolicy&& policy, std::size_t num_tasks,
        hpx::parallel::detail::reduction_helper<T, Op>&& r, F&& f, Ts&&... ts)
    {
        static_assert(hpx::is_execution_policy_v<ExPolicy>,
            "hpx::is_execution_policy_v<ExPolicy>");

        // Configure executor with proper scheduling hints
        hpx::threads::thread_schedule_hint hint;
        hint.sharing_mode(
            hpx::threads::thread_sharing_hint::do_not_share_function);

        auto exec = hpx::execution::experimental::with_processing_units_count(
            hpx::execution::parallel_executor(
                hpx::threads::thread_priority::bound,
                hpx::threads::thread_stacksize::default_, hint),
            num_tasks);
        exec.set_hierarchical_threshold(0);

        // Initialize reduction and ensure proper cleanup
        r.init_iteration(0, 0);
        auto on_exit =
            hpx::experimental::scope_exit([&] { r.exit_iteration(0); });

        // Execute based on policy type
        if constexpr (hpx::is_async_execution_policy_v<ExPolicy>)
        {
            return hpx::parallel::execution::bulk_async_execute(
                exec, [&](auto i) { f(r.iteration_value(i), ts...); },
                num_tasks, HPX_FORWARD(Ts, ts)...);
        }
        else
        {
            return hpx::wait_all(hpx::parallel::execution::bulk_async_execute(
                exec, [&](auto i) { f(r.iteration_value(i), ts...); },
                num_tasks, HPX_FORWARD(Ts, ts)...));
        }
    }

    /// \brief Run a function on all available worker threads with reduction support
    /// \tparam ExPolicy The execution policy type
    /// \tparam T The reduction value type
    /// \tparam Op The reduction operation type
    /// \tparam F The function type to execute
    /// \tparam Ts Additional argument types
    /// \param policy The execution policy to use
    /// \param r The reduction helper
    /// \param f The function to execute
    /// \param ts Additional arguments to pass to the function
    template <typename ExPolicy, typename T, typename Op, typename F,
        typename... Ts>
    decltype(auto) run_on_all(ExPolicy&& policy,
        hpx::parallel::detail::reduction_helper<T, Op>&& r, F&& f, Ts&&... ts)
    {
        static_assert(hpx::is_execution_policy_v<ExPolicy>,
            "hpx::is_execution_policy_v<ExPolicy>");

        std::size_t cores =
            hpx::parallel::execution::detail::get_os_thread_count();
        return run_on_all(HPX_FORWARD(ExPolicy, policy), cores, HPX_MOVE(r),
            HPX_FORWARD(F, f), HPX_FORWARD(Ts, ts)...);
    }

    /// \brief Run a function on all available worker threads
    /// \tparam ExPolicy The execution policy type
    /// \tparam F The function type to execute
    /// \tparam Ts Additional argument types
    /// \param policy The execution policy to use
    /// \param num_tasks The number of tasks to create
    /// \param f The function to execute
    /// \param ts Additional arguments to pass to the function
    template <typename ExPolicy, typename F, typename... Ts>
    decltype(auto) run_on_all(ExPolicy&& policy, std::size_t num_tasks, F&& f, Ts&&... ts)
    {
        static_assert(hpx::is_execution_policy_v<ExPolicy>,
            "hpx::is_execution_policy_v<ExPolicy>");

        // Configure executor with proper scheduling hints
        hpx::threads::thread_schedule_hint hint;
        hint.sharing_mode(
            hpx::threads::thread_sharing_hint::do_not_share_function);

        auto exec = hpx::execution::experimental::with_processing_units_count(
            hpx::execution::parallel_executor(
                hpx::threads::thread_priority::bound,
                hpx::threads::thread_stacksize::default_, hint),
            num_tasks);
        exec.set_hierarchical_threshold(0);

        // Execute based on policy type
        if constexpr (hpx::is_async_execution_policy_v<ExPolicy>)
        {
            return hpx::parallel::execution::bulk_async_execute(
                exec, [&](auto) { f(ts...); }, num_tasks,
                HPX_FORWARD(Ts, ts)...);
        }
        else
        {
            return hpx::wait_all(hpx::parallel::execution::bulk_async_execute(
                exec, [&](auto) { f(ts...); }, num_tasks,
                HPX_FORWARD(Ts, ts)...));
        }
    }

    /// \brief Run a function on all available worker threads
    /// \tparam ExPolicy The execution policy type
    /// \tparam F The function type to execute
    /// \tparam Ts Additional argument types
    /// \param policy The execution policy to use
    /// \param f The function to execute
    /// \param ts Additional arguments to pass to the function
    template <typename ExPolicy, typename F, typename... Ts,
        HPX_CONCEPT_REQUIRES_(std::is_invocable_v<F&&, Ts&&...>)>
    decltype(auto) run_on_all(ExPolicy&& policy, F&& f, Ts&&... ts)
    {
        static_assert(hpx::is_execution_policy_v<ExPolicy>,
            "hpx::is_execution_policy_v<ExPolicy>");

        std::size_t cores =
            hpx::parallel::execution::detail::get_os_thread_count();
        return run_on_all(HPX_FORWARD(ExPolicy, policy), cores, HPX_FORWARD(F, f),
            HPX_FORWARD(Ts, ts)...);
    }

    // Overloads without execution policy (default to sequential execution)
    template <typename T, typename Op, typename F, typename... Ts>
    decltype(auto) run_on_all(std::size_t num_tasks,
        hpx::parallel::detail::reduction_helper<T, Op>&& r, F&& f, Ts&&... ts)
    {
        return run_on_all(hpx::execution::seq, num_tasks, HPX_MOVE(r),
            HPX_FORWARD(F, f), HPX_FORWARD(Ts, ts)...);
    }

    template <typename T, typename Op, typename F, typename... Ts>
    decltype(auto) run_on_all(
        hpx::parallel::detail::reduction_helper<T, Op>&& r, F&& f, Ts&&... ts)
    {
        return run_on_all(hpx::execution::seq, HPX_MOVE(r), HPX_FORWARD(F, f),
            HPX_FORWARD(Ts, ts)...);
    }

    template <typename F, typename... Ts>
    decltype(auto) run_on_all(std::size_t num_tasks, F&& f, Ts&&... ts)
    {
        return run_on_all(hpx::execution::seq, num_tasks, HPX_FORWARD(F, f),
            HPX_FORWARD(Ts, ts)...);
    }

    template <typename F, typename... Ts,
        HPX_CONCEPT_REQUIRES_(std::is_invocable_v<F&&, Ts&&...>)>
    decltype(auto) run_on_all(F&& f, Ts&&... ts)
    {
        return run_on_all(
            hpx::execution::seq, HPX_FORWARD(F, f), HPX_FORWARD(Ts, ts)...);
    }
}    // namespace hpx::experimental
