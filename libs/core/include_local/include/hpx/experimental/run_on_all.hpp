//  Copyright (c) 2025 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>
#include <hpx/assert.hpp>
#include <hpx/async_base/launch_policy.hpp>
#include <hpx/concepts/concepts.hpp>
#include <hpx/execution/algorithms/detail/predicates.hpp>
#include <hpx/execution/executors/execution.hpp>
#include <hpx/execution/executors/execution_parameters.hpp>
#include <hpx/execution/executors/static_chunk_size.hpp>
#include <hpx/execution_base/execution.hpp>
#include <hpx/execution_base/traits/is_executor.hpp>
#include <hpx/functional/detail/tag_fallback_invoke.hpp>
#include <hpx/iterator_support/range.hpp>
#include <hpx/iterator_support/traits/is_iterator.hpp>
#include <hpx/parallel/algorithms/detail/advance_to_sentinel.hpp>
#include <hpx/parallel/algorithms/detail/distance.hpp>
#include <hpx/parallel/algorithms/detail/execution_parameters.hpp>
#include <hpx/parallel/algorithms/detail/predicates.hpp>
#include <hpx/parallel/util/detail/algorithm_result.hpp>
#include <hpx/parallel/util/detail/chunk_size.hpp>
#include <hpx/parallel/util/detail/handle_local_exceptions.hpp>
#include <hpx/parallel/util/detail/sender_util.hpp>
#include <hpx/parallel/util/loop.hpp>
#include <hpx/parallel/util/partitioner.hpp>
#include <hpx/parallel/util/result_types.hpp>
#include <hpx/parallel/util/scan_partitioner.hpp>
#include <hpx/parallel/util/transfer.hpp>
#include <hpx/type_support/empty_function.hpp>
#include <hpx/type_support/unused.hpp>

#include <algorithm>
#include <cstddef>
#include <functional>
#include <iterator>
#include <type_traits>
#include <utility>
#include <vector>

namespace hpx::experimental {

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
    template <typename ExPolicy, typename... Reductions, typename F,
        typename... Ts>
    decltype(auto) run_on_all(ExPolicy&& policy, Reductions&&... reductions,
        F&& f, [[maybe_unused]] Ts&&... ts)
    {
        static_assert(hpx::is_execution_policy_v<ExPolicy>,
            "hpx::is_execution_policy_v<ExPolicy>");

        [[maybe_unused]] std::size_t cores =
            hpx::parallel::execution::detail::get_os_thread_count();

        // Create executor with proper configuration
        auto exec = hpx::execution::experimental::with_processing_units_count(
            HPX_FORWARD(ExPolicy, policy), cores);

        // Initialize all reductions
        std::vector<std::reference_wrapper<std::decay_t<Reductions>>>
            all_reductions{
                std::forward_as_tuple(HPX_FORWARD(Reductions, reductions)...)};
        for (auto& r : all_reductions)
        {
            r.get().init_iteration(0, 0);
        }

        // Create a lambda that captures all reductions
        auto task = [all_reductions = HPX_MOVE(all_reductions), &f, &ts...](
                        std::size_t index) {
            // Create tuple of reductions using index sequence
            auto make_reduction_tuple = [&all_reductions](auto&& seq) {
                return std::tuple<std::decay_t<Reductions>...>{
                    HPX_MOVE(all_reductions[seq.index].get())...};
            };
            auto reduction_tuple = make_reduction_tuple(
                std::make_index_sequence<sizeof...(Reductions)>{});
            f(index, HPX_MOVE(reduction_tuple), HPX_FORWARD(Ts, ts)...);
        };

        // Execute based on policy type
        if constexpr (hpx::is_async_execution_policy_v<ExPolicy>)
        {
            auto fut = hpx::parallel::execution::bulk_async_execute(
                exec, task, cores, HPX_FORWARD(Ts, ts)...);

            // Create a cleanup function that will be called when all tasks complete
            auto cleanup = [all_reductions =
                                   HPX_MOVE(all_reductions)]() mutable {
                for (auto& r : all_reductions)
                {
                    r.get().exit_iteration(0);
                }
            };

            // Return a future that performs cleanup after all tasks complete
            return fut.then(
                [cleanup = HPX_MOVE(cleanup)](auto&& fut_inner) mutable {
                    cleanup();
                    return HPX_MOVE(fut_inner.get());
                });
        }
        else
        {
            auto result =
                hpx::wait_all(hpx::parallel::execution::bulk_async_execute(
                    exec, task, cores, HPX_FORWARD(Ts, ts)...));

            // Clean up reductions
            for (auto& r : all_reductions)
            {
                r.get().exit_iteration(0);
            }
            return result;
        }
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
    decltype(auto) run_on_all([[maybe_unused]] ExPolicy&& policy,
        std::size_t num_tasks, F&& f, [[maybe_unused]] Ts&&... ts)
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
    decltype(auto) run_on_all(
        ExPolicy&& policy, F&& f, [[maybe_unused]] Ts&&... ts)
    {
        static_assert(hpx::is_execution_policy_v<ExPolicy>,
            "hpx::is_execution_policy_v<ExPolicy>");

        std::size_t cores =
            hpx::parallel::execution::detail::get_os_thread_count();
        return run_on_all(HPX_FORWARD(ExPolicy, policy), cores,
            HPX_FORWARD(F, f), HPX_FORWARD(Ts, ts)...);
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
