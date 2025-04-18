# Copyright (c) 2025 The STE||AR-Group
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

if(HPX_WITH_EXAMPLES)
  add_hpx_pseudo_target(examples.modules.algorithms)
  add_hpx_pseudo_dependencies(examples.modules examples.modules.algorithms)
  if(HPX_WITH_TESTS AND HPX_WITH_TESTS_EXAMPLES)
    add_hpx_pseudo_target(tests.examples.modules.algorithms)
    add_hpx_pseudo_dependencies(
      tests.examples.modules tests.examples.modules.algorithms
    )
  endif()
else()
  return()
endif()

set(example_programs run_on_all)

foreach(example_program ${example_programs})
  set(sources ${example_program}.cpp)

  source_group("Source Files" FILES ${sources})

  # add example executable
  add_hpx_executable(
    ${example_program} INTERNAL_FLAGS
    SOURCES ${sources} ${${example_program}_FLAGS}
    FOLDER "Examples/Modules/Core/Algorithms"
  )

  add_hpx_example_target_dependencies("modules.algorithms" ${example_program})

  if(HPX_WITH_TESTS AND HPX_WITH_TESTS_EXAMPLES)
    add_hpx_example_test(
      "modules.algorithms" ${example_program} ${${example_program}_PARAMETERS}
    )
  endif()
endforeach()
