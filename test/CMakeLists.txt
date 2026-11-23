find_program(Julia_EXECUTABLE julia)

if(NOT Julia_EXECUTABLE)
  message(STATUS "Julia executable not found; skipping Julia tests.")
  return()
endif()

# Both tests load the library just built through load(path), so they run
# against it whatever nlolib_jll provides.
add_test(NAME test_julia_api_smoke
  COMMAND "${Julia_EXECUTABLE}" "--project=${CMAKE_CURRENT_LIST_DIR}/.." "${CMAKE_CURRENT_LIST_DIR}/runtests.jl"
          "$<TARGET_FILE:nlolib>"
)

add_test(NAME test_julia_example_soliton
  COMMAND "${Julia_EXECUTABLE}" "--project=${CMAKE_CURRENT_LIST_DIR}/../examples"
          "-e"
          "using Pkg; Pkg.instantiate(); include(raw\"${CMAKE_CURRENT_LIST_DIR}/../examples/second_order_soliton.jl\"); NonlinearOptics.load(raw\"$<TARGET_FILE:nlolib>\"); main(String[])"
)

set(_NLOLIB_TEST_PATH "$ENV{PATH}")
string(REPLACE ";" "\;" _NLOLIB_TEST_PATH "${_NLOLIB_TEST_PATH}")

set_tests_properties(test_julia_api_smoke test_julia_example_soliton PROPERTIES
  ENVIRONMENT "PATH=${_NLOLIB_TEST_PATH}"
)
