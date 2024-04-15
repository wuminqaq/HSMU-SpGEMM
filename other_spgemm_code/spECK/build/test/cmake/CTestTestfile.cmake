# CMake generated Testfile for 
# Source directory: /home/wm/open_source_code/OpSparse-main/spECK/cub-1.17.2/test/cmake
# Build directory: /home/wm/open_source_code/OpSparse-main/spECK/build/test/cmake
# 
# This file includes the relevant testing commands required for 
# testing this directory and lists subdirectories to be tested as well.
add_test(cub.test.cmake.test_install "/usr/local/bin/cmake" "--log-level=VERBOSE" "-G" "Ninja" "-S" "/home/wm/open_source_code/OpSparse-main/spECK/cub-1.17.2/test/cmake/test_install" "-B" "/home/wm/open_source_code/OpSparse-main/spECK/build/test/cmake/test_install" "-D" "CUB_BINARY_DIR=/home/wm/open_source_code/OpSparse-main/spECK/build" "-D" "CMAKE_CXX_COMPILER=/usr/bin/g++" "-D" "CMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc" "-D" "CMAKE_BUILD_TYPE=Debug")
set_tests_properties(cub.test.cmake.test_install PROPERTIES  _BACKTRACE_TRIPLES "/home/wm/open_source_code/OpSparse-main/spECK/cub-1.17.2/test/cmake/CMakeLists.txt;3;add_test;/home/wm/open_source_code/OpSparse-main/spECK/cub-1.17.2/test/cmake/CMakeLists.txt;0;")
add_test(cub.test.cmake.check_source_files "/usr/local/bin/cmake" "-D" "CUB_SOURCE_DIR=/home/wm/open_source_code/OpSparse-main/spECK/cub-1.17.2" "-P" "/home/wm/open_source_code/OpSparse-main/spECK/cub-1.17.2/test/cmake/check_source_files.cmake")
set_tests_properties(cub.test.cmake.check_source_files PROPERTIES  _BACKTRACE_TRIPLES "/home/wm/open_source_code/OpSparse-main/spECK/cub-1.17.2/test/cmake/CMakeLists.txt;18;add_test;/home/wm/open_source_code/OpSparse-main/spECK/cub-1.17.2/test/cmake/CMakeLists.txt;0;")
