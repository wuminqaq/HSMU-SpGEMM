# CMAKE generated file: DO NOT EDIT!
# Generated by CMake Version 3.22
cmake_policy(SET CMP0009 NEW)

# example_srcs at CMakeLists.txt:1 (file)
file(GLOB_RECURSE NEW_GLOB FOLLOW_SYMLINKS LIST_DIRECTORIES false RELATIVE "/home/wm/open_source_code/NHC_SPGEMM/cub-1.17/examples/block" "/home/wm/open_source_code/NHC_SPGEMM/cub-1.17/examples/block/example_*.cu")
set(OLD_GLOB
  "example_block_radix_sort.cu"
  "example_block_reduce.cu"
  "example_block_reduce_dyn_smem.cu"
  "example_block_scan.cu"
  )
if(NOT "${NEW_GLOB}" STREQUAL "${OLD_GLOB}")
  message("-- GLOB mismatch!")
  file(TOUCH_NOCREATE "/home/wm/open_source_code/NHC_SPGEMM/cub-1.17/examples/block/CMakeFiles/cmake.verify_globs")
endif()
