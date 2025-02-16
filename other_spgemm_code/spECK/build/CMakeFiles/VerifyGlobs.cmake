# CMAKE generated file: DO NOT EDIT!
# Generated by CMake Version 3.27
cmake_policy(SET CMP0009 NEW)

# test_srcs at test/CMakeLists.txt:75 (file)
file(GLOB NEW_GLOB LIST_DIRECTORIES true RELATIVE "/home/wm/open_source_code/OpSparse-main/spECK/cub-1.17.2/test" "/home/wm/open_source_code/OpSparse-main/spECK/cub-1.17.2/test/test_*.cu")
set(OLD_GLOB
  "test_allocator.cu"
  "test_block_adjacent_difference.cu"
  "test_block_histogram.cu"
  "test_block_load_store.cu"
  "test_block_merge_sort.cu"
  "test_block_radix_sort.cu"
  "test_block_reduce.cu"
  "test_block_run_length_decode.cu"
  "test_block_scan.cu"
  "test_block_shuffle.cu"
  "test_device_adjacent_difference.cu"
  "test_device_histogram.cu"
  "test_device_merge_sort.cu"
  "test_device_radix_sort.cu"
  "test_device_reduce.cu"
  "test_device_reduce_by_key.cu"
  "test_device_run_length_encode.cu"
  "test_device_scan.cu"
  "test_device_scan_by_key.cu"
  "test_device_segmented_sort.cu"
  "test_device_select_if.cu"
  "test_device_select_unique.cu"
  "test_device_select_unique_by_key.cu"
  "test_device_spmv.cu"
  "test_device_three_way_partition.cu"
  "test_grid_barrier.cu"
  "test_iterator.cu"
  "test_iterator_deprecated.cu"
  "test_namespace_wrapped.cu"
  "test_temporary_storage_layout.cu"
  "test_thread_sort.cu"
  "test_warp_exchange.cu"
  "test_warp_load.cu"
  "test_warp_mask.cu"
  "test_warp_merge_sort.cu"
  "test_warp_reduce.cu"
  "test_warp_scan.cu"
  "test_warp_store.cu"
  )
if(NOT "${NEW_GLOB}" STREQUAL "${OLD_GLOB}")
  message("-- GLOB mismatch!")
  file(TOUCH_NOCREATE "/home/wm/open_source_code/OpSparse-main/spECK/build/CMakeFiles/cmake.verify_globs")
endif()

# headers at cmake/CubHeaderTesting.cmake:10 (file)
file(GLOB_RECURSE NEW_GLOB LIST_DIRECTORIES false RELATIVE "/home/wm/open_source_code/OpSparse-main/spECK/cub-1.17.2/cub" "/home/wm/open_source_code/OpSparse-main/spECK/cub-1.17.2/cub/*.cuh")
set(OLD_GLOB
  "agent/agent_adjacent_difference.cuh"
  "agent/agent_histogram.cuh"
  "agent/agent_merge_sort.cuh"
  "agent/agent_radix_sort_downsweep.cuh"
  "agent/agent_radix_sort_histogram.cuh"
  "agent/agent_radix_sort_onesweep.cuh"
  "agent/agent_radix_sort_upsweep.cuh"
  "agent/agent_reduce.cuh"
  "agent/agent_reduce_by_key.cuh"
  "agent/agent_rle.cuh"
  "agent/agent_scan.cuh"
  "agent/agent_scan_by_key.cuh"
  "agent/agent_segment_fixup.cuh"
  "agent/agent_segmented_radix_sort.cuh"
  "agent/agent_select_if.cuh"
  "agent/agent_spmv_orig.cuh"
  "agent/agent_sub_warp_merge_sort.cuh"
  "agent/agent_three_way_partition.cuh"
  "agent/agent_unique_by_key.cuh"
  "agent/single_pass_scan_operators.cuh"
  "block/block_adjacent_difference.cuh"
  "block/block_discontinuity.cuh"
  "block/block_exchange.cuh"
  "block/block_histogram.cuh"
  "block/block_load.cuh"
  "block/block_merge_sort.cuh"
  "block/block_radix_rank.cuh"
  "block/block_radix_sort.cuh"
  "block/block_raking_layout.cuh"
  "block/block_reduce.cuh"
  "block/block_run_length_decode.cuh"
  "block/block_scan.cuh"
  "block/block_shuffle.cuh"
  "block/block_store.cuh"
  "block/radix_rank_sort_operations.cuh"
  "block/specializations/block_histogram_atomic.cuh"
  "block/specializations/block_histogram_sort.cuh"
  "block/specializations/block_reduce_raking.cuh"
  "block/specializations/block_reduce_raking_commutative_only.cuh"
  "block/specializations/block_reduce_warp_reductions.cuh"
  "block/specializations/block_scan_raking.cuh"
  "block/specializations/block_scan_warp_scans.cuh"
  "block/specializations/block_scan_warp_scans2.cuh"
  "block/specializations/block_scan_warp_scans3.cuh"
  "config.cuh"
  "cub.cuh"
  "detail/choose_offset.cuh"
  "detail/device_double_buffer.cuh"
  "detail/device_synchronize.cuh"
  "detail/exec_check_disable.cuh"
  "detail/temporary_storage.cuh"
  "detail/type_traits.cuh"
  "device/device_adjacent_difference.cuh"
  "device/device_histogram.cuh"
  "device/device_merge_sort.cuh"
  "device/device_partition.cuh"
  "device/device_radix_sort.cuh"
  "device/device_reduce.cuh"
  "device/device_run_length_encode.cuh"
  "device/device_scan.cuh"
  "device/device_segmented_radix_sort.cuh"
  "device/device_segmented_reduce.cuh"
  "device/device_segmented_sort.cuh"
  "device/device_select.cuh"
  "device/device_spmv.cuh"
  "device/dispatch/dispatch_adjacent_difference.cuh"
  "device/dispatch/dispatch_histogram.cuh"
  "device/dispatch/dispatch_merge_sort.cuh"
  "device/dispatch/dispatch_radix_sort.cuh"
  "device/dispatch/dispatch_reduce.cuh"
  "device/dispatch/dispatch_reduce_by_key.cuh"
  "device/dispatch/dispatch_rle.cuh"
  "device/dispatch/dispatch_scan.cuh"
  "device/dispatch/dispatch_scan_by_key.cuh"
  "device/dispatch/dispatch_segmented_sort.cuh"
  "device/dispatch/dispatch_select_if.cuh"
  "device/dispatch/dispatch_spmv_orig.cuh"
  "device/dispatch/dispatch_three_way_partition.cuh"
  "device/dispatch/dispatch_unique_by_key.cuh"
  "grid/grid_barrier.cuh"
  "grid/grid_even_share.cuh"
  "grid/grid_mapping.cuh"
  "grid/grid_queue.cuh"
  "host/mutex.cuh"
  "iterator/arg_index_input_iterator.cuh"
  "iterator/cache_modified_input_iterator.cuh"
  "iterator/cache_modified_output_iterator.cuh"
  "iterator/constant_input_iterator.cuh"
  "iterator/counting_input_iterator.cuh"
  "iterator/discard_output_iterator.cuh"
  "iterator/tex_obj_input_iterator.cuh"
  "iterator/tex_ref_input_iterator.cuh"
  "iterator/transform_input_iterator.cuh"
  "thread/thread_load.cuh"
  "thread/thread_operators.cuh"
  "thread/thread_reduce.cuh"
  "thread/thread_scan.cuh"
  "thread/thread_search.cuh"
  "thread/thread_sort.cuh"
  "thread/thread_store.cuh"
  "util_allocator.cuh"
  "util_arch.cuh"
  "util_compiler.cuh"
  "util_cpp_dialect.cuh"
  "util_debug.cuh"
  "util_deprecated.cuh"
  "util_device.cuh"
  "util_macro.cuh"
  "util_math.cuh"
  "util_namespace.cuh"
  "util_ptx.cuh"
  "util_type.cuh"
  "version.cuh"
  "warp/specializations/warp_reduce_shfl.cuh"
  "warp/specializations/warp_reduce_smem.cuh"
  "warp/specializations/warp_scan_shfl.cuh"
  "warp/specializations/warp_scan_smem.cuh"
  "warp/warp_exchange.cuh"
  "warp/warp_load.cuh"
  "warp/warp_merge_sort.cuh"
  "warp/warp_reduce.cuh"
  "warp/warp_scan.cuh"
  "warp/warp_store.cuh"
  )
if(NOT "${NEW_GLOB}" STREQUAL "${OLD_GLOB}")
  message("-- GLOB mismatch!")
  file(TOUCH_NOCREATE "/home/wm/open_source_code/OpSparse-main/spECK/build/CMakeFiles/cmake.verify_globs")
endif()

# example_srcs at examples/block/CMakeLists.txt:1 (file)
file(GLOB_RECURSE NEW_GLOB LIST_DIRECTORIES false RELATIVE "/home/wm/open_source_code/OpSparse-main/spECK/cub-1.17.2/examples/block" "/home/wm/open_source_code/OpSparse-main/spECK/cub-1.17.2/examples/block/example_*.cu")
set(OLD_GLOB
  "example_block_radix_sort.cu"
  "example_block_reduce.cu"
  "example_block_reduce_dyn_smem.cu"
  "example_block_scan.cu"
  )
if(NOT "${NEW_GLOB}" STREQUAL "${OLD_GLOB}")
  message("-- GLOB mismatch!")
  file(TOUCH_NOCREATE "/home/wm/open_source_code/OpSparse-main/spECK/build/CMakeFiles/cmake.verify_globs")
endif()

# example_srcs at examples/device/CMakeLists.txt:1 (file)
file(GLOB_RECURSE NEW_GLOB LIST_DIRECTORIES false RELATIVE "/home/wm/open_source_code/OpSparse-main/spECK/cub-1.17.2/examples/device" "/home/wm/open_source_code/OpSparse-main/spECK/cub-1.17.2/examples/device/example_*.cu")
set(OLD_GLOB
  "example_device_partition_flagged.cu"
  "example_device_partition_if.cu"
  "example_device_radix_sort.cu"
  "example_device_reduce.cu"
  "example_device_scan.cu"
  "example_device_select_flagged.cu"
  "example_device_select_if.cu"
  "example_device_select_unique.cu"
  "example_device_sort_find_non_trivial_runs.cu"
  )
if(NOT "${NEW_GLOB}" STREQUAL "${OLD_GLOB}")
  message("-- GLOB mismatch!")
  file(TOUCH_NOCREATE "/home/wm/open_source_code/OpSparse-main/spECK/build/CMakeFiles/cmake.verify_globs")
endif()
