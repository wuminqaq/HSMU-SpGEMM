// This source file checks that:
// 1) Header <cub/device/dispatch/dispatch_unique_by_key.cuh> compiles without error.
// 2) Common macro collisions with platform/system headers are avoided.

// Define CUB_MACRO_CHECK(macro, header), which emits a diagnostic indicating
// a potential macro collision and halts.
//
// Use raw platform checks instead of the CUB_HOST_COMPILER macros since we
// don't want to #include any headers other than the one being tested.
//
// This is only implemented for MSVC/GCC/Clang.
#if defined(_MSC_VER) // MSVC

// Fake up an error for MSVC
#define CUB_MACRO_CHECK_IMPL(msg)                                              \
  /* Print message that looks like an error: */                                \
  __pragma(message(__FILE__ ":" CUB_MACRO_CHECK_IMPL0(__LINE__)                \
                   ": error: " #msg))                                          \
  /* abort compilation due to static_assert or syntax error: */                \
  static_assert(false, #msg);
#define CUB_MACRO_CHECK_IMPL0(x) CUB_MACRO_CHECK_IMPL1(x)
#define CUB_MACRO_CHECK_IMPL1(x) #x

#elif defined(__clang__) || defined(__GNUC__)

// GCC/clang are easy:
#define CUB_MACRO_CHECK_IMPL(msg) CUB_MACRO_CHECK_IMPL0(GCC error #msg)
#define CUB_MACRO_CHECK_IMPL0(expr) _Pragma(#expr)

#endif

// Hacky way to build a string, but it works on all tested platforms.
#define CUB_MACRO_CHECK(MACRO, HEADER)                                         \
  CUB_MACRO_CHECK_IMPL(Identifier MACRO should not be used from CUB            \
                       headers due to conflicts with HEADER macros.)

// complex.h conflicts
#define I CUB_MACRO_CHECK('I', complex.h)

// windows.h conflicts
#define small CUB_MACRO_CHECK('small', windows.h)
// We can't enable these checks without breaking some builds -- some standard
// library implementations unconditionally `#undef` these macros, which then
// causes random failures later.
// Leaving these commented out as a warning: Here be dragons.
//#define min(...) CUB_MACRO_CHECK('min', windows.h)
//#define max(...) CUB_MACRO_CHECK('max', windows.h)

// termios.h conflicts (NVIDIA/thrust#1547)
#define B0 CUB_MACRO_CHECK("B0", termios.h)

#include <cub/device/dispatch/dispatch_unique_by_key.cuh>
