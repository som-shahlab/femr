#pragma once

#ifdef __GNU__

#include <malloc.h>

inline void malloc_trim_wrapper() { malloc_trim(0); }

#else

inline void malloc_trim_wrapper() {}

#endif
