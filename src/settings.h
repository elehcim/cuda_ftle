// define USE_PLPLOT to turn on PLPLOT-related functions

// #define USE_PLPLOT

#ifdef USE_PLPLOT

#include <plplot/plplot.h>

#endif

// configuration for graph and layout

#define GRID_WIDTH 200
#define GRID_HEIGHT 200

#define MIN_X -0.8
#define MAX_X -0.2
#define MIN_Y -2.0
#define MAX_Y 2.0

// minimum CUDA card specs

#define MIN_MAJOR 1
#define MIN_MINOR 1

// CUDA blocksize for the RK4 alg 
// and the FTLE calculations. 
//
// tweak these only after changing vec()
// and running through the profiler.

#define POINTS_BLOCKSIZE 448
#define FTLE_BLOCKSIZE 512
#define TRACER_BLOCKSIZE 128

