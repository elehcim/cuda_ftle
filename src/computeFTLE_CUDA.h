#ifndef computeFTLE_CUDA_H
#define computeFTLE_CUDA_H

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <getopt.h>
#include <ctype.h>

#include <cuda.h>
#include <pthread.h>

#define XPITCH ((float)((MAX_X-MIN_X)/(float)GRID_WIDTH))
#define YPITCH ((float)((MAX_Y-MIN_Y)/(float)GRID_HEIGHT))

#define GRIDXY(i,j) (((i)*GRID_WIDTH)+(j))
#define GRIDXYZA(i,j,k,a) (((i)*GRID_WIDTH*4*2)+(2*4*(j))+(2*(k))+a)
 
// specifically, use GRIDXYZA for the evolved pointmap
// element 0 of a point = x-ds
// element 1 of a point = x+ds
// element 2 of a point = y-ds
// element 3 of a point = y+ds

/* flowmap is organized like this:
  _ _ _
 |_|_|_| ...
 |_|_|_| ...
  . . .
  . . .
  . . .

  Each cell has four elements in the beginning:
   _____
  |  .  | x_x+ds x_y+ds
  |. x .| x_x-ds x_y-ds
  |__.__|


  We evolve this forward and then take the Jacobian.

  The Jacobian looks like:

  J=( dx_x/dx  dx_x/dy
      dx_y/dx  dx_y/dy )

*/

struct tracer {
  float x_start[2]; // initial x, only different in batch mode
  float x0[2]; // start x at cur_time
  float x[2]; // end x at cur_time+w
  float z; // starting time
  float w; // int time
} tracer;


struct thread_args {
  float *grid;
  int N;
  int nBlocks;
  int blocksize;
  int blocksize2;
  int device;
  int offset;
  float ds;
  float t;
  float T;
  struct tracer *tracers;
  int nTracers;
  int tracer_offset;
} thread_args;

int verbose = 0; // verbosity flag
int tracer_enable = 0; // are we tracing something?
int reverse_enable = 0; // are we calculating the reverse FTLE, too?
int batch_enable = 0; // are we running on dummy and running in a batch mode?
int benchmark_enable = 0; // disable file write-out so we can get an accurate
                          // speed count

const static struct option longopts[] = {
 {"verbose", no_argument, &verbose, 'v'},
 {"data-file", required_argument, NULL, 'd'},
 {"start-time", required_argument, NULL, 't'},
 {"integrate-time", required_argument, NULL, 'T'},
 {"help", no_argument, NULL, 'h'},
#ifdef USE_PLPLOT
 {"graph-file", required_argument, NULL, 'g'},
#endif 
 {"reverse-enable", no_argument, &reverse_enable, 'r'},
 {"reverse-file", required_argument, NULL, 's'},
 {"tracer-enable", no_argument, &tracer_enable, 'e'},
 {"tracer-file", required_argument, NULL, 'f'},
 {"tracer-x", required_argument, NULL, 'x'},
 {"tracer-y", required_argument, NULL, 'y'},
 {"tracer-start-time", required_argument, NULL, 'z'},
 {"tracer-integrate-time", required_argument, NULL, 'w'},
 {"batch-enable", no_argument, &batch_enable, 'b'},
 {"batch-frames", required_argument, NULL, 1},
 {"batch-interval", required_argument, NULL, 2},
 {"batch-offset", required_argument, NULL, 3},
 {"benchmark", no_argument, &benchmark_enable, 4},
 {0, 0, 0, 0}

}; 

#endif

