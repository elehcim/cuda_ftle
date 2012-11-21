/* computeFTLE_CUDA.cu
 *
 * Compute an FTLE field for a given vector flow
 * using multiple CUDA-enabled devices. See settings.h
 * for CUDA-related tweaks, and see vectorfield.c 
 * to replace the vector field function.
 *
 * Takes a number of command line arguments,
 * default behavior is to compute using all CUDA >=1.3 
 * cards and output to ftle.data in ASCII and optionally
 * to a 1280x720 ftle.png via PLplot.
 *
 * Raymond Jimenez
 * <raymondj@caltech.edu>
 *
 */

#include "settings.h"
#include "computeFTLE_CUDA.h"

// non-standard to include the file itself,
// but nvcc does not support linking device
// objects together from separate files

#include "RK4.cu"
#include "vectorfield.cu"

__global__ void compute_points(int offset, float *points_grid_device, const float ds, const float t, const float T)
{
  float x[2];
  float output[2]; 
  int N,mask;

  N=(4*offset)+(blockIdx.x*blockDim.x)+threadIdx.x; 


  if(N > (GRID_WIDTH*GRID_HEIGHT*4))
  {
    return;
  }

  x[0] = MIN_X+((float)XPITCH*(int)((N%(GRID_WIDTH*4))/4));
  x[1] = MIN_Y+((float)YPITCH*(int)(N/(GRID_WIDTH*4)));  

  vec_mask(x,t,&mask);
  if(!mask)
  {
    points_grid_device[(2*(N-(4*offset)))] = -1.0/0.0;
    points_grid_device[(2*(N-(4*offset)))+1] = -1.0/0.0;
    return;
  }
  
  if(T == 0.0)
  {
    points_grid_device[(2*(N-(4*offset)))] = x[0];
    points_grid_device[(2*(N-(4*offset)))+1] = x[1];
    return;
  }  



  switch(N%4) {
    case 0:
      x[0]-=ds;
      break;
    case 1:
      x[0]+=ds;
      break;
    case 2:
      x[1]-=ds;
      break;
    case 3:
      x[1]+=ds;
      break;
    default:
      break;
  }

  RK4(x,t,T,output);



  points_grid_device[(2*(N-(4*offset)))] = output[0];
  points_grid_device[(2*(N-(4*offset)))+1] = output[1];


   

  return;

}

__global__ void compute_tracer(float *tracer_grid_device, int nTracers)
{
  int N, mask;
  
  N=(blockIdx.x*blockDim.x)+threadIdx.x;  

  if(N>(nTracers-1))
    return;

  vec_mask(&tracer_grid_device[N*6],tracer_grid_device[(N*6)+4],&mask);

  if(!mask)
  {
    tracer_grid_device[(N*6)+2] = -1.0/0.0;
    tracer_grid_device[(N*6)+3] = -1.0/0.0;
    return;
  }

  RK4(&tracer_grid_device[N*6],tracer_grid_device[(N*6)+4],tracer_grid_device[(N*6)+5],&tracer_grid_device[(N*6)+2]);

  return;

  // That was (not) easy.
}


__global__ void compute_FTLE(int offset, float *points_grid_device, float *grid_device, const float ds, const float t, const float T)
{
  // this is a 2d-specific FTLE calculator
  float ja[2][2],sq[2][2]; // 2x2 matrix for the jacobian
  int N,i;
  float lambda_max; // max eigenvalue of ja^T*ja


  N=(blockIdx.x*blockDim.x)+threadIdx.x;

  if((N+offset) >(GRID_WIDTH*GRID_HEIGHT))
    return;

  if(T==0)
  {
    grid_device[N] = 0.0;
    return;
  }

/* | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 
   x-ds = (0,1)
   x+ds = (2,3)
   y-ds = (4,5)
   y+ds = (6,7)
 
   See flowmap note in computeFTLE_CUDA.h for more explanation.
*/



#pragma unroll 8
  for(i=0;i<8;i++)
  {
    if(points_grid_device[(N*4*2)+i] == -1.0/0.0)
    {
      grid_device[N] = -1.0/0.0;
      return;
    }
  }

  ja[0][0] = (points_grid_device[(N*4*2)+2] -\
              points_grid_device[(N*4*2)+0])/(2.0*ds);

  ja[0][1] = (points_grid_device[(N*4*2)+6] -\
              points_grid_device[(N*4*2)+4])/(2.0*ds);

  ja[1][0] = (points_grid_device[(N*4*2)+3] -\
              points_grid_device[(N*4*2)+1])/(2.0*ds);

  ja[1][1] = (points_grid_device[(N*4*2)+7] -\
              points_grid_device[(N*4*2)+5])/(2.0*ds);

  // this is bad macro magic: make a virtual ja^T.

#define jaT_0_0 ja[0][0]
#define jaT_0_1 ja[1][0]
#define jaT_1_0 ja[0][1]
#define jaT_1_1 ja[1][1]
  
  // calculate ja^T*ja

  sq[0][0] = (jaT_0_0*ja[0][0])+(jaT_0_1*ja[1][0]);
  sq[0][1] = (jaT_0_0*ja[0][1])+(jaT_0_1*ja[1][1]);
  sq[1][0] = (jaT_1_0*ja[0][0])+(jaT_1_1*ja[1][0]);
  sq[1][1] = (jaT_1_0*ja[0][1])+(jaT_1_1*ja[1][1]);

  lambda_max = ((sq[0][0] + sq[1][1])/2.0) + ( (sqrtf((4.0*sq[0][1]*sq[1][0])+((sq[0][0]-sq[1][1])*(sq[0][0]-sq[1][1]))))/2.0);    
    
  grid_device[N] = logf(lambda_max)/(2.0*fabs(T));

  return;

}

void *compute_partial_FTLE(void *args)
{
  cudaError_t err;
  float *grid_device, *points_grid_device;
  struct thread_args *l;
  float *tracer_grid, *tracer_grid_device;
  int nBlocks2, nBlocks3;  
  int nPoints; // number of points this thread deals with
  int i;

  l = (struct thread_args *)args; 

if(verbose)
{ printf("Thread now spawned. offset=%i\n", l->offset); fflush(stdout); }
 
  if(((l->N)-(l->offset) < (l->blocksize*l->nBlocks)) && (l->N)-(l->offset) > 0)
  {
    nPoints = (l->N)-(l->offset);
  } else {
    nPoints = (l->blocksize*l->nBlocks);
  }

if(verbose)
{ printf("Thread still alive..., nPoints, N, nBlocks, blocksize: %i, %i, %i, %i\n", nPoints, l->N, l->nBlocks, l->blocksize); fflush(stdout); }

  if((4*nPoints)%(l->blocksize2) == 0) {
    nBlocks2=((int)((4*nPoints)/l->blocksize2));
  } else {
    nBlocks2=((int)((4*nPoints)/l->blocksize2))+1;
  }

if(verbose)
{ printf("Entering CUDA routines!\n"); fflush(stdout); }

  err = cudaSetDevice(l->device);
  if(err != cudaSuccess) {
    fprintf(stderr, "Something went terribly wrong in starting the device.\n");
    fprintf(stderr, cudaGetErrorString(err));
    exit(1);
  }  

  err = cudaSetDeviceFlags(cudaDeviceBlockingSync);
  if(err != cudaSuccess) {
    fprintf(stderr, "Something went terribly wrong in starting the device.\n");
    fprintf(stderr, cudaGetErrorString(err));
    exit(1);
  }  

  err = cudaMalloc((void **) &grid_device, (int)(nPoints)*sizeof(float)+1);
  if(err != cudaSuccess) {
    fprintf(stderr, "Something went terribly wrong in allocating grid_device.\n");
    fprintf(stderr, cudaGetErrorString(err));
    exit(1);
  }

  err = cudaMalloc((void **) &points_grid_device, 2*4*(nPoints)*sizeof(float)+1);
  if(err != cudaSuccess) {
    fprintf(stderr, "Something went terribly wrong in allocating points_grid_device.\n");
    fprintf(stderr, cudaGetErrorString(err));
    exit(1);
  }



 
  compute_points<<<nBlocks2,l->blocksize2>>>(l->offset,points_grid_device,l->ds,l->t,l->T);

  err = cudaThreadSynchronize();
  if(err != cudaSuccess) {
    fprintf(stderr, "Something went terribly wrong in compute_points. %i %i\n", l->offset, nBlocks2);
    fprintf(stderr, cudaGetErrorString(err));
    exit(1);
  }


  compute_FTLE<<<l->nBlocks,l->blocksize>>>(l->offset,points_grid_device,grid_device,l->ds,l->t,l->T);
  

  err = cudaThreadSynchronize();
  if(err != cudaSuccess) {
    fprintf(stderr, "Something went terribly wrong in compute_FTLE.\n");
    fprintf(stderr, cudaGetErrorString(err));
    exit(1);
  }
  

  cudaMemcpy((void *)&l->grid[l->offset], grid_device, sizeof(float)*nPoints, cudaMemcpyDeviceToHost);

  cudaFree(grid_device);
  cudaFree(points_grid_device);

  if(tracer_enable && (l->nTracers > 0))
  {  
        
    if(l->nTracers%TRACER_BLOCKSIZE == 0) {
      nBlocks3=((int)((l->nTracers)/TRACER_BLOCKSIZE));
    } else {
      nBlocks3=((int)((l->nTracers)/TRACER_BLOCKSIZE))+1;
    }
    
    tracer_grid = (float *)malloc(6*l->nTracers*sizeof(float));
  
    if (tracer_grid==NULL)
    {
      printf("Couldn't allocate tracer_grid.\n");
      exit(1);
    }

    err = cudaMalloc((void **) &tracer_grid_device, 6*(l->nTracers)*sizeof(float));
    if(err != cudaSuccess) {
      fprintf(stderr, "Something went terribly wrong in allocating tracer_grid_device.\n");
      fprintf(stderr, cudaGetErrorString(err));
      exit(1);
    }

    // pack info into a float array because CUDA is a PITA with data types    
    for(i=0;i<l->nTracers;i++)
    {
      tracer_grid[6*i] = l->tracers[i+l->tracer_offset].x0[0];       
      tracer_grid[(6*i)+1] = l->tracers[i+l->tracer_offset].x0[1];       
      tracer_grid[(6*i)+4] = l->tracers[i+l->tracer_offset].z;       
      tracer_grid[(6*i)+5] = l->tracers[i+l->tracer_offset].w;       
    }

    cudaMemcpy(tracer_grid_device, tracer_grid, sizeof(float)*6*l->nTracers, cudaMemcpyHostToDevice);

if(verbose)
{
    printf("Launching tracers in %i blocks of %i...\n", nBlocks3, TRACER_BLOCKSIZE);
    printf("tracer_w: %f; tracer_z: %f\n", l->tracers[0].w, l->tracers[0].z);
    fflush(stdout);
}
    compute_tracer<<<nBlocks3,TRACER_BLOCKSIZE>>>(tracer_grid_device,l->nTracers);

    err = cudaThreadSynchronize();
    if(err != cudaSuccess) {
      fprintf(stderr, "Something went terribly wrong in compute_tracer.\n");
      fprintf(stderr, cudaGetErrorString(err));
      exit(1);
    }

if(verbose)
{ printf("Returned from CUDA tracer...\n", l->nTracers); fflush(stdout); }

    cudaMemcpy(tracer_grid, tracer_grid_device, sizeof(float)*6*l->nTracers, cudaMemcpyDeviceToHost);
 
    for(i=0;i<l->nTracers;i++)
    {
      l->tracers[i+l->tracer_offset].x[0] = tracer_grid[(6*i)+2];       
      l->tracers[i+l->tracer_offset].x[1] = tracer_grid[(6*i)+3];       
    }

    cudaFree(tracer_grid_device);
    free(tracer_grid);
  }
  
  cudaThreadExit(); 

  pthread_exit((void *)0);
}

void print_help(void)
{
  fprintf(stderr, "computeFTLE_CUDA v0.2\n\n\
computeFTLE_CUDA takes up to several command line options:\n\
-v (--verbose) \t\t\tIncrease verbosity (only once).\n\
-d (--data-file) <file>\t\tChange the output ASCII data file (default=ftle.data).\n\
-t (--start-time) <time>\tChange the start time (default t=0).\n\
-T (--integrate-time) <time>\tChange the intergration time (default T=15).\n\
-h (--help)\t\t\tThis is it.\n");

#ifdef USE_PLPLOT
  fprintf(stderr, "-g (--graph-file) <file>\tChange the output plot file (default=ftle.png).\n");
#endif

  fprintf(stderr, "-r (--reverse-enable)\t\tEnable calculating the reverse FTLE in blue (same t,-T)\n\
-s (--reverse-file)\t\tChange the output reverse FTLE data file (default=ftle_rev.data).\n\
-e (--tracer-enable)\t\t\tEnable a grid of tracers that's integrated from a specified time.\n\
-f (--tracer-file)\t\t\tChange the output tracer data file (default=ftle_trac.data).\n\
-x (--tracer-x) <x> \t\t\tNumber of initial X tracers (default 1).\n\
-y (--tracer-y) <y> \t\t\tSet tracer initial Y tracers (default 1).\n\
-z (--tracer-start-time) <time> \tSet tracer start t (default t=0).\n\
-w (--tracer-integrate-time) <time> \tSet integration time for tracer (default T=t).\n");
 
  fprintf(stderr, "\nThere is also a special batch mode. In this mode, the\n\
-s, -f, -d, (-g), -z, and -w flags are invalid. They will be set for you.\n\
-t sets the start time for the batch, -T sets the integration time. If\n\
tracers are set, they will be rendered along according to time.\n\
\n\
The files will be ouput in the current working directory as:\n\
\n\
{n}.data \tFTLE data table\n\
{n}_rev.data \tFTLE reverse data table (if set)\n\
{n}_trac.data \tTracer data table (if set)\n\
{n}.png \tPNG graph (if compiled with USE_PLPLOT)\n\
\n\
where n is the frame number. (NB: Remember what interval\n\
you used!)\n\
\n\
--batch-enable \t\t\t\tEnable batch mode.\n\
--batch-frames <int>\t\t\tSet the number of frames produced.\n\
--batch-interval <float>\t\tSet the delta-t between each frame.\n\
--batch-offset <int>\t\tIf you've interrupted a job, begin from this frame number.\n");
  
  fprintf(stderr, "\nNote that this program will only work on CUDA v1.3 and above\n\
cards by default, so please look at settings.h\n\
to customize to taste.\n");
  fprintf(stderr, "\nWritten by Raymond Jimenez <raymondj@caltech.edu>.\n");
  exit(1);
}

int main(int argc, char** argv)
{

  int x,y,i,rc; // buffers
  float *grid; // FTLE result grid
  float *grid_reverse; // back-FTLE result grid

  const int N = GRID_WIDTH*GRID_HEIGHT;

  const int blocksize = FTLE_BLOCKSIZE;
  int nBlocks;
  const int blocksize2 = POINTS_BLOCKSIZE;


  float t=0.0;
  float T=15.0;
  const float ds=0.001;  

  pthread_t *threads;
  pthread_attr_t thread_attr;
  struct thread_args *args;

  int device_count;
  cudaDeviceProp device_prop;
  int usable_devices=0;
  int dev_num[16]; // 16 devices per machine is reasonable, right?

  char *datafile=NULL;
  char datafile_default[]="ftle.data";
  char *revdatafile=NULL;
  char revdatafile_default[]="ftle_rev.data";
  FILE *fileout;

  struct tracer *tracers; // array of tracer structs!
  int tracer_x = 1;
  int tracer_y = 1;
  float tracer_w = 0.0;
  float tracer_z = 0.0;
  float tracer_xp = 0.0;
  float tracer_yp = 0.0;
  char *tracedatafile=NULL;
  char tracedatafile_default[]="ftle_trac.data";

  int batch_frames = 1;
  int cur_frame = 1;
  float batch_int = 0.0;  

#ifdef USE_PLPLOT
  PLFLT **plgrid;
  PLFLT zmin,zmax;
  PLFLT h[4],s[4],l[4],z[4],a[4];
  
  PLFLT *clevels;


  char *plotfile=NULL;
  char plotfile_default[]="ftle.png";
  FILE *graphout;
  PLFLT plploinx;
  PLFLT plploiny;
  char geometry[256];

#define CLEVEL 127

#endif

  if(N%blocksize == 0)
    nBlocks = ((int)(N/blocksize));
  else
    nBlocks = ((int)(N/blocksize))+1;


  opterr = 1;

#ifdef USE_PLPLOT
  while ((x = getopt_long(argc, argv, "hd:g:t:T:vrs:ef:x:y:z:w:b", &longopts[0], NULL)) != -1)
#else
  while ((x = getopt_long(argc, argv, "hd:t:T:vrs:ef:x:y:z:w:b", &longopts[0], NULL)) != -1)
#endif
  {
    switch (x)
    {
      case 'v':
        verbose = 1;
        break;
      case 'd':
        datafile = optarg;
        break;
      case 't':
        t = strtof(optarg,NULL);
        break;
      case 'T':
        T = strtof(optarg,NULL);
        break;
      case 'h':
        print_help();
        break;
#ifdef USE_PLPLOT
      case 'g':
        plotfile = optarg;
        break;
#endif
      case 'r':
        reverse_enable = 1;
        break;
      case 's':
        revdatafile = optarg;
        break;

      case 'e':
        tracer_enable = 1;
        break;
      case 'f':
        tracedatafile = optarg;
        break;
      case 'w':
        tracer_w = strtof(optarg,NULL);
        break;
      case 'x':
        tracer_x = (int)strtol(optarg,NULL,0);
        break;
      case 'y':
        tracer_y = (int)strtol(optarg,NULL,0);
        break;
      case 'z':
        tracer_z = strtof(optarg,NULL);
        break;

      case 'b':
        batch_enable = 1;
        break;
      case 1:
        batch_frames = (int)strtol(optarg,NULL,0);
        break;
      case 2:
        batch_int = strtof(optarg,NULL);
        break;
      case 3:
        cur_frame = (int)strtol(optarg,NULL,0);
        break;
      case 4:
        benchmark_enable = 1;
        break;

      case '?':
        print_help();
        exit(1);
      default:
        break;
    }
  }

/* validate options */
  if( ((batch_frames!=1) || (batch_int!=0.0)) && !batch_enable )
  {
    fprintf(stderr, "You must enable batch mode to use these flags!\n");
    exit(1);
  }    

  if( ((batch_frames==1) || (batch_int==0.0)) && batch_enable )
  {
    fprintf(stderr, "You must give batch mode the necessary arguments (frames and interval)!\n");
    exit(1);
  }    
  
/* Setup tracers */

  if(tracer_enable && tracer_w==0.0)
  {
    tracer_w=t;
  }

  if(tracer_enable && (!(tracer_y > 0) || !(tracer_x > 0)))  
  {
    fprintf(stderr, "We need at least one tracer in each dimension!\n");
    exit(1);
  } 
 

  if(batch_enable)
  {
    tracer_w=0.0;
    tracer_z=t;
    printf("Entering batch mode...\n");
  }


  if(cur_frame!=1)
  {
    if(tracer_enable)
    {
      fprintf(stderr, "Sorry, resuming tracer runs is not currently possible at this time.\n");
      exit(1);
    }
    for(i=cur_frame;i>1;i--)
    {
      t+=batch_int;
    }
  }

  tracers = (struct tracer *)calloc(tracer_y*tracer_x, sizeof(struct tracer));  
  if (tracers==NULL)
  {
    fprintf(stderr,"Couldn't allocate tracer memory.\n");
    exit(1);
  }

  tracer_xp = (MAX_X - MIN_X)/(float)(tracer_x+1);
  tracer_yp = (MAX_Y - MIN_Y)/(float)(tracer_y+1);

  i = 0;
	
  for(x=1;x<=tracer_x;x++)
  {
    for(y=1;y<=tracer_y;y++) 
    {
      tracers[i].x0[0] = MIN_X+(x*tracer_xp);
      tracers[i].x0[1] = MIN_Y+(y*tracer_yp);
      tracers[i].x_start[0] = tracers[i].x0[0];
      tracers[i].x_start[1] = tracers[i].x0[1];

      tracers[i].w = tracer_w;
      tracers[i].z = tracer_z;
      i++;
    }
  }

/* Setup grids */

  grid=(float *)malloc(N*sizeof(float));

if(reverse_enable)
  grid_reverse=(float *)malloc(N*sizeof(float));


  cudaGetDeviceCount(&device_count);
  for(x=0; x < device_count; x++)
  {
    cudaGetDeviceProperties(&device_prop,x);

if(verbose)
{  printf("Device %i has compute capability %i.%i.\n", x, device_prop.major, device_prop.minor); fflush(stdout); }

    if(device_prop.major>=MIN_MAJOR && device_prop.minor>=MIN_MINOR)
    {
      dev_num[usable_devices]=x;
      usable_devices++;
    }
  } 
 
  if(usable_devices==0) 
  {
    fprintf(stderr,"\tUnfortunately, I could not detect any CUDA 1.3 cards \n\
on your system. We only support 1.3 cards at the moment to ensure equal \n\
distribution of work; if you'd like to proceed anyway, change \n\
min_minor to 0 in the source.\n");
  }

/* From here, we can repeat everything. NB: check for leaks, since over
   long batch runs, leaks can and will hurt. */

while(cur_frame<=batch_frames)
{

/* set up our output filenames */
  if(batch_enable)
  {
    if(reverse_enable)
    { 
/* don't use asprintf since it's a GNU-only extension */
      i = snprintf(NULL, 0, "%i_rev.data", cur_frame);
      revdatafile = (char *)malloc(i*sizeof(char));
      if(!revdatafile) 
      {
        fprintf(stderr,"Error in memory allocation!\n");
        exit(1); 
      }
      snprintf(revdatafile, i, "%i_rev.data", cur_frame);
    }

    if(tracer_enable)
    {
      i = snprintf(NULL, 0, "%i_trac.data", cur_frame);
      tracedatafile = (char *)malloc(i*sizeof(char));
      if(!tracedatafile) 
      {
        fprintf(stderr,"Error in memory allocation!\n");
        exit(1); 
      }
      snprintf(tracedatafile, i, "%i_trac.data", cur_frame);
    }
   
#ifdef USE_PLPLOT
    i = snprintf(NULL, 0, "%i.png", cur_frame);
    plotfile = (char *)malloc(i*sizeof(char));
    if(!plotfile) 
    {
      fprintf(stderr,"Error in memory allocation!\n");
      exit(1); 
    }
    snprintf(plotfile, i, "%i.png", cur_frame);

#endif 

    i = snprintf(NULL, 0, "%i.data", cur_frame);
    datafile = (char *)malloc(i*sizeof(char));
    if(!datafile) 
    {
      fprintf(stderr,"Error in memory allocation!\n");
      exit(1); 
    }
    snprintf(datafile, i, "%i.data", cur_frame);
    
    printf("Creating frame %i...\n", cur_frame);    
    fflush(stdout);
 
  } 
    


/* Begin starting CUDA threads */

  pthread_attr_init(&thread_attr);
  pthread_attr_setdetachstate(&thread_attr, PTHREAD_CREATE_JOINABLE);

  threads=(pthread_t *)malloc(usable_devices*sizeof(pthread_t));
  args=(struct thread_args *)calloc(usable_devices,sizeof(struct thread_args));

  for(x=0;x<nBlocks;x++)
  {
    args[x%usable_devices].nBlocks++;
  }

  for(x=0;x<(tracer_x*tracer_y);x++)
  {
    args[x%usable_devices].nTracers++;
  }

  y=0;
  i=0;
  for(x=0;x<usable_devices;x++)
  {
    args[x].grid = grid;
    args[x].N = N;
    args[x].blocksize = blocksize;
    args[x].blocksize2 = blocksize2;
    args[x].offset = y;
    args[x].t = t;
    args[x].T = T;  
    args[x].ds = ds;  
    args[x].device = dev_num[x];
    y+=(args[x].nBlocks*args[x].blocksize);

    args[x].tracers = tracers;
    args[x].tracer_offset = i;
    i+=args[x].nTracers;

if(verbose)
{ printf("Launching thread:\tN: %i, nBlocks: %i, blocksize: %i, blocksize2: %i\n\t\t\toffset: %i, device: %i\n\t\t\tnTracers: %i, Tracer offset: %i\n",\
           args[x].N, args[x].nBlocks, args[x].blocksize, args[x].blocksize2, args[x].offset, args[x].device, args[x].nTracers, args[x].tracer_offset); 
  fflush(stdout);
}
    rc = pthread_create(&threads[x], NULL, compute_partial_FTLE, (void *)(&args[x]));
    if(rc)
    { 
      fprintf(stderr,"Thread failed to start on device %i.\n", args[x].device);
      exit(1);
    }
  }

  pthread_attr_destroy(&thread_attr);
  for(x=0;x<usable_devices;x++)
  {
    rc = pthread_join(threads[x], NULL);
  }
	
  free(threads);


if(verbose)
{ printf("Done with FTLE computation; CUDA threads returned.\n"); fflush(stdout); }

if(reverse_enable)
{
/* If we are calculating the reverse FTLE, too, do it all again! */

  pthread_attr_init(&thread_attr);
  pthread_attr_setdetachstate(&thread_attr, PTHREAD_CREATE_JOINABLE);


  threads=(pthread_t *)malloc(usable_devices*sizeof(pthread_t));

  y=0;
  i=0;
  for(x=0;x<usable_devices;x++)
  {
    args[x].grid = grid_reverse;
    args[x].T = -T;  

    args[x].nTracers=0;

if(verbose)
{  printf("Launching reverse thread:\tN: %i, nBlocks: %i, blocksize: %i, blocksize2: %i\n\t\t\toffset: %i, device: %i\n\t\t\tnTracers: %i, Tracer offset: %i\n",\
           args[x].N, args[x].nBlocks, args[x].blocksize, args[x].blocksize2, args[x].offset, args[x].device, args[x].nTracers, args[x].tracer_offset); 
   fflush(stdout);
}
    rc = pthread_create(&threads[x], NULL, compute_partial_FTLE, (void *)(&args[x])); 
    if(rc)
    { 
      fprintf(stderr,"Reverse thread failed to start on device %i.\n", args[x].device);
      exit(1);
    }
  }

  pthread_attr_destroy(&thread_attr);
  for(x=0;x<usable_devices;x++)
  {
    rc = pthread_join(threads[x], NULL);
  }
	
  free(threads);
/* End reverse FTLE */

if(verbose)
{  printf("Done with reverse FTLE computation; reverse CUDA threads returned.\n"); fflush(stdout); }

}  


  free(args);

/* End of CUDA-related calls */

/* print FTLE fields to file for later use/graph_data */

if(!benchmark_enable)
{
  if(datafile==NULL)
    datafile=&datafile_default[0];

  fileout=fopen(datafile,"w");
  if (fileout==NULL)
  {
    printf("Couldn't open data file for writing.\n");
    exit(1);
  }

  for(x=0;x<GRID_WIDTH;x++) 
  {
    for(y=0;y<GRID_HEIGHT;y++)
    {
      fprintf(fileout,"%f\t%f\t%f\n", MIN_X+x*XPITCH,MIN_Y+y*YPITCH, grid[GRIDXY(y,x)]);
    }
  }
  fclose(fileout);

if(reverse_enable)
{
  if(revdatafile==NULL)
    revdatafile=&revdatafile_default[0];

  fileout=fopen(revdatafile,"w");
  if (fileout==NULL)
  {
    printf("Couldn't open data file for writing.\n");
    exit(1);
  }

  for(x=0;x<GRID_WIDTH;x++) 
  {
    for(y=0;y<GRID_HEIGHT;y++)
    {
      fprintf(fileout,"%f\t%f\t%f\n", MIN_X+x*XPITCH,MIN_Y+y*YPITCH, grid_reverse[GRIDXY(y,x)]);
    }
  }
  fclose(fileout);
  
}

if(tracer_enable)  
{
  if(tracedatafile==NULL)
    tracedatafile=&tracedatafile_default[0];

  fileout=fopen(tracedatafile,"w");
  if (fileout==NULL)
  {
    printf("Couldn't open tracer data file for writing.\n");
    exit(1);
  }

  for(i=(tracer_x*tracer_y)-1;i>-1;i--)
  {
    fprintf(fileout,"%i\t%f\t%f\t%f\t%f\n", i, tracers[i].x_start[0], tracers[i].x_start[1], tracers[i].x[0], tracers[i].x[1]);
  }

  fclose(fileout);
}

}
/* Begin plotting code, skipped entirely if USE_PLPLOT is undefined */ 


#ifdef USE_PLPLOT


  if(plotfile==NULL)
    plotfile=&plotfile_default[0];
  
  graphout = fopen (plotfile,"w");

  if (graphout==NULL)
  {
    printf("Couldn't open graph file for writing.\n");
    exit(1);
  }

  plsdev("png"); // use PNG driver for now, though xwin is possible
  plsfile(graphout); 
  
  snprintf(geometry, 256, "%ix%i", (int)(1.2*GRID_WIDTH), (int)(1.2*GRID_HEIGHT)); 
  plsetopt("geometry", geometry);

  plinit();
   
  plAlloc2dGrid(&plgrid,GRID_WIDTH,GRID_HEIGHT);
  
  if(plgrid==NULL)
    { printf("could not alloc plgrid\n");  exit(1); }

  for(x=0;x<GRID_WIDTH;x++) 
  {
    for(y=0;y<GRID_HEIGHT;y++)
    {
      plgrid[x][y]=grid[GRIDXY(y,x)];
    }
  }
  

  plMinMax2dGrid(plgrid,GRID_WIDTH,GRID_HEIGHT,&zmax,&zmin);

  clevels = (PLFLT *)calloc(CLEVEL+1, sizeof(PLFLT));  
  if((zmax-zmin)<0.01)
    fprintf(stderr, "Warning: all FTLE values are within +/-0.01, specifically %f.\n\
Graph scaling may make things look funny; the graph is normallized to\n\
the min and max FTLE value, so rounding errors now appear as mountains.\n", (zmax-zmin));

  for(i=0;i<=CLEVEL;i++)
  {
    clevels[i] = zmin + (i*((zmax-zmin)/(PLFLT)CLEVEL)); 
  }

  z[0] = 0.0;
  z[1] = 0.0;
  z[2] = 0.0;
  z[3] = 1.0;

  h[0] = 0.0;
  h[1] = 0.0;
  h[2] = 0.0;
  h[3] = 0.0;

  l[0] = 0.0;
  l[1] = 0.0;
  l[2] = 1.0;
  l[3] = 0.5;

  s[0] = 1.0;
  s[1] = 1.0;
  s[2] = 1.0;
  s[3] = 1.0;
  plscmap1l(0, 4, z, h, l, s, NULL);


  pladv(0);
  plvpor(0.1, 0.9, 0.1, 0.9); // give ourselves a margin of .1 on all sides
  plwind(MIN_X,MAX_X,MIN_Y,MAX_Y); // window settings here

  plpsty(0); // no line fills

  plshades(plgrid, GRID_WIDTH,GRID_HEIGHT,NULL,\
           (PLFLT)MIN_X,(PLFLT)MAX_X,\
           (PLFLT)MIN_Y,(PLFLT)MAX_Y,\
           clevels,CLEVEL+1,2,0,0,plfill,1,NULL,NULL); // see x16c.c


  plcol0(1);






  

  plFree2dGrid(plgrid,GRID_WIDTH,GRID_HEIGHT);
  free((void *)clevels);

/* Now repeat the above again, but do it for the reverse FTLE field */

if(reverse_enable)
{
  
  plAlloc2dGrid(&plgrid,GRID_WIDTH,GRID_HEIGHT);
  
  if(plgrid==NULL)
    { printf("could not alloc plgrid\n");  exit(1); }

  for(x=0;x<GRID_WIDTH;x++) 
  {
    for(y=0;y<GRID_HEIGHT;y++)
    {
      plgrid[x][y]=grid_reverse[GRIDXY(y,x)];
    }
  }
  

  plMinMax2dGrid(plgrid,GRID_WIDTH,GRID_HEIGHT,&zmax,&zmin);

  clevels = (PLFLT *)calloc(CLEVEL+1, sizeof(PLFLT));  
  if((zmax-zmin)<0.01)
    fprintf(stderr, "Warning: all reverse FTLE values are within +/-0.01, specifically %f.\n\
Graph scaling may make things look funny; the graph is normallized to\n\
the min and max FTLE value, so rounding errors now appear as mountains.\n", (zmax-zmin));

  for(i=0;i<=CLEVEL;i++)
  {
    clevels[i] = zmin + (i*((zmax-zmin)/(PLFLT)CLEVEL)); 
  }

  z[0] = 0.0;
  z[1] = 0.0;
  z[2] = 0.0;
  z[3] = 1.0;

  h[0] = 244.0;
  h[1] = 244.0;
  h[2] = 244.0;
  h[3] = 244.0;

  l[0] = 0.0;
  l[1] = 0.0;
  l[2] = 1.0;
  l[3] = 0.5;

  s[0] = 1.0;
  s[1] = 1.0;
  s[2] = 1.0;
  s[3] = 1.0;

  a[0] = 0.0;
  a[1] = 0.0;
  a[2] = 0.0;
  a[3] = 1.0;
  plscmap1la(0, 4, z, h, l, s, a, NULL);

  plshades(plgrid, GRID_WIDTH,GRID_HEIGHT,NULL,\
           (PLFLT)MIN_X,(PLFLT)MAX_X,\
           (PLFLT)MIN_Y,(PLFLT)MAX_Y,\
           clevels,CLEVEL+1,2,0,0,plfill,1,NULL,NULL); // see x16c.c


}

  plcol0(1);

  plbox("bcnst", 0.0, 0, "bcnstv", 0.0, 0);

  plcol0(8);

  if(tracer_enable)
  {
    for(i=(tracer_x*tracer_y)-1;i>-1;i--)
    {
      plploinx = tracers[i].x[0];
      plploiny = tracers[i].x[1];

      plpoin(1,&plploinx,&plploiny,20);
    }

  }

  plend();

#endif



  if(batch_enable)
  {
    if(reverse_enable)
      free(revdatafile);

    if(tracer_enable)
      free(tracedatafile); 
  
#ifdef USE_PLPLOT
    free(plotfile);
#endif
    free(datafile);
  
    t+=batch_int;
    tracer_w=batch_int;
	 
    for(i=(tracer_x*tracer_y)-1;i>-1;i--)
    {
      tracers[i].x0[0] = tracers[i].x[0];
      tracers[i].x0[1] = tracers[i].x[1];

      tracers[i].z = t;
      tracers[i].w = batch_int;

    }


  }

  cur_frame++; 
  

/* end main batch loop */
}

  if(tracer_enable)
    free(tracers);

  if(reverse_enable)
    free(grid_reverse);

  free(grid);

  exit(0);
}

