#include <stdio.h>
#include <stdlib.h>

#include <math.h>

//undefine the following to turn off PLPLOT-related functions

// #define USE_PLPLOT

#ifdef USE_PLPLOT

#include <plplot/plplot.h>

#endif

#define MAX(a, b) (a > b ? a : b)

// time-dependent double gyre

inline void vec(float *x, float t, float *xout)
{ 
  
//  float *vecout; // vecout[0] = x', vecout[1] = y'
  const float A=0.1;
  const float eps=0.25;
  const float ome=(2.0/10.0)*M_PI;

#define a(t) (eps*sinf(ome*(t)))
#define b(t) (1-(2*eps*sinf(ome*(t))))
#define f(x,t) ((a(t)*(x)*(x))+(b(t)*(x)))
#define df(x,t) ((2*a(t)*(x))+b(t))

  xout[0] = -1.0*M_PI*A*sinf(M_PI*f(x[0],t))*cosf(M_PI*x[1]);
  xout[1] = M_PI*A*cosf(M_PI*f(x[0],t))*sinf(M_PI*x[1])*df(x[0],t);

//  xout[0] = 0.0;
//  xout[1] = 0.0;


  return;
}

/* vec takes a pointer to a float[2] array and returns
   x' and y' according to the simple pendulum phasespace.
   
   cheap and easy and verifiable example for RK4 
*/

inline void vec2(float *x, float t, float *xout)
{ 
  
  //system for a simple pendulum   

//  xout[0] = x[1];
//  xout[1] = -1.0*sinf(x[0]);

  xout[0] = x[0]; 
  xout[1] = -x[1];

  return;
}


/* RK4 does one step of RK4 with step h, only handles T>0 for now */

float *RK4(void (*func)(float *, float, float *), float *x, float t, float T, float *output)
{
  // NB func must be in this form:
  // void func(float *x, float t, float *xout)
  // where *x is a float[2] representing x y
  // and *xout[2] represents x' y'

  float k1[2], k2[2], k3[2], k4[2]; // Runge-Kutta terms
  float xpos[2], xnext[2]; // temporary x buffers 
 
  // emperically "good enough" stepize 
  float h = 0.001;

  float i;

  // initialize our buffer with our initial state
  xpos[0] = x[0];
  xpos[1] = x[1];

  xnext[0] = x[0];
  xnext[1] = x[1];

  for(i=fabsf(T/h);i>0;i-=1.0)
  {
//    printf("calling func: %f, %f, %f\n", xnext[0], xnext[1],t); 
    func(xnext, t, k1);

    xpos[0] = xnext[0]+(.5*h*k1[0]);
    xpos[1] = xnext[1]+(.5*h*k1[1]);
    func(xpos, t+(.5*h), k2);

    xpos[0] = xnext[0]+(.5*h*k2[0]);
    xpos[1] = xnext[1]+(.5*h*k2[1]);
    func(xpos, t+(.5*h), k3);

    xpos[0] = xnext[0]+(h*k3[0]);
    xpos[1] = xnext[1]+(h*k3[1]);
    func(xpos, t+h, k4);
  
    xpos[0] = xnext[0];
    xpos[1] = xnext[1];

    xnext[0] = xpos[0]+((1.0/6.0)*h*(k1[0]+(2.0*k2[0])+(2.0*k3[0])+k4[0]));
    xnext[1] = xpos[1]+((1.0/6.0)*h*(k1[1]+(2.0*k2[1])+(2.0*k3[1])+k4[1]));

    t+=h;
  }

  output[0] = xnext[0];
  output[1] = xnext[1];

  return output;
}

float compute_FTLE(void (*vec)(float *, float, float *), float *x, float t, float T)
{
  // this is a 2d-specific FTLE calculator
  float ja[2][2],jaT[2][2],sq[2][2]; // 2x2 matrix for the jacobian
  int i,j;
  float xlow[2], xhigh[2]; // position buffers
  float lambda1, lambda2; // eigenvalues of our end matrix
  float ds=0.001; // what is our step to deterimine the jacobian?
  float xlout[2], xhout[2]; // xout for low and high
  float ftle;

  // following logic copied from numJacobian in coherentStruct.py
  // fixed code; jacobian now computed correctly

  for (i=0;i<2;i++)
  { 
    for (j=0;j<2;j++)
    {
      xlow[0]=x[0];
      xlow[1]=x[1];
      xlow[j]-=ds;
      
      RK4(vec,xlow,t,T,xlout);

      xhigh[0]=x[0];
      xhigh[1]=x[1];
      xhigh[j]+=ds;
      
      RK4(vec,xhigh,t,T,xhout);

      
      ja[i][j] = (xhout[i]-xlout[i])/(2.0*ds);
          
       
    }
  }

  
//  printf("Jacobian: [[%f, %f], [%f, %f]]\n", ja[0][0], ja[0][1], ja[1][0], ja[1][1]);

  
  // setup the transpose

  jaT[0][0] = ja[0][0];
  jaT[0][1] = ja[1][0];
  jaT[1][0] = ja[0][1];
  jaT[1][1] = ja[1][1];
 
  // do sq = jaT*ja

  sq[0][0] = (jaT[0][0]*ja[0][0])+(jaT[0][1]*ja[1][0]);
  sq[0][1] = (jaT[0][0]*ja[0][1])+(jaT[0][1]*ja[1][1]);
  sq[1][0] = (jaT[1][0]*ja[0][0])+(jaT[1][1]*ja[1][0]);
  sq[1][1] = (jaT[1][0]*ja[0][1])+(jaT[1][1]*ja[1][1]);

  lambda1 = ((sq[0][0] + sq[1][1])/2.0) + ( (sqrtf((4*sq[0][1]*sq[1][0])+((sq[0][0]-sq[1][1])*(sq[0][0]-sq[1][1]))))/2.0);    
//  lambda2 = ((sq[0][0] + sq[1][1])/2.0) - ( (sqrtf((4*sq[0][1]*sq[1][0])+((sq[0][0]-sq[1][1])*(sq[0][0]-sq[1][1]))))/2.0);    
  

  ftle = (1.0/(2.0*T))*logf(lambda1);
 
//  printf("FTLE: %f\n", ftle); 

  return ftle;
}


int main(int argc, char* args[])
{
#define GRID_WIDTH 40
#define GRID_HEIGHT 20

#define MIN_X 0.0
#define MAX_X 2.0
#define MIN_Y 0.0
#define MAX_Y 1.0

#define XPITCH ((float)((MAX_X-MIN_X)/(float)GRID_WIDTH))
#define YPITCH ((float)((MAX_Y-MIN_Y)/(float)GRID_HEIGHT))

#define GRIDXY(i,j) ((i*GRID_WIDTH)+j)

  float vecinit[2];
  int x,y,i; // buffers
  float *grid;
  float *gridptr;


#ifdef USE_PLPLOT
  PLFLT **plgrid;
  PLFLT zmin,zmax;
  
  PLFLT *clevels;

#define CLEVEL 255

#endif
  FILE *fileout;
  fileout = fopen ("ftle_1.data","w");
  if (fileout==NULL)
  {
    printf("Couldn't open file for writing.\n");
    exit(1);
  }

  grid=(float *)malloc(sizeof(float)*GRID_WIDTH*GRID_HEIGHT);

  gridptr=grid;


  for(y=0;y<GRID_HEIGHT;y++)
  {
//    printf("Row %i...\n", y);
    for(x=0;x<GRID_WIDTH;x++)  
    {     
      vecinit[0] = MIN_X+(x*XPITCH);
      vecinit[1] = MIN_Y+(y*YPITCH);
      *gridptr = compute_FTLE(vec,vecinit,0,15);
      fprintf(fileout, "%f\t%f\t%f\n",vecinit[0],vecinit[1],*gridptr);
//      printf("%f\t%f\t%f\n",vecinit[0],vecinit[1],*gridptr);
      gridptr++;
    }
  }

  

//  vecinit[0] = 1.8;
//  vecinit[1] = 0.6;

//  printf("FTLE for (%f, %f): %f\n", vecinit[0], vecinit[1], compute_FTLE(vec,vecinit,0,15);

#ifdef USE_PLPLOT

  plsdev("xwin"); // use PNG driver for now, though xwin is possible

  plinit();
   
  plAlloc2dGrid(&plgrid,GRID_WIDTH,GRID_HEIGHT);
  
  if(plgrid==NULL)
    { printf("could not alloc plgrid\n");  exit(1); }

  

  for(x=0;x<GRID_WIDTH;x++) 
  {
    for(y=0;y<GRID_HEIGHT;y++)
    {
      plgrid[x][y]=*(grid+GRIDXY(y,x));
    }
  }

  
  plMinMax2dGrid(plgrid,GRID_WIDTH,GRID_HEIGHT,&zmax,&zmin);

  clevels = (PLFLT *)calloc(CLEVEL, sizeof(PLFLT));  
  for(i=0;i<CLEVEL;i++)
  {
    clevels[i] = zmin + ((zmax-zmin) * ((PLFLT)i) / ((PLFLT)CLEVEL)); 
  }


  pladv(0);
  plvpor(0.1, 0.9, 0.1, 0.9); // give ourselves a margin of .1 on all sides
  plwind(0.,2.,0.,1.); // window settings here

  plpsty(0); // no line fills

  plshades(plgrid, GRID_WIDTH,GRID_HEIGHT,NULL,\
           (PLFLT) 0.,(PLFLT) 2.,\
           (PLFLT) 0.,(PLFLT) 1.,\
           clevels,CLEVEL+1,2,0,0,plfill,1,NULL,NULL); // see x16c.c

  plcol0(1);
  plbox("bcnst", 0.0, 0, "bcnstv", 0.0, 0);
  plcol0(2);
  
  plFree2dGrid(plgrid,GRID_WIDTH,GRID_HEIGHT);
  free((void *)clevels);

  plend();

#endif


  free(grid);
  fclose(fileout);

  exit(0);
}
