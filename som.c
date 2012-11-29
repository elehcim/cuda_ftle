#include <stdio.h>
#include <math.h>
vec(float *x, float t, float *xout)
{
#define mu 0.1
#define mu1 1-mu
#define mu2 mu

#define r3 pow(((x[0]+mu2)*(x[0]+mu2)+x[1]*x[1]),1.5)     // r: distance to m1, LARGER MASS
#define R3 pow(((x[0]-mu1)*(x[0]-mu1)+x[1]*x[1]),1.5)      // R: distance to m2, smaller mass

#define Ux -x[0]+mu1*(x[0]+mu2)/r3+mu2*(x[0]-mu1)/R3
#define Uy -x[1]+mu1*x[1]/r3+mu2*x[1]/R3

xout[0] = x[2];
xout[1] = x[3];
xout[2] = 2.0*x[3]-Ux;
xout[3] = -2.0*x[4]-Uy;
}

main()
{
float xout[4],x[4],t;
printf("Dimmi 4 numeri e t");
scanf("%f %f %f %f %f",&x[0],&x[1],&x[2],&x[3],&t);
vec(x,t,xout);
printf("risposta [%f, %f, %f, %f] ",xout[0],xout[1],xout[2],xout[3]);
}
