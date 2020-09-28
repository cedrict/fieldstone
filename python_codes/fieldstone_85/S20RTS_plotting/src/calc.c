//
// calc -
// simple math in csh scripts
// (a selection of utilities only)
//
// Jeroen Ritsema August 2000, Caltech

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

void usage();

main( argc, argv )
int   argc;
char       *argv[];
{
    double value1, value2;
    double rad = 0.0174532;
    char   oper;
    char   opers[5];
    char   ss[120];


if (argc == 4) {
   sscanf(argv[1],"%lf", &value1);
   sscanf(argv[2],"%c",   &oper);
   sscanf(argv[3],"%lf",  &value2);

   if (oper == 'x') fprintf(stdout,"%e\n", value1*value2);
   if (oper == '/') fprintf(stdout,"%e\n", value1/value2);
   if (oper == '+') fprintf(stdout,"%e\n", value1+value2);
   if (oper == '-') fprintf(stdout,"%e\n", value1-value2);
   if (oper == 'm') fprintf(stdout,"%e\n", fmod(value1,value2));
   
   if (oper == '>') {
     if (value1 > value2) {
             fprintf(stdout,"%d\n", 1);
     } else {
             fprintf(stdout,"%d\n", 0);
     }
   }
   if (oper == '<') {
     if (value1 < value2) {
             fprintf(stdout,"%d\n", 1);
     } else {
             fprintf(stdout,"%d\n", 0);
     }
   }
} else if (argc == 3) {
   sscanf(argv[1],"%s",  &opers);
   sscanf(argv[2],"%lf", &value1);

   if (strncmp(opers,"e2",2)==0) fprintf(stdout,"%.2e\n", value1*1.0);
   if (strncmp(opers,"e3",2)==0) fprintf(stdout,"%.3e\n", value1*1.0);
   if (strncmp(opers,"int",3)==0) fprintf(stdout,"%d\n",  (int) rintf(value1));
   if (strncmp(opers,"dbl",3)==0) fprintf(stdout,"%lf\n", value1*1.0);
   if (strncmp(opers,"3f1",3)==0) fprintf(stdout,"%3.1f\n", value1*1.0);
   if (strncmp(opers,"4f1",3)==0) fprintf(stdout,"%4.1f\n", value1*1.0);
   if (strncmp(opers,"4f2",3)==0) fprintf(stdout,"%4.2f\n", value1*1.0);
   if (strncmp(opers,"5f1",3)==0) fprintf(stdout,"%5.1f\n", value1*1.0);
   if (strncmp(opers,"6f1",3)==0) fprintf(stdout,"%6.1f\n", value1*1.0);
   if (strncmp(opers,"7f2",3)==0) fprintf(stdout,"%7.2f\n", value1*1.0);
   if (strncmp(opers,"8f3",3)==0) fprintf(stdout,"%8.3f\n", value1*1.0);
   if (strncmp(opers,"9f2",3)==0) fprintf(stdout,"%9.2f\n", value1*1.0);
   if (strncmp(opers,"9f4",3)==0) fprintf(stdout,"%9.4f\n", value1*1.0);
   if (strncmp(opers,"9f6",3)==0) fprintf(stdout,"%9.6f\n", value1*1.0);
   if (strncmp(opers,"02d",3)==0) fprintf(stdout,"%02d\n", (int) rintf(value1));
   if (strncmp(opers,"03d",3)==0) fprintf(stdout,"%03d\n", (int) rintf(value1));
   if (strncmp(opers,"06d",3)==0) fprintf(stdout,"%06d\n", (int) rintf(value1));
   if (strncmp(opers,"abs",3)==0) fprintf(stdout,"%lf\n", fabs(value1));
   if (strncmp(opers,"abs",3)==0) fprintf(stdout,"%lf\n", fabs(value1));
   if (strncmp(opers,"sqrt",4)==0) fprintf(stdout,"%lf\n", sqrt(value1));
   if (strncmp(opers,"10f4",4)==0) fprintf(stdout,"%10.4f\n", value1*1.0);
   if (strncmp(opers,"dcos",4)==0) fprintf(stdout,"%lf\n", cos(value1*rad));
   if (strncmp(opers,"dsin",4)==0) fprintf(stdout,"%lf\n", sin(value1*rad));
   if (strncmp(opers,"log10",5)==0) fprintf(stdout,"%lf\n", log10(value1));
} else {
   usage(-1);
}

exit( 0 );
}

void    usage( exitstatus )
int     exitstatus;
{
   fprintf(stderr,"Usage: (3 input parameters) \n"); 
   fprintf(stderr,"calc X x Y \n"); 
   fprintf(stderr,"calc X / Y \n"); 
   fprintf(stderr,"calc X + Y \n"); 
   fprintf(stderr,"calc X m Y (fmod[x,y] \n"); 
   fprintf(stderr,"calc X \\< Y (=1 if X<Y, else =0)\n"); 
   fprintf(stderr,"calc X \\> Y (=1 if X>Y, else =0)\n"); 
   fprintf(stderr,"       (2 input parameters)\n"); 
   fprintf(stderr,"calc int X (nearest integer value of X)\n"); 
   fprintf(stderr,"calc dbl X (double float of X)\n"); 
   fprintf(stderr,"calc 3f1 X (prints X with [3.1f] format)\n"); 
   fprintf(stderr,"calc 4f1 X (prints X with [4.1f] format)\n"); 
   fprintf(stderr,"calc 4f2 X (prints X with [4.2f] format)\n"); 
   fprintf(stderr,"calc 5f1 X (prints X with [5.1f] format)\n"); 
   fprintf(stderr,"calc 6f1 X (prints X with [6.1f] format)\n"); 
   fprintf(stderr,"calc 7f2 X (prints X with [7.2f] format)\n"); 
   fprintf(stderr,"calc 8f3 X (prints X with [8.3f] format)\n"); 
   fprintf(stderr,"calc 9f2 X (prints X with [9.2f] format)\n"); 
   fprintf(stderr,"calc 9f4 X (prints X with [9.4f] format)\n"); 
   fprintf(stderr,"calc 9f6 X (prints X with [9.6f] format)\n"); 
   fprintf(stderr,"calc 10f4 X (prints X with [10.4f] format)\n"); 
   fprintf(stderr,"calc e2  X (prints X with [.2e] format)\n"); 
   fprintf(stderr,"calc e3  X (prints X with [.3e] format)\n"); 
   fprintf(stderr,"calc 02d X (prints X with [2d] format)\n"); 
   fprintf(stderr,"calc 03d X (prints X with [3d] format)\n"); 
   fprintf(stderr,"calc 06d X (prints X with [6d] format)\n"); 
   fprintf(stderr,"calc srt X (square-root of X)\n"); 
   fprintf(stderr,"calc abs X (absolute value of X)\n"); 
   fprintf(stderr,"calc dcos X (cosine of X [in deg])\n"); 
   fprintf(stderr,"calc dsin X (sine of X [in deg])\n"); 
// fprintf(stderr,"calc exp10 X (10**X)\n"); 
   fprintf(stderr,"calc log10 X (log10 of X)\n");
   exit( exitstatus );
}

