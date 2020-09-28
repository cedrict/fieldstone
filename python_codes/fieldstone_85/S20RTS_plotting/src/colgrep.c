//-------------------------------------------------------------
//  colgrep - displays fields on a given input line,
//  that are separated by one of more spaces:
//
//  prompt> echo a bc  d e  f | colgrep -c 1 5 2 4  
//
//  returns  "a f bc e"
//
//  COLGREP is my inky-dinky (and buggier?)
//  substitute of the UNIX routine
//  "CUT -d ' ' -f field1 field2 ..."
//  until I have figured out how to
//  use CUT when a random number of
//  white spaces separate the fields.
//
//  Jeroen Ritsema, May 2001, Caltech
//-------------------------------------------------------------

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void usage();

main( argc, argv )
int   argc;
char       *argv[];
{
   int       index = 0, space0, sw;
   int       field[50], begch[50], endch[50];
   int       i, j, k, n, sslen;
   char      ss[300];

if (argc < 3) usage(-1);
n = argc - 1;
for(i=0; i<50; ++i) {
  field[i]=-1; begch[i]=-1; endch[i]=-2;
}

while ( ++index < argc && argv[index][0] == '-' ) {
     switch ( argv[index][1] ) {
         case 'c':
             for(i=1; i<n; ++i)
                sscanf( argv[++index], "%d", &field[i] );
             break;
         default:
             usage(-1);
     }
}

while (fgets(ss,300,stdin) != NULL ) {
   sslen = strlen(ss);

   // check whether the line starts with spaces:
   space0 = -1;
   if( ss[0] == ' ') {
     j=0; 
     while(ss[j] == ' ') {
       space0 = j;
       j++;
     }
   }
   k = 1;
   sw = 0;
   for(i=space0+1; i<sslen; ++i) {
     if (sw == 0) {
       if ( ss[i] != ' ') {
         begch[k] = i;
         sw = 1;
       }
     }
     if (sw == 1) {
       if ( ss[i] == ' ' || i == sslen-1 ) {
         endch[k] = i-1;
         sw = 0;
         k++;
       }
     }
   }
   // fprintf(stdout,"%s\n", ss);
   // fprintf(stdout,"BEGIN: %d %d %d %d %d\n",
   //    begch[1], begch[2], begch[3], begch[4], begch[5]);
   // fprintf(stdout,"END:   %d %d %d %d %d\n",
   //    endch[1], endch[2], endch[3], endch[4], endch[5]);

   for (k=1; k<n;++k) {
      for(i=begch[field[k]]; i<=endch[field[k]]; ++i)
          fprintf(stdout,"%c", ss[i]);
      fprintf(stdout," ");
   }
   fprintf(stdout,"\n");
}
exit( 0 );
}

void    usage( exitstatus )
int     exitstatus;
{
   fprintf(stderr,"Usage: colgrep -c field1 field2 ....\n"); 
   exit( exitstatus );
}

