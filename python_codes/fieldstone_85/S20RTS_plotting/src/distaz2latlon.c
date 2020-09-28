#include	<stdio.h>
#include        <stdlib.h>
#include	<math.h>

#define deg_to_rad 0.017453292
#define rad_to_deg 57.29577951

float           lat1, lon1, delta, az;


main(argc, argv)
	int             argc;
	char           *argv[];
{
        float           lat1, lon1, delta, az;
	float		theta, ct, st, phi, cp, sp, bearing;
	float		cb, sb, cd, sd, earth_radius;
	float		ez, ex, ey, lat2, lon2;

        if (argc != 5) {
          fprintf(stderr,"Usage: distaz2latlon Lat1 Lon1 dist az\n");
          fprintf(stderr,"       Returns:  Lat2 Lon2\n");
          exit(1);
        } else {
          lat1  = atof(argv[1]);
          lon1  = atof(argv[2]);
          delta = atof(argv[3]);
          az    = atof(argv[4]);
        }

	earth_radius = 6378.2064;
	theta = deg_to_rad * lat1;
	ct = cos(theta);
	st = sin(theta);
	phi = deg_to_rad * lon1;
	cp = cos(phi);
	sp = sin(phi);
	bearing = deg_to_rad * az;
        delta = deg_to_rad * delta;
	cb = cos(bearing);
	sb = sin(bearing);
	cd = cos(delta);
	sd = sin(delta);
	ez = cd * st + sd * cb * ct;
	ey = cd * ct * cp + sd * (-cb * st * cp - sb * sp);
	ex = cd * ct * sp + sd * (-cb * st * sp + sb * cp);

	lat2 = (float) (rad_to_deg * atan2(ez, sqrt(ex * ex + ey * ey)));
	lon2 = (float) (rad_to_deg * atan2(ex, ey));

	fprintf(stdout,"%8.3f %8.3f\n", lat2, lon2);

exit(0);
}				/* end main */
