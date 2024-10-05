def triple_product (Ax,Ay,Az,Bx,By,Bz,Cx,Cy,Cz):
    val = Ax * ( By * Cz - Bz * Cy )\
        - Ay * ( Bx * Cz - Bz * Cx )\
        + Az * ( Bx * Cy - By * Cx )
    return val


# internal numbering of the cell
#
#     4---------7
#    /|        /|
#   / |       / |
#  5--|------6  |
#  |  |      |  |
#  |  |      |  |
#  |  0------|--3
#  | /       | /
#  |/        |/
#  1---------2


def hexahedron_volume (x,y,z):
    val = (  triple_product ( x[6]-x[1]+x[7]-x[0], y[6]-y[1]+y[7]-y[0], z[6]-z[1]+z[7]-z[0], \
                              x[6]-x[3],           y[6]-y[3],           z[6]-z[3],           \
                              x[2]-x[0],           y[2]-y[0],           z[2]-z[0]           )\
           + triple_product ( x[7]-x[0],           y[7]-y[0],           z[7]-z[0],           \
                              x[6]-x[3]+x[5]-x[0], y[6]-y[3]+y[5]-y[0], z[6]-z[3]+z[5]-z[0], \
                              x[6]-x[4],           y[6]-y[4],           z[6]-z[4]           )\
           + triple_product ( x[6]-x[1],           y[6]-y[1],           z[6]-z[1],           \
                              x[5]-x[0],           y[5]-y[0],           z[5]-z[0],           \
                              x[6]-x[4]+x[2]-x[0], y[6]-y[4]+y[2]-y[0], z[6]-z[4]+z[2]-z[0]  ) )/12.

    return val


