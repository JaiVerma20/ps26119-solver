* Small hand LP: maximise with ranged, equality, free and bounded columns.
* max 3x + 2y - z   s.t.  x + y + z <= 4  (L),  x - y in [-1, 2] (range on E),  x + 3y >= 1 (G)
*                         0 <= x <= 3,  y free,  -1 <= z <= 5
NAME          MAXRANGED
OBJSENSE
    MAX
ROWS
 N  obj
 L  c1
 E  c2
 G  c3
COLUMNS
    x         obj       3.0          c1        1.0
    x         c2        1.0          c3        1.0
    y         obj       2.0          c1        1.0
    y         c2        -1.0         c3        3.0
    z         obj       -1.0         c1        1.0
RHS
    rhs       c1        4.0          c2        -1.0
    rhs       c3        1.0          obj       -7.5
RANGES
    rng       c2        3.0
BOUNDS
 UP bnd       x         3.0
 FR bnd       y
 LO bnd       z         -1.0
 UP bnd       z         5.0
ENDATA
