* 0-1 knapsack. Integer columns via MARKER (A, B) and BV bound (C).
*   maximize 10a + 13b + 7c  s.t.  4a + 6b + 3c <= 10,  a,b,c in {0,1}
* LP relaxation optimum: 23.5 (a=1, b=0.5, c=1). Integer optimum: 23 (a=1, b=1).
NAME          KNAPSACK
OBJSENSE MAX
ROWS
 N  VALUE
 L  WEIGHT
COLUMNS
    MARKER                 'MARKER'                 'INTORG'
    A         VALUE       10.0   WEIGHT       4.0
    B         VALUE       13.0   WEIGHT       6.0
    MARKER                 'MARKER'                 'INTEND'
    C         VALUE        7.0   WEIGHT       3.0
RHS
    RHS       WEIGHT      10.0
BOUNDS
 UP BND       A            1.0
 UP BND       B            1.0
 BV BND       C
ENDATA
