* Exercises most MPS features in one small model.
*   minimize  x1 + 2 x2 - x3 + x4 + x5 + 10     (RHS on COST = -10 -> offset +10)
*   LIM1   L  x1 + x2       <= 4
*   LIM2   G  x1            >= 1
*   MYEQN  E  -x2 + x3       = 7
*   RNGEQ  E  x3 + x4  in [2, 5]    (E row, range +3)
*   RNGL   L  x4 + x5  in [-5, -1]  (L row, range 4)
*   bounds: x1 in [0,4], x2 in (-inf,1] (MI + UP), x3 in [-1,8],
*           x4 fixed at 0.5, x5 UP -1 with default lower -> (-inf,-1]
* Known optimum: -6.5  (x1=1, x2=-5.5, x3=1.5, x4=0.5, x5=-5.5)
NAME          FEATURES
ROWS
 N  COST
 L  LIM1
 G  LIM2
 E  MYEQN
 E  RNGEQ
 L  RNGL
COLUMNS
    X1        COST         1.0   LIM1         1.0
    X1        LIM2         1.0
    X2        COST         2.0   LIM1         1.0
    X2        MYEQN       -1.0
    X3        COST        -1.0   MYEQN        1.0
    X3        RNGEQ        1.0
    X4        COST         1.0   RNGEQ        1.0
    X4        RNGL         1.0
    X5        COST         1.0   RNGL         1.0
RHS
    RHS       COST       -10.0
    RHS       LIM1         4.0   LIM2         1.0
    RHS       MYEQN        7.0   RNGEQ        2.0
    RHS       RNGL        -1.0
RANGES
    RNG       RNGEQ        3.0   RNGL         4.0
BOUNDS
 UP BND       X1           4.0
 MI BND       X2
 UP BND       X2           1.0
 LO BND       X3          -1.0
 UP BND       X3           8.0
 FX BND       X4           0.5
 UP BND       X5          -1.0
ENDATA
