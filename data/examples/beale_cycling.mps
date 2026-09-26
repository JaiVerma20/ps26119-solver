* Beale (1955): the classic LP on which the textbook simplex with
* Dantzig's rule cycles forever. Bland's rule must terminate.
*   minimize -0.75 x4 + 150 x5 - 0.02 x6 + 6 x7
*   R1: 0.25 x4 -  60 x5 - 0.04 x6 + 9 x7 <= 0
*   R2: 0.50 x4 -  90 x5 - 0.02 x6 + 3 x7 <= 0
*   R3:                          x6       <= 1
* Known optimum: -0.05  (x4 = 0.04, x6 = 1)
NAME          BEALE
ROWS
 N  OBJ
 L  R1
 L  R2
 L  R3
COLUMNS
    X4        OBJ        -0.75   R1          0.25
    X4        R2          0.5
    X5        OBJ        150.0   R1         -60.0
    X5        R2         -90.0
    X6        OBJ        -0.02   R1         -0.04
    X6        R2         -0.02   R3           1.0
    X7        OBJ          6.0   R1           9.0
    X7        R2           3.0
RHS
    RHS       R3           1.0
ENDATA
