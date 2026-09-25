* Smallest possible demo: a two-product production plan.
*   maximize 3x + 2y
*   s.t.     x +  y <= 4
*            x + 3y <= 6
*            0 <= x <= 3,  y >= 0
* Known optimum: x = 3, y = 1, objective = 11
NAME          TINY_MAX
OBJSENSE
    MAX
ROWS
 N  PROFIT
 L  C1
 L  C2
COLUMNS
    X         PROFIT       3.0   C1           1.0
    X         C2           1.0
    Y         PROFIT       2.0   C1           1.0
    Y         C2           3.0
RHS
    RHS       C1           4.0   C2           6.0
BOUNDS
 UP BND       X            3.0
ENDATA
