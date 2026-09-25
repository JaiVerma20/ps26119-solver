* Small 0/1 knapsack MILP for the verifier's MILP mode (optimum 28: HiGHS and ps26119 branch-and-bound agree).
* max 12a + 7b + 9c + 4d + 11e  s.t.  5a + 3b + 4c + 2d + 6e <= 12,  a..e binary
NAME          KNAPMIP
OBJSENSE
    MAX
ROWS
 N  obj
 L  cap
COLUMNS
    MARKER                 'MARKER'                 'INTORG'
    a         obj       12.0         cap       5.0
    b         obj       7.0          cap       3.0
    c         obj       9.0          cap       4.0
    d         obj       4.0          cap       2.0
    e         obj       11.0         cap       6.0
    MARKER                 'MARKER'                 'INTEND'
RHS
    rhs       cap       12.0
BOUNDS
 UP bnd       a         1.0
 UP bnd       b         1.0
 UP bnd       c         1.0
 UP bnd       d         1.0
 UP bnd       e         1.0
ENDATA
