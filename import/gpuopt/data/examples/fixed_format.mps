* FIXED-format MPS: names contain spaces, so only column positions
* can separate the fields. Free-format parsing fails and the reader
* falls back to fixed format automatically.
*   minimize -5 p1 - 4 p2   s.t.  p1 + p2 <= 40,  2 p1 + p2 <= 60
* Known optimum: -180  (p1 = 20, p2 = 20)
NAME          FIXED FORMAT DEMO
ROWS
 N  PROFIT
 L  MACH A
 L  MACH B
COLUMNS
    PROD 1    PROFIT            -5.0   MACH A             1.0
    PROD 1    MACH B             2.0
    PROD 2    PROFIT            -4.0   MACH A             1.0
    PROD 2    MACH B             1.0
RHS
    RHS       MACH A            40.0   MACH B            60.0
ENDATA
