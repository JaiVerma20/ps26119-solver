* Toy refinery crude-selection model (units: kt of crude per day).
*   Buy crude A (low sulphur, cost 60) and crude B (high sulphur, cost 45).
*   minimize  60 A + 45 B                         (crude purchase cost)
*   PETROL:   0.40 A + 0.30 B >= 12               (petrol demand)
*   DIESEL:   0.50 A + 0.40 B >= 15               (diesel demand)
*   SULPHUR:  0.5 A + 2.5 B <= 1.5 (A + B)  ->  -A + B <= 0   (avg sulphur <= 1.5 %)
*   CAPACITY: A + B <= 50                         (distillation unit capacity)
*   bounds:   0 <= A <= 30 (contract limit), B >= 0
* Known optimum: 1800  (A = B = 120/7 = 17.142857; petrol demand and sulphur limit bind)
NAME          REFINERY_BLEND
ROWS
 N  COST
 G  PETROL
 G  DIESEL
 L  SULPHUR
 L  CAPACITY
COLUMNS
    CRUDE_A   COST        60.0   PETROL       0.40
    CRUDE_A   DIESEL      0.50   SULPHUR     -1.0
    CRUDE_A   CAPACITY     1.0
    CRUDE_B   COST        45.0   PETROL       0.30
    CRUDE_B   DIESEL      0.40   SULPHUR      1.0
    CRUDE_B   CAPACITY     1.0
RHS
    RHS       PETROL      12.0   DIESEL      15.0
    RHS       CAPACITY    50.0
BOUNDS
 UP BND       CRUDE_A     30.0
ENDATA
