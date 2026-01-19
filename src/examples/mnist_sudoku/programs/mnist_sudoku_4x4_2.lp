% MNIST Sudoku rules (normal rules only) for 4x4

digit(1,1,1) :- not digit(1,1,2), not digit(1,1,3), not digit(1,1,4).
digit(1,1,2) :- not digit(1,1,1), not digit(1,1,3), not digit(1,1,4).
digit(1,1,3) :- not digit(1,1,1), not digit(1,1,2), not digit(1,1,4).
digit(1,1,4) :- not digit(1,1,1), not digit(1,1,2), not digit(1,1,3).
digit(1,2,1) :- not digit(1,2,2), not digit(1,2,3), not digit(1,2,4).
digit(1,2,2) :- not digit(1,2,1), not digit(1,2,3), not digit(1,2,4).
digit(1,2,3) :- not digit(1,2,1), not digit(1,2,2), not digit(1,2,4).
digit(1,2,4) :- not digit(1,2,1), not digit(1,2,2), not digit(1,2,3).
digit(1,3,1) :- not digit(1,3,2), not digit(1,3,3), not digit(1,3,4).
digit(1,3,2) :- not digit(1,3,1), not digit(1,3,3), not digit(1,3,4).
digit(1,3,3) :- not digit(1,3,1), not digit(1,3,2), not digit(1,3,4).
digit(1,3,4) :- not digit(1,3,1), not digit(1,3,2), not digit(1,3,3).
digit(1,4,1) :- not digit(1,4,2), not digit(1,4,3), not digit(1,4,4).
digit(1,4,2) :- not digit(1,4,1), not digit(1,4,3), not digit(1,4,4).
digit(1,4,3) :- not digit(1,4,1), not digit(1,4,2), not digit(1,4,4).
digit(1,4,4) :- not digit(1,4,1), not digit(1,4,2), not digit(1,4,3).
digit(2,1,1) :- not digit(2,1,2), not digit(2,1,3), not digit(2,1,4).
digit(2,1,2) :- not digit(2,1,1), not digit(2,1,3), not digit(2,1,4).
digit(2,1,3) :- not digit(2,1,1), not digit(2,1,2), not digit(2,1,4).
digit(2,1,4) :- not digit(2,1,1), not digit(2,1,2), not digit(2,1,3).
digit(2,2,1) :- not digit(2,2,2), not digit(2,2,3), not digit(2,2,4).
digit(2,2,2) :- not digit(2,2,1), not digit(2,2,3), not digit(2,2,4).
digit(2,2,3) :- not digit(2,2,1), not digit(2,2,2), not digit(2,2,4).
digit(2,2,4) :- not digit(2,2,1), not digit(2,2,2), not digit(2,2,3).
digit(2,3,1) :- not digit(2,3,2), not digit(2,3,3), not digit(2,3,4).
digit(2,3,2) :- not digit(2,3,1), not digit(2,3,3), not digit(2,3,4).
digit(2,3,3) :- not digit(2,3,1), not digit(2,3,2), not digit(2,3,4).
digit(2,3,4) :- not digit(2,3,1), not digit(2,3,2), not digit(2,3,3).
digit(2,4,1) :- not digit(2,4,2), not digit(2,4,3), not digit(2,4,4).
digit(2,4,2) :- not digit(2,4,1), not digit(2,4,3), not digit(2,4,4).
digit(2,4,3) :- not digit(2,4,1), not digit(2,4,2), not digit(2,4,4).
digit(2,4,4) :- not digit(2,4,1), not digit(2,4,2), not digit(2,4,3).
digit(3,1,1) :- not digit(3,1,2), not digit(3,1,3), not digit(3,1,4).
digit(3,1,2) :- not digit(3,1,1), not digit(3,1,3), not digit(3,1,4).
digit(3,1,3) :- not digit(3,1,1), not digit(3,1,2), not digit(3,1,4).
digit(3,1,4) :- not digit(3,1,1), not digit(3,1,2), not digit(3,1,3).
digit(3,2,1) :- not digit(3,2,2), not digit(3,2,3), not digit(3,2,4).
digit(3,2,2) :- not digit(3,2,1), not digit(3,2,3), not digit(3,2,4).
digit(3,2,3) :- not digit(3,2,1), not digit(3,2,2), not digit(3,2,4).
digit(3,2,4) :- not digit(3,2,1), not digit(3,2,2), not digit(3,2,3).
digit(3,3,1) :- not digit(3,3,2), not digit(3,3,3), not digit(3,3,4).
digit(3,3,2) :- not digit(3,3,1), not digit(3,3,3), not digit(3,3,4).
digit(3,3,3) :- not digit(3,3,1), not digit(3,3,2), not digit(3,3,4).
digit(3,3,4) :- not digit(3,3,1), not digit(3,3,2), not digit(3,3,3).
digit(3,4,1) :- not digit(3,4,2), not digit(3,4,3), not digit(3,4,4).
digit(3,4,2) :- not digit(3,4,1), not digit(3,4,3), not digit(3,4,4).
digit(3,4,3) :- not digit(3,4,1), not digit(3,4,2), not digit(3,4,4).
digit(3,4,4) :- not digit(3,4,1), not digit(3,4,2), not digit(3,4,3).
digit(4,1,1) :- not digit(4,1,2), not digit(4,1,3), not digit(4,1,4).
digit(4,1,2) :- not digit(4,1,1), not digit(4,1,3), not digit(4,1,4).
digit(4,1,3) :- not digit(4,1,1), not digit(4,1,2), not digit(4,1,4).
digit(4,1,4) :- not digit(4,1,1), not digit(4,1,2), not digit(4,1,3).
digit(4,2,1) :- not digit(4,2,2), not digit(4,2,3), not digit(4,2,4).
digit(4,2,2) :- not digit(4,2,1), not digit(4,2,3), not digit(4,2,4).
digit(4,2,3) :- not digit(4,2,1), not digit(4,2,2), not digit(4,2,4).
digit(4,2,4) :- not digit(4,2,1), not digit(4,2,2), not digit(4,2,3).
digit(4,3,1) :- not digit(4,3,2), not digit(4,3,3), not digit(4,3,4).
digit(4,3,2) :- not digit(4,3,1), not digit(4,3,3), not digit(4,3,4).
digit(4,3,3) :- not digit(4,3,1), not digit(4,3,2), not digit(4,3,4).
digit(4,3,4) :- not digit(4,3,1), not digit(4,3,2), not digit(4,3,3).
digit(4,4,1) :- not digit(4,4,2), not digit(4,4,3), not digit(4,4,4).
digit(4,4,2) :- not digit(4,4,1), not digit(4,4,3), not digit(4,4,4).
digit(4,4,3) :- not digit(4,4,1), not digit(4,4,2), not digit(4,4,4).
digit(4,4,4) :- not digit(4,4,1), not digit(4,4,2), not digit(4,4,3).


bad :- digit(1,1,1), digit(1,2,1), not bad.
bad :- digit(1,1,1), digit(1,3,1), not bad.
bad :- digit(1,1,1), digit(1,4,1), not bad.
bad :- digit(1,2,1), digit(1,3,1), not bad.
bad :- digit(1,2,1), digit(1,4,1), not bad.
bad :- digit(1,3,1), digit(1,4,1), not bad.
bad :- digit(1,1,2), digit(1,2,2), not bad.
bad :- digit(1,1,2), digit(1,3,2), not bad.
bad :- digit(1,1,2), digit(1,4,2), not bad.
bad :- digit(1,2,2), digit(1,3,2), not bad.
bad :- digit(1,2,2), digit(1,4,2), not bad.
bad :- digit(1,3,2), digit(1,4,2), not bad.
bad :- digit(1,1,3), digit(1,2,3), not bad.
bad :- digit(1,1,3), digit(1,3,3), not bad.
bad :- digit(1,1,3), digit(1,4,3), not bad.
bad :- digit(1,2,3), digit(1,3,3), not bad.
bad :- digit(1,2,3), digit(1,4,3), not bad.
bad :- digit(1,3,3), digit(1,4,3), not bad.
bad :- digit(1,1,4), digit(1,2,4), not bad.
bad :- digit(1,1,4), digit(1,3,4), not bad.
bad :- digit(1,1,4), digit(1,4,4), not bad.
bad :- digit(1,2,4), digit(1,3,4), not bad.
bad :- digit(1,2,4), digit(1,4,4), not bad.
bad :- digit(1,3,4), digit(1,4,4), not bad.
bad :- digit(2,1,1), digit(2,2,1), not bad.
bad :- digit(2,1,1), digit(2,3,1), not bad.
bad :- digit(2,1,1), digit(2,4,1), not bad.
bad :- digit(2,2,1), digit(2,3,1), not bad.
bad :- digit(2,2,1), digit(2,4,1), not bad.
bad :- digit(2,3,1), digit(2,4,1), not bad.
bad :- digit(2,1,2), digit(2,2,2), not bad.
bad :- digit(2,1,2), digit(2,3,2), not bad.
bad :- digit(2,1,2), digit(2,4,2), not bad.
bad :- digit(2,2,2), digit(2,3,2), not bad.
bad :- digit(2,2,2), digit(2,4,2), not bad.
bad :- digit(2,3,2), digit(2,4,2), not bad.
bad :- digit(2,1,3), digit(2,2,3), not bad.
bad :- digit(2,1,3), digit(2,3,3), not bad.
bad :- digit(2,1,3), digit(2,4,3), not bad.
bad :- digit(2,2,3), digit(2,3,3), not bad.
bad :- digit(2,2,3), digit(2,4,3), not bad.
bad :- digit(2,3,3), digit(2,4,3), not bad.
bad :- digit(2,1,4), digit(2,2,4), not bad.
bad :- digit(2,1,4), digit(2,3,4), not bad.
bad :- digit(2,1,4), digit(2,4,4), not bad.
bad :- digit(2,2,4), digit(2,3,4), not bad.
bad :- digit(2,2,4), digit(2,4,4), not bad.
bad :- digit(2,3,4), digit(2,4,4), not bad.
bad :- digit(3,1,1), digit(3,2,1), not bad.
bad :- digit(3,1,1), digit(3,3,1), not bad.
bad :- digit(3,1,1), digit(3,4,1), not bad.
bad :- digit(3,2,1), digit(3,3,1), not bad.
bad :- digit(3,2,1), digit(3,4,1), not bad.
bad :- digit(3,3,1), digit(3,4,1), not bad.
bad :- digit(3,1,2), digit(3,2,2), not bad.
bad :- digit(3,1,2), digit(3,3,2), not bad.
bad :- digit(3,1,2), digit(3,4,2), not bad.
bad :- digit(3,2,2), digit(3,3,2), not bad.
bad :- digit(3,2,2), digit(3,4,2), not bad.
bad :- digit(3,3,2), digit(3,4,2), not bad.
bad :- digit(3,1,3), digit(3,2,3), not bad.
bad :- digit(3,1,3), digit(3,3,3), not bad.
bad :- digit(3,1,3), digit(3,4,3), not bad.
bad :- digit(3,2,3), digit(3,3,3), not bad.
bad :- digit(3,2,3), digit(3,4,3), not bad.
bad :- digit(3,3,3), digit(3,4,3), not bad.
bad :- digit(3,1,4), digit(3,2,4), not bad.
bad :- digit(3,1,4), digit(3,3,4), not bad.
bad :- digit(3,1,4), digit(3,4,4), not bad.
bad :- digit(3,2,4), digit(3,3,4), not bad.
bad :- digit(3,2,4), digit(3,4,4), not bad.
bad :- digit(3,3,4), digit(3,4,4), not bad.
bad :- digit(4,1,1), digit(4,2,1), not bad.
bad :- digit(4,1,1), digit(4,3,1), not bad.
bad :- digit(4,1,1), digit(4,4,1), not bad.
bad :- digit(4,2,1), digit(4,3,1), not bad.
bad :- digit(4,2,1), digit(4,4,1), not bad.
bad :- digit(4,3,1), digit(4,4,1), not bad.
bad :- digit(4,1,2), digit(4,2,2), not bad.
bad :- digit(4,1,2), digit(4,3,2), not bad.
bad :- digit(4,1,2), digit(4,4,2), not bad.
bad :- digit(4,2,2), digit(4,3,2), not bad.
bad :- digit(4,2,2), digit(4,4,2), not bad.
bad :- digit(4,3,2), digit(4,4,2), not bad.
bad :- digit(4,1,3), digit(4,2,3), not bad.
bad :- digit(4,1,3), digit(4,3,3), not bad.
bad :- digit(4,1,3), digit(4,4,3), not bad.
bad :- digit(4,2,3), digit(4,3,3), not bad.
bad :- digit(4,2,3), digit(4,4,3), not bad.
bad :- digit(4,3,3), digit(4,4,3), not bad.
bad :- digit(4,1,4), digit(4,2,4), not bad.
bad :- digit(4,1,4), digit(4,3,4), not bad.
bad :- digit(4,1,4), digit(4,4,4), not bad.
bad :- digit(4,2,4), digit(4,3,4), not bad.
bad :- digit(4,2,4), digit(4,4,4), not bad.
bad :- digit(4,3,4), digit(4,4,4), not bad.

bad :- digit(1,1,1), digit(2,1,1), not bad.
bad :- digit(1,1,1), digit(3,1,1), not bad.
bad :- digit(1,1,1), digit(4,1,1), not bad.
bad :- digit(2,1,1), digit(3,1,1), not bad.
bad :- digit(2,1,1), digit(4,1,1), not bad.
bad :- digit(3,1,1), digit(4,1,1), not bad.
bad :- digit(1,1,2), digit(2,1,2), not bad.
bad :- digit(1,1,2), digit(3,1,2), not bad.
bad :- digit(1,1,2), digit(4,1,2), not bad.
bad :- digit(2,1,2), digit(3,1,2), not bad.
bad :- digit(2,1,2), digit(4,1,2), not bad.
bad :- digit(3,1,2), digit(4,1,2), not bad.
bad :- digit(1,1,3), digit(2,1,3), not bad.
bad :- digit(1,1,3), digit(3,1,3), not bad.
bad :- digit(1,1,3), digit(4,1,3), not bad.
bad :- digit(2,1,3), digit(3,1,3), not bad.
bad :- digit(2,1,3), digit(4,1,3), not bad.
bad :- digit(3,1,3), digit(4,1,3), not bad.
bad :- digit(1,1,4), digit(2,1,4), not bad.
bad :- digit(1,1,4), digit(3,1,4), not bad.
bad :- digit(1,1,4), digit(4,1,4), not bad.
bad :- digit(2,1,4), digit(3,1,4), not bad.
bad :- digit(2,1,4), digit(4,1,4), not bad.
bad :- digit(3,1,4), digit(4,1,4), not bad.
bad :- digit(1,2,1), digit(2,2,1), not bad.
bad :- digit(1,2,1), digit(3,2,1), not bad.
bad :- digit(1,2,1), digit(4,2,1), not bad.
bad :- digit(2,2,1), digit(3,2,1), not bad.
bad :- digit(2,2,1), digit(4,2,1), not bad.
bad :- digit(3,2,1), digit(4,2,1), not bad.
bad :- digit(1,2,2), digit(2,2,2), not bad.
bad :- digit(1,2,2), digit(3,2,2), not bad.
bad :- digit(1,2,2), digit(4,2,2), not bad.
bad :- digit(2,2,2), digit(3,2,2), not bad.
bad :- digit(2,2,2), digit(4,2,2), not bad.
bad :- digit(3,2,2), digit(4,2,2), not bad.
bad :- digit(1,2,3), digit(2,2,3), not bad.
bad :- digit(1,2,3), digit(3,2,3), not bad.
bad :- digit(1,2,3), digit(4,2,3), not bad.
bad :- digit(2,2,3), digit(3,2,3), not bad.
bad :- digit(2,2,3), digit(4,2,3), not bad.
bad :- digit(3,2,3), digit(4,2,3), not bad.
bad :- digit(1,2,4), digit(2,2,4), not bad.
bad :- digit(1,2,4), digit(3,2,4), not bad.
bad :- digit(1,2,4), digit(4,2,4), not bad.
bad :- digit(2,2,4), digit(3,2,4), not bad.
bad :- digit(2,2,4), digit(4,2,4), not bad.
bad :- digit(3,2,4), digit(4,2,4), not bad.
bad :- digit(1,3,1), digit(2,3,1), not bad.
bad :- digit(1,3,1), digit(3,3,1), not bad.
bad :- digit(1,3,1), digit(4,3,1), not bad.
bad :- digit(2,3,1), digit(3,3,1), not bad.
bad :- digit(2,3,1), digit(4,3,1), not bad.
bad :- digit(3,3,1), digit(4,3,1), not bad.
bad :- digit(1,3,2), digit(2,3,2), not bad.
bad :- digit(1,3,2), digit(3,3,2), not bad.
bad :- digit(1,3,2), digit(4,3,2), not bad.
bad :- digit(2,3,2), digit(3,3,2), not bad.
bad :- digit(2,3,2), digit(4,3,2), not bad.
bad :- digit(3,3,2), digit(4,3,2), not bad.
bad :- digit(1,3,3), digit(2,3,3), not bad.
bad :- digit(1,3,3), digit(3,3,3), not bad.
bad :- digit(1,3,3), digit(4,3,3), not bad.
bad :- digit(2,3,3), digit(3,3,3), not bad.
bad :- digit(2,3,3), digit(4,3,3), not bad.
bad :- digit(3,3,3), digit(4,3,3), not bad.
bad :- digit(1,3,4), digit(2,3,4), not bad.
bad :- digit(1,3,4), digit(3,3,4), not bad.
bad :- digit(1,3,4), digit(4,3,4), not bad.
bad :- digit(2,3,4), digit(3,3,4), not bad.
bad :- digit(2,3,4), digit(4,3,4), not bad.
bad :- digit(3,3,4), digit(4,3,4), not bad.
bad :- digit(1,4,1), digit(2,4,1), not bad.
bad :- digit(1,4,1), digit(3,4,1), not bad.
bad :- digit(1,4,1), digit(4,4,1), not bad.
bad :- digit(2,4,1), digit(3,4,1), not bad.
bad :- digit(2,4,1), digit(4,4,1), not bad.
bad :- digit(3,4,1), digit(4,4,1), not bad.
bad :- digit(1,4,2), digit(2,4,2), not bad.
bad :- digit(1,4,2), digit(3,4,2), not bad.
bad :- digit(1,4,2), digit(4,4,2), not bad.
bad :- digit(2,4,2), digit(3,4,2), not bad.
bad :- digit(2,4,2), digit(4,4,2), not bad.
bad :- digit(3,4,2), digit(4,4,2), not bad.
bad :- digit(1,4,3), digit(2,4,3), not bad.
bad :- digit(1,4,3), digit(3,4,3), not bad.
bad :- digit(1,4,3), digit(4,4,3), not bad.
bad :- digit(2,4,3), digit(3,4,3), not bad.
bad :- digit(2,4,3), digit(4,4,3), not bad.
bad :- digit(3,4,3), digit(4,4,3), not bad.
bad :- digit(1,4,4), digit(2,4,4), not bad.
bad :- digit(1,4,4), digit(3,4,4), not bad.
bad :- digit(1,4,4), digit(4,4,4), not bad.
bad :- digit(2,4,4), digit(3,4,4), not bad.
bad :- digit(2,4,4), digit(4,4,4), not bad.
bad :- digit(3,4,4), digit(4,4,4), not bad.

bad :- digit(1,1,1), digit(1,2,1), not bad.
bad :- digit(1,1,1), digit(2,1,1), not bad.
bad :- digit(1,1,1), digit(2,2,1), not bad.
bad :- digit(1,2,1), digit(2,1,1), not bad.
bad :- digit(1,2,1), digit(2,2,1), not bad.
bad :- digit(2,1,1), digit(2,2,1), not bad.
bad :- digit(1,1,2), digit(1,2,2), not bad.
bad :- digit(1,1,2), digit(2,1,2), not bad.
bad :- digit(1,1,2), digit(2,2,2), not bad.
bad :- digit(1,2,2), digit(2,1,2), not bad.
bad :- digit(1,2,2), digit(2,2,2), not bad.
bad :- digit(2,1,2), digit(2,2,2), not bad.
bad :- digit(1,1,3), digit(1,2,3), not bad.
bad :- digit(1,1,3), digit(2,1,3), not bad.
bad :- digit(1,1,3), digit(2,2,3), not bad.
bad :- digit(1,2,3), digit(2,1,3), not bad.
bad :- digit(1,2,3), digit(2,2,3), not bad.
bad :- digit(2,1,3), digit(2,2,3), not bad.
bad :- digit(1,1,4), digit(1,2,4), not bad.
bad :- digit(1,1,4), digit(2,1,4), not bad.
bad :- digit(1,1,4), digit(2,2,4), not bad.
bad :- digit(1,2,4), digit(2,1,4), not bad.
bad :- digit(1,2,4), digit(2,2,4), not bad.
bad :- digit(2,1,4), digit(2,2,4), not bad.
bad :- digit(1,3,1), digit(1,4,1), not bad.
bad :- digit(1,3,1), digit(2,3,1), not bad.
bad :- digit(1,3,1), digit(2,4,1), not bad.
bad :- digit(1,4,1), digit(2,3,1), not bad.
bad :- digit(1,4,1), digit(2,4,1), not bad.
bad :- digit(2,3,1), digit(2,4,1), not bad.
bad :- digit(1,3,2), digit(1,4,2), not bad.
bad :- digit(1,3,2), digit(2,3,2), not bad.
bad :- digit(1,3,2), digit(2,4,2), not bad.
bad :- digit(1,4,2), digit(2,3,2), not bad.
bad :- digit(1,4,2), digit(2,4,2), not bad.
bad :- digit(2,3,2), digit(2,4,2), not bad.
bad :- digit(1,3,3), digit(1,4,3), not bad.
bad :- digit(1,3,3), digit(2,3,3), not bad.
bad :- digit(1,3,3), digit(2,4,3), not bad.
bad :- digit(1,4,3), digit(2,3,3), not bad.
bad :- digit(1,4,3), digit(2,4,3), not bad.
bad :- digit(2,3,3), digit(2,4,3), not bad.
bad :- digit(1,3,4), digit(1,4,4), not bad.
bad :- digit(1,3,4), digit(2,3,4), not bad.
bad :- digit(1,3,4), digit(2,4,4), not bad.
bad :- digit(1,4,4), digit(2,3,4), not bad.
bad :- digit(1,4,4), digit(2,4,4), not bad.
bad :- digit(2,3,4), digit(2,4,4), not bad.
bad :- digit(3,1,1), digit(3,2,1), not bad.
bad :- digit(3,1,1), digit(4,1,1), not bad.
bad :- digit(3,1,1), digit(4,2,1), not bad.
bad :- digit(3,2,1), digit(4,1,1), not bad.
bad :- digit(3,2,1), digit(4,2,1), not bad.
bad :- digit(4,1,1), digit(4,2,1), not bad.
bad :- digit(3,1,2), digit(3,2,2), not bad.
bad :- digit(3,1,2), digit(4,1,2), not bad.
bad :- digit(3,1,2), digit(4,2,2), not bad.
bad :- digit(3,2,2), digit(4,1,2), not bad.
bad :- digit(3,2,2), digit(4,2,2), not bad.
bad :- digit(4,1,2), digit(4,2,2), not bad.
bad :- digit(3,1,3), digit(3,2,3), not bad.
bad :- digit(3,1,3), digit(4,1,3), not bad.
bad :- digit(3,1,3), digit(4,2,3), not bad.
bad :- digit(3,2,3), digit(4,1,3), not bad.
bad :- digit(3,2,3), digit(4,2,3), not bad.
bad :- digit(4,1,3), digit(4,2,3), not bad.
bad :- digit(3,1,4), digit(3,2,4), not bad.
bad :- digit(3,1,4), digit(4,1,4), not bad.
bad :- digit(3,1,4), digit(4,2,4), not bad.
bad :- digit(3,2,4), digit(4,1,4), not bad.
bad :- digit(3,2,4), digit(4,2,4), not bad.
bad :- digit(4,1,4), digit(4,2,4), not bad.
bad :- digit(3,3,1), digit(3,4,1), not bad.
bad :- digit(3,3,1), digit(4,3,1), not bad.
bad :- digit(3,3,1), digit(4,4,1), not bad.
bad :- digit(3,4,1), digit(4,3,1), not bad.
bad :- digit(3,4,1), digit(4,4,1), not bad.
bad :- digit(4,3,1), digit(4,4,1), not bad.
bad :- digit(3,3,2), digit(3,4,2), not bad.
bad :- digit(3,3,2), digit(4,3,2), not bad.
bad :- digit(3,3,2), digit(4,4,2), not bad.
bad :- digit(3,4,2), digit(4,3,2), not bad.
bad :- digit(3,4,2), digit(4,4,2), not bad.
bad :- digit(4,3,2), digit(4,4,2), not bad.
bad :- digit(3,3,3), digit(3,4,3), not bad.
bad :- digit(3,3,3), digit(4,3,3), not bad.
bad :- digit(3,3,3), digit(4,4,3), not bad.
bad :- digit(3,4,3), digit(4,3,3), not bad.
bad :- digit(3,4,3), digit(4,4,3), not bad.
bad :- digit(4,3,3), digit(4,4,3), not bad.
bad :- digit(3,3,4), digit(3,4,4), not bad.
bad :- digit(3,3,4), digit(4,3,4), not bad.
bad :- digit(3,3,4), digit(4,4,4), not bad.
bad :- digit(3,4,4), digit(4,3,4), not bad.
bad :- digit(3,4,4), digit(4,4,4), not bad.
bad :- digit(4,3,4), digit(4,4,4), not bad.
