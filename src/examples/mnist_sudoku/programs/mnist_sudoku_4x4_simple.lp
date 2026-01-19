% MNIST Sudoku rules (normal rules only) for 4x4

digit(1,1,1) :- not digit(1,1,2), not digit(1,1,3), not digit(1,1,4), not digit(1,2,1), not digit(1,3,1), not digit(1,4,1), not digit(2,1,1), not digit(3,1,1), not digit(4,1,1), not digit(1,2,1), not digit(2,1,1), not digit(2,2,1).
digit(1,1,2) :- not digit(1,1,1), not digit(1,1,3), not digit(1,1,4), not digit(1,2,2), not digit(1,3,2), not digit(1,4,2), not digit(2,1,2), not digit(3,1,2), not digit(4,1,2), not digit(1,2,2), not digit(2,1,2), not digit(2,2,2).
digit(1,1,3) :- not digit(1,1,1), not digit(1,1,2), not digit(1,1,4), not digit(1,2,3), not digit(1,3,3), not digit(1,4,3), not digit(2,1,3), not digit(3,1,3), not digit(4,1,3), not digit(1,2,3), not digit(2,1,3), not digit(2,2,3).
digit(1,1,4) :- not digit(1,1,1), not digit(1,1,2), not digit(1,1,3), not digit(1,2,4), not digit(1,3,4), not digit(1,4,4), not digit(2,1,4), not digit(3,1,4), not digit(4,1,4), not digit(1,2,4), not digit(2,1,4), not digit(2,2,4).
bad_0 :- not digit(1,1,1), not digit(1,1,2), not digit(1,1,3), not digit(1,1,4), not bad_0.
digit(1,2,1) :- not digit(1,2,2), not digit(1,2,3), not digit(1,2,4), not digit(1,1,1), not digit(1,3,1), not digit(1,4,1), not digit(2,2,1), not digit(3,2,1), not digit(4,2,1), not digit(1,1,1), not digit(2,1,1), not digit(2,2,1).
digit(1,2,2) :- not digit(1,2,1), not digit(1,2,3), not digit(1,2,4), not digit(1,1,2), not digit(1,3,2), not digit(1,4,2), not digit(2,2,2), not digit(3,2,2), not digit(4,2,2), not digit(1,1,2), not digit(2,1,2), not digit(2,2,2).
digit(1,2,3) :- not digit(1,2,1), not digit(1,2,2), not digit(1,2,4), not digit(1,1,3), not digit(1,3,3), not digit(1,4,3), not digit(2,2,3), not digit(3,2,3), not digit(4,2,3), not digit(1,1,3), not digit(2,1,3), not digit(2,2,3).
digit(1,2,4) :- not digit(1,2,1), not digit(1,2,2), not digit(1,2,3), not digit(1,1,4), not digit(1,3,4), not digit(1,4,4), not digit(2,2,4), not digit(3,2,4), not digit(4,2,4), not digit(1,1,4), not digit(2,1,4), not digit(2,2,4).
bad_1 :- not digit(1,2,1), not digit(1,2,2), not digit(1,2,3), not digit(1,2,4), not bad_1.
digit(1,3,1) :- not digit(1,3,2), not digit(1,3,3), not digit(1,3,4), not digit(1,1,1), not digit(1,2,1), not digit(1,4,1), not digit(2,3,1), not digit(3,3,1), not digit(4,3,1), not digit(1,4,1), not digit(2,3,1), not digit(2,4,1).
digit(1,3,2) :- not digit(1,3,1), not digit(1,3,3), not digit(1,3,4), not digit(1,1,2), not digit(1,2,2), not digit(1,4,2), not digit(2,3,2), not digit(3,3,2), not digit(4,3,2), not digit(1,4,2), not digit(2,3,2), not digit(2,4,2).
digit(1,3,3) :- not digit(1,3,1), not digit(1,3,2), not digit(1,3,4), not digit(1,1,3), not digit(1,2,3), not digit(1,4,3), not digit(2,3,3), not digit(3,3,3), not digit(4,3,3), not digit(1,4,3), not digit(2,3,3), not digit(2,4,3).
digit(1,3,4) :- not digit(1,3,1), not digit(1,3,2), not digit(1,3,3), not digit(1,1,4), not digit(1,2,4), not digit(1,4,4), not digit(2,3,4), not digit(3,3,4), not digit(4,3,4), not digit(1,4,4), not digit(2,3,4), not digit(2,4,4).
bad_2 :- not digit(1,3,1), not digit(1,3,2), not digit(1,3,3), not digit(1,3,4), not bad_2.
digit(1,4,1) :- not digit(1,4,2), not digit(1,4,3), not digit(1,4,4), not digit(1,1,1), not digit(1,2,1), not digit(1,3,1), not digit(2,4,1), not digit(3,4,1), not digit(4,4,1), not digit(1,3,1), not digit(2,3,1), not digit(2,4,1).
digit(1,4,2) :- not digit(1,4,1), not digit(1,4,3), not digit(1,4,4), not digit(1,1,2), not digit(1,2,2), not digit(1,3,2), not digit(2,4,2), not digit(3,4,2), not digit(4,4,2), not digit(1,3,2), not digit(2,3,2), not digit(2,4,2).
digit(1,4,3) :- not digit(1,4,1), not digit(1,4,2), not digit(1,4,4), not digit(1,1,3), not digit(1,2,3), not digit(1,3,3), not digit(2,4,3), not digit(3,4,3), not digit(4,4,3), not digit(1,3,3), not digit(2,3,3), not digit(2,4,3).
digit(1,4,4) :- not digit(1,4,1), not digit(1,4,2), not digit(1,4,3), not digit(1,1,4), not digit(1,2,4), not digit(1,3,4), not digit(2,4,4), not digit(3,4,4), not digit(4,4,4), not digit(1,3,4), not digit(2,3,4), not digit(2,4,4).
bad_3 :- not digit(1,4,1), not digit(1,4,2), not digit(1,4,3), not digit(1,4,4), not bad_3.
digit(2,1,1) :- not digit(2,1,2), not digit(2,1,3), not digit(2,1,4), not digit(2,2,1), not digit(2,3,1), not digit(2,4,1), not digit(1,1,1), not digit(3,1,1), not digit(4,1,1), not digit(1,1,1), not digit(1,2,1), not digit(2,2,1).
digit(2,1,2) :- not digit(2,1,1), not digit(2,1,3), not digit(2,1,4), not digit(2,2,2), not digit(2,3,2), not digit(2,4,2), not digit(1,1,2), not digit(3,1,2), not digit(4,1,2), not digit(1,1,2), not digit(1,2,2), not digit(2,2,2).
digit(2,1,3) :- not digit(2,1,1), not digit(2,1,2), not digit(2,1,4), not digit(2,2,3), not digit(2,3,3), not digit(2,4,3), not digit(1,1,3), not digit(3,1,3), not digit(4,1,3), not digit(1,1,3), not digit(1,2,3), not digit(2,2,3).
digit(2,1,4) :- not digit(2,1,1), not digit(2,1,2), not digit(2,1,3), not digit(2,2,4), not digit(2,3,4), not digit(2,4,4), not digit(1,1,4), not digit(3,1,4), not digit(4,1,4), not digit(1,1,4), not digit(1,2,4), not digit(2,2,4).
bad_4 :- not digit(2,1,1), not digit(2,1,2), not digit(2,1,3), not digit(2,1,4), not bad_4.
digit(2,2,1) :- not digit(2,2,2), not digit(2,2,3), not digit(2,2,4), not digit(2,1,1), not digit(2,3,1), not digit(2,4,1), not digit(1,2,1), not digit(3,2,1), not digit(4,2,1), not digit(1,1,1), not digit(1,2,1), not digit(2,1,1).
digit(2,2,2) :- not digit(2,2,1), not digit(2,2,3), not digit(2,2,4), not digit(2,1,2), not digit(2,3,2), not digit(2,4,2), not digit(1,2,2), not digit(3,2,2), not digit(4,2,2), not digit(1,1,2), not digit(1,2,2), not digit(2,1,2).
digit(2,2,3) :- not digit(2,2,1), not digit(2,2,2), not digit(2,2,4), not digit(2,1,3), not digit(2,3,3), not digit(2,4,3), not digit(1,2,3), not digit(3,2,3), not digit(4,2,3), not digit(1,1,3), not digit(1,2,3), not digit(2,1,3).
digit(2,2,4) :- not digit(2,2,1), not digit(2,2,2), not digit(2,2,3), not digit(2,1,4), not digit(2,3,4), not digit(2,4,4), not digit(1,2,4), not digit(3,2,4), not digit(4,2,4), not digit(1,1,4), not digit(1,2,4), not digit(2,1,4).
bad_5 :- not digit(2,2,1), not digit(2,2,2), not digit(2,2,3), not digit(2,2,4), not bad_5.
digit(2,3,1) :- not digit(2,3,2), not digit(2,3,3), not digit(2,3,4), not digit(2,1,1), not digit(2,2,1), not digit(2,4,1), not digit(1,3,1), not digit(3,3,1), not digit(4,3,1), not digit(1,3,1), not digit(1,4,1), not digit(2,4,1).
digit(2,3,2) :- not digit(2,3,1), not digit(2,3,3), not digit(2,3,4), not digit(2,1,2), not digit(2,2,2), not digit(2,4,2), not digit(1,3,2), not digit(3,3,2), not digit(4,3,2), not digit(1,3,2), not digit(1,4,2), not digit(2,4,2).
digit(2,3,3) :- not digit(2,3,1), not digit(2,3,2), not digit(2,3,4), not digit(2,1,3), not digit(2,2,3), not digit(2,4,3), not digit(1,3,3), not digit(3,3,3), not digit(4,3,3), not digit(1,3,3), not digit(1,4,3), not digit(2,4,3).
digit(2,3,4) :- not digit(2,3,1), not digit(2,3,2), not digit(2,3,3), not digit(2,1,4), not digit(2,2,4), not digit(2,4,4), not digit(1,3,4), not digit(3,3,4), not digit(4,3,4), not digit(1,3,4), not digit(1,4,4), not digit(2,4,4).
bad_6 :- not digit(2,3,1), not digit(2,3,2), not digit(2,3,3), not digit(2,3,4), not bad_6.
digit(2,4,1) :- not digit(2,4,2), not digit(2,4,3), not digit(2,4,4), not digit(2,1,1), not digit(2,2,1), not digit(2,3,1), not digit(1,4,1), not digit(3,4,1), not digit(4,4,1), not digit(1,3,1), not digit(1,4,1), not digit(2,3,1).
digit(2,4,2) :- not digit(2,4,1), not digit(2,4,3), not digit(2,4,4), not digit(2,1,2), not digit(2,2,2), not digit(2,3,2), not digit(1,4,2), not digit(3,4,2), not digit(4,4,2), not digit(1,3,2), not digit(1,4,2), not digit(2,3,2).
digit(2,4,3) :- not digit(2,4,1), not digit(2,4,2), not digit(2,4,4), not digit(2,1,3), not digit(2,2,3), not digit(2,3,3), not digit(1,4,3), not digit(3,4,3), not digit(4,4,3), not digit(1,3,3), not digit(1,4,3), not digit(2,3,3).
digit(2,4,4) :- not digit(2,4,1), not digit(2,4,2), not digit(2,4,3), not digit(2,1,4), not digit(2,2,4), not digit(2,3,4), not digit(1,4,4), not digit(3,4,4), not digit(4,4,4), not digit(1,3,4), not digit(1,4,4), not digit(2,3,4).
bad_7 :- not digit(2,4,1), not digit(2,4,2), not digit(2,4,3), not digit(2,4,4), not bad_7.
digit(3,1,1) :- not digit(3,1,2), not digit(3,1,3), not digit(3,1,4), not digit(3,2,1), not digit(3,3,1), not digit(3,4,1), not digit(1,1,1), not digit(2,1,1), not digit(4,1,1), not digit(3,2,1), not digit(4,1,1), not digit(4,2,1).
digit(3,1,2) :- not digit(3,1,1), not digit(3,1,3), not digit(3,1,4), not digit(3,2,2), not digit(3,3,2), not digit(3,4,2), not digit(1,1,2), not digit(2,1,2), not digit(4,1,2), not digit(3,2,2), not digit(4,1,2), not digit(4,2,2).
digit(3,1,3) :- not digit(3,1,1), not digit(3,1,2), not digit(3,1,4), not digit(3,2,3), not digit(3,3,3), not digit(3,4,3), not digit(1,1,3), not digit(2,1,3), not digit(4,1,3), not digit(3,2,3), not digit(4,1,3), not digit(4,2,3).
digit(3,1,4) :- not digit(3,1,1), not digit(3,1,2), not digit(3,1,3), not digit(3,2,4), not digit(3,3,4), not digit(3,4,4), not digit(1,1,4), not digit(2,1,4), not digit(4,1,4), not digit(3,2,4), not digit(4,1,4), not digit(4,2,4).
bad_8 :- not digit(3,1,1), not digit(3,1,2), not digit(3,1,3), not digit(3,1,4), not bad_8.
digit(3,2,1) :- not digit(3,2,2), not digit(3,2,3), not digit(3,2,4), not digit(3,1,1), not digit(3,3,1), not digit(3,4,1), not digit(1,2,1), not digit(2,2,1), not digit(4,2,1), not digit(3,1,1), not digit(4,1,1), not digit(4,2,1).
digit(3,2,2) :- not digit(3,2,1), not digit(3,2,3), not digit(3,2,4), not digit(3,1,2), not digit(3,3,2), not digit(3,4,2), not digit(1,2,2), not digit(2,2,2), not digit(4,2,2), not digit(3,1,2), not digit(4,1,2), not digit(4,2,2).
digit(3,2,3) :- not digit(3,2,1), not digit(3,2,2), not digit(3,2,4), not digit(3,1,3), not digit(3,3,3), not digit(3,4,3), not digit(1,2,3), not digit(2,2,3), not digit(4,2,3), not digit(3,1,3), not digit(4,1,3), not digit(4,2,3).
digit(3,2,4) :- not digit(3,2,1), not digit(3,2,2), not digit(3,2,3), not digit(3,1,4), not digit(3,3,4), not digit(3,4,4), not digit(1,2,4), not digit(2,2,4), not digit(4,2,4), not digit(3,1,4), not digit(4,1,4), not digit(4,2,4).
bad_9 :- not digit(3,2,1), not digit(3,2,2), not digit(3,2,3), not digit(3,2,4), not bad_9.
digit(3,3,1) :- not digit(3,3,2), not digit(3,3,3), not digit(3,3,4), not digit(3,1,1), not digit(3,2,1), not digit(3,4,1), not digit(1,3,1), not digit(2,3,1), not digit(4,3,1), not digit(3,4,1), not digit(4,3,1), not digit(4,4,1).
digit(3,3,2) :- not digit(3,3,1), not digit(3,3,3), not digit(3,3,4), not digit(3,1,2), not digit(3,2,2), not digit(3,4,2), not digit(1,3,2), not digit(2,3,2), not digit(4,3,2), not digit(3,4,2), not digit(4,3,2), not digit(4,4,2).
digit(3,3,3) :- not digit(3,3,1), not digit(3,3,2), not digit(3,3,4), not digit(3,1,3), not digit(3,2,3), not digit(3,4,3), not digit(1,3,3), not digit(2,3,3), not digit(4,3,3), not digit(3,4,3), not digit(4,3,3), not digit(4,4,3).
digit(3,3,4) :- not digit(3,3,1), not digit(3,3,2), not digit(3,3,3), not digit(3,1,4), not digit(3,2,4), not digit(3,4,4), not digit(1,3,4), not digit(2,3,4), not digit(4,3,4), not digit(3,4,4), not digit(4,3,4), not digit(4,4,4).
bad_10 :- not digit(3,3,1), not digit(3,3,2), not digit(3,3,3), not digit(3,3,4), not bad_10.
digit(3,4,1) :- not digit(3,4,2), not digit(3,4,3), not digit(3,4,4), not digit(3,1,1), not digit(3,2,1), not digit(3,3,1), not digit(1,4,1), not digit(2,4,1), not digit(4,4,1), not digit(3,3,1), not digit(4,3,1), not digit(4,4,1).
digit(3,4,2) :- not digit(3,4,1), not digit(3,4,3), not digit(3,4,4), not digit(3,1,2), not digit(3,2,2), not digit(3,3,2), not digit(1,4,2), not digit(2,4,2), not digit(4,4,2), not digit(3,3,2), not digit(4,3,2), not digit(4,4,2).
digit(3,4,3) :- not digit(3,4,1), not digit(3,4,2), not digit(3,4,4), not digit(3,1,3), not digit(3,2,3), not digit(3,3,3), not digit(1,4,3), not digit(2,4,3), not digit(4,4,3), not digit(3,3,3), not digit(4,3,3), not digit(4,4,3).
digit(3,4,4) :- not digit(3,4,1), not digit(3,4,2), not digit(3,4,3), not digit(3,1,4), not digit(3,2,4), not digit(3,3,4), not digit(1,4,4), not digit(2,4,4), not digit(4,4,4), not digit(3,3,4), not digit(4,3,4), not digit(4,4,4).
bad_11 :- not digit(3,4,1), not digit(3,4,2), not digit(3,4,3), not digit(3,4,4), not bad_11.
digit(4,1,1) :- not digit(4,1,2), not digit(4,1,3), not digit(4,1,4), not digit(4,2,1), not digit(4,3,1), not digit(4,4,1), not digit(1,1,1), not digit(2,1,1), not digit(3,1,1), not digit(3,1,1), not digit(3,2,1), not digit(4,2,1).
digit(4,1,2) :- not digit(4,1,1), not digit(4,1,3), not digit(4,1,4), not digit(4,2,2), not digit(4,3,2), not digit(4,4,2), not digit(1,1,2), not digit(2,1,2), not digit(3,1,2), not digit(3,1,2), not digit(3,2,2), not digit(4,2,2).
digit(4,1,3) :- not digit(4,1,1), not digit(4,1,2), not digit(4,1,4), not digit(4,2,3), not digit(4,3,3), not digit(4,4,3), not digit(1,1,3), not digit(2,1,3), not digit(3,1,3), not digit(3,1,3), not digit(3,2,3), not digit(4,2,3).
digit(4,1,4) :- not digit(4,1,1), not digit(4,1,2), not digit(4,1,3), not digit(4,2,4), not digit(4,3,4), not digit(4,4,4), not digit(1,1,4), not digit(2,1,4), not digit(3,1,4), not digit(3,1,4), not digit(3,2,4), not digit(4,2,4).
bad_12 :- not digit(4,1,1), not digit(4,1,2), not digit(4,1,3), not digit(4,1,4), not bad_12.
digit(4,2,1) :- not digit(4,2,2), not digit(4,2,3), not digit(4,2,4), not digit(4,1,1), not digit(4,3,1), not digit(4,4,1), not digit(1,2,1), not digit(2,2,1), not digit(3,2,1), not digit(3,1,1), not digit(3,2,1), not digit(4,1,1).
digit(4,2,2) :- not digit(4,2,1), not digit(4,2,3), not digit(4,2,4), not digit(4,1,2), not digit(4,3,2), not digit(4,4,2), not digit(1,2,2), not digit(2,2,2), not digit(3,2,2), not digit(3,1,2), not digit(3,2,2), not digit(4,1,2).
digit(4,2,3) :- not digit(4,2,1), not digit(4,2,2), not digit(4,2,4), not digit(4,1,3), not digit(4,3,3), not digit(4,4,3), not digit(1,2,3), not digit(2,2,3), not digit(3,2,3), not digit(3,1,3), not digit(3,2,3), not digit(4,1,3).
digit(4,2,4) :- not digit(4,2,1), not digit(4,2,2), not digit(4,2,3), not digit(4,1,4), not digit(4,3,4), not digit(4,4,4), not digit(1,2,4), not digit(2,2,4), not digit(3,2,4), not digit(3,1,4), not digit(3,2,4), not digit(4,1,4).
bad_13 :- not digit(4,2,1), not digit(4,2,2), not digit(4,2,3), not digit(4,2,4), not bad_13.
digit(4,3,1) :- not digit(4,3,2), not digit(4,3,3), not digit(4,3,4), not digit(4,1,1), not digit(4,2,1), not digit(4,4,1), not digit(1,3,1), not digit(2,3,1), not digit(3,3,1), not digit(3,3,1), not digit(3,4,1), not digit(4,4,1).
digit(4,3,2) :- not digit(4,3,1), not digit(4,3,3), not digit(4,3,4), not digit(4,1,2), not digit(4,2,2), not digit(4,4,2), not digit(1,3,2), not digit(2,3,2), not digit(3,3,2), not digit(3,3,2), not digit(3,4,2), not digit(4,4,2).
digit(4,3,3) :- not digit(4,3,1), not digit(4,3,2), not digit(4,3,4), not digit(4,1,3), not digit(4,2,3), not digit(4,4,3), not digit(1,3,3), not digit(2,3,3), not digit(3,3,3), not digit(3,3,3), not digit(3,4,3), not digit(4,4,3).
digit(4,3,4) :- not digit(4,3,1), not digit(4,3,2), not digit(4,3,3), not digit(4,1,4), not digit(4,2,4), not digit(4,4,4), not digit(1,3,4), not digit(2,3,4), not digit(3,3,4), not digit(3,3,4), not digit(3,4,4), not digit(4,4,4).
bad_14 :- not digit(4,3,1), not digit(4,3,2), not digit(4,3,3), not digit(4,3,4), not bad_14.
digit(4,4,1) :- not digit(4,4,2), not digit(4,4,3), not digit(4,4,4), not digit(4,1,1), not digit(4,2,1), not digit(4,3,1), not digit(1,4,1), not digit(2,4,1), not digit(3,4,1), not digit(3,3,1), not digit(3,4,1), not digit(4,3,1).
digit(4,4,2) :- not digit(4,4,1), not digit(4,4,3), not digit(4,4,4), not digit(4,1,2), not digit(4,2,2), not digit(4,3,2), not digit(1,4,2), not digit(2,4,2), not digit(3,4,2), not digit(3,3,2), not digit(3,4,2), not digit(4,3,2).
digit(4,4,3) :- not digit(4,4,1), not digit(4,4,2), not digit(4,4,4), not digit(4,1,3), not digit(4,2,3), not digit(4,3,3), not digit(1,4,3), not digit(2,4,3), not digit(3,4,3), not digit(3,3,3), not digit(3,4,3), not digit(4,3,3).
digit(4,4,4) :- not digit(4,4,1), not digit(4,4,2), not digit(4,4,3), not digit(4,1,4), not digit(4,2,4), not digit(4,3,4), not digit(1,4,4), not digit(2,4,4), not digit(3,4,4), not digit(3,3,4), not digit(3,4,4), not digit(4,3,4).
bad_15 :- not digit(4,4,1), not digit(4,4,2), not digit(4,4,3), not digit(4,4,4), not bad_15.
