%===========================================
% MNIST MEMBER3 — membership over 3 digits
%===========================================

digit(0, 0) :- not digit(0, 1), not digit(0, 2), not digit(0, 3), not digit(0, 4), not digit(0, 5), not digit(0, 6), not digit(0, 7), not digit(0, 8), not digit(0, 9).
digit(0, 1) :- not digit(0, 0), not digit(0, 2), not digit(0, 3), not digit(0, 4), not digit(0, 5), not digit(0, 6), not digit(0, 7), not digit(0, 8), not digit(0, 9).
digit(0, 2) :- not digit(0, 0), not digit(0, 1), not digit(0, 3), not digit(0, 4), not digit(0, 5), not digit(0, 6), not digit(0, 7), not digit(0, 8), not digit(0, 9).
digit(0, 3) :- not digit(0, 0), not digit(0, 1), not digit(0, 2), not digit(0, 4), not digit(0, 5), not digit(0, 6), not digit(0, 7), not digit(0, 8), not digit(0, 9).
digit(0, 4) :- not digit(0, 0), not digit(0, 1), not digit(0, 2), not digit(0, 3), not digit(0, 5), not digit(0, 6), not digit(0, 7), not digit(0, 8), not digit(0, 9).
digit(0, 5) :- not digit(0, 0), not digit(0, 1), not digit(0, 2), not digit(0, 3), not digit(0, 4), not digit(0, 6), not digit(0, 7), not digit(0, 8), not digit(0, 9).
digit(0, 6) :- not digit(0, 0), not digit(0, 1), not digit(0, 2), not digit(0, 3), not digit(0, 4), not digit(0, 5), not digit(0, 7), not digit(0, 8), not digit(0, 9).
digit(0, 7) :- not digit(0, 0), not digit(0, 1), not digit(0, 2), not digit(0, 3), not digit(0, 4), not digit(0, 5), not digit(0, 6), not digit(0, 8), not digit(0, 9).
digit(0, 8) :- not digit(0, 0), not digit(0, 1), not digit(0, 2), not digit(0, 3), not digit(0, 4), not digit(0, 5), not digit(0, 6), not digit(0, 7), not digit(0, 9).
digit(0, 9) :- not digit(0, 0), not digit(0, 1), not digit(0, 2), not digit(0, 3), not digit(0, 4), not digit(0, 5), not digit(0, 6), not digit(0, 7), not digit(0, 8).

digit(1, 0) :- not digit(1, 1), not digit(1, 2), not digit(1, 3), not digit(1, 4), not digit(1, 5), not digit(1, 6), not digit(1, 7), not digit(1, 8), not digit(1, 9).
digit(1, 1) :- not digit(1, 0), not digit(1, 2), not digit(1, 3), not digit(1, 4), not digit(1, 5), not digit(1, 6), not digit(1, 7), not digit(1, 8), not digit(1, 9).
digit(1, 2) :- not digit(1, 0), not digit(1, 1), not digit(1, 3), not digit(1, 4), not digit(1, 5), not digit(1, 6), not digit(1, 7), not digit(1, 8), not digit(1, 9).
digit(1, 3) :- not digit(1, 0), not digit(1, 1), not digit(1, 2), not digit(1, 4), not digit(1, 5), not digit(1, 6), not digit(1, 7), not digit(1, 8), not digit(1, 9).
digit(1, 4) :- not digit(1, 0), not digit(1, 1), not digit(1, 2), not digit(1, 3), not digit(1, 5), not digit(1, 6), not digit(1, 7), not digit(1, 8), not digit(1, 9).
digit(1, 5) :- not digit(1, 0), not digit(1, 1), not digit(1, 2), not digit(1, 3), not digit(1, 4), not digit(1, 6), not digit(1, 7), not digit(1, 8), not digit(1, 9).
digit(1, 6) :- not digit(1, 0), not digit(1, 1), not digit(1, 2), not digit(1, 3), not digit(1, 4), not digit(1, 5), not digit(1, 7), not digit(1, 8), not digit(1, 9).
digit(1, 7) :- not digit(1, 0), not digit(1, 1), not digit(1, 2), not digit(1, 3), not digit(1, 4), not digit(1, 5), not digit(1, 6), not digit(1, 8), not digit(1, 9).
digit(1, 8) :- not digit(1, 0), not digit(1, 1), not digit(1, 2), not digit(1, 3), not digit(1, 4), not digit(1, 5), not digit(1, 6), not digit(1, 7), not digit(1, 9).
digit(1, 9) :- not digit(1, 0), not digit(1, 1), not digit(1, 2), not digit(1, 3), not digit(1, 4), not digit(1, 5), not digit(1, 6), not digit(1, 7), not digit(1, 8).

digit(2, 0) :- not digit(2, 1), not digit(2, 2), not digit(2, 3), not digit(2, 4), not digit(2, 5), not digit(2, 6), not digit(2, 7), not digit(2, 8), not digit(2, 9).
digit(2, 1) :- not digit(2, 0), not digit(2, 2), not digit(2, 3), not digit(2, 4), not digit(2, 5), not digit(2, 6), not digit(2, 7), not digit(2, 8), not digit(2, 9).
digit(2, 2) :- not digit(2, 0), not digit(2, 1), not digit(2, 3), not digit(2, 4), not digit(2, 5), not digit(2, 6), not digit(2, 7), not digit(2, 8), not digit(2, 9).
digit(2, 3) :- not digit(2, 0), not digit(2, 1), not digit(2, 2), not digit(2, 4), not digit(2, 5), not digit(2, 6), not digit(2, 7), not digit(2, 8), not digit(2, 9).
digit(2, 4) :- not digit(2, 0), not digit(2, 1), not digit(2, 2), not digit(2, 3), not digit(2, 5), not digit(2, 6), not digit(2, 7), not digit(2, 8), not digit(2, 9).
digit(2, 5) :- not digit(2, 0), not digit(2, 1), not digit(2, 2), not digit(2, 3), not digit(2, 4), not digit(2, 6), not digit(2, 7), not digit(2, 8), not digit(2, 9).
digit(2, 6) :- not digit(2, 0), not digit(2, 1), not digit(2, 2), not digit(2, 3), not digit(2, 4), not digit(2, 5), not digit(2, 7), not digit(2, 8), not digit(2, 9).
digit(2, 7) :- not digit(2, 0), not digit(2, 1), not digit(2, 2), not digit(2, 3), not digit(2, 4), not digit(2, 5), not digit(2, 6), not digit(2, 8), not digit(2, 9).
digit(2, 8) :- not digit(2, 0), not digit(2, 1), not digit(2, 2), not digit(2, 3), not digit(2, 4), not digit(2, 5), not digit(2, 6), not digit(2, 7), not digit(2, 9).
digit(2, 9) :- not digit(2, 0), not digit(2, 1), not digit(2, 2), not digit(2, 3), not digit(2, 4), not digit(2, 5), not digit(2, 6), not digit(2, 7), not digit(2, 8).

check(0) :- not check(1), not check(2), not check(3), not check(4), not check(5), not check(6), not check(7), not check(8), not check(9).
check(1) :- not check(0), not check(2), not check(3), not check(4), not check(5), not check(6), not check(7), not check(8), not check(9).
check(2) :- not check(0), not check(1), not check(3), not check(4), not check(5), not check(6), not check(7), not check(8), not check(9).
check(3) :- not check(0), not check(1), not check(2), not check(4), not check(5), not check(6), not check(7), not check(8), not check(9).
check(4) :- not check(0), not check(1), not check(2), not check(3), not check(5), not check(6), not check(7), not check(8), not check(9).
check(5) :- not check(0), not check(1), not check(2), not check(3), not check(4), not check(6), not check(7), not check(8), not check(9).
check(6) :- not check(0), not check(1), not check(2), not check(3), not check(4), not check(5), not check(7), not check(8), not check(9).
check(7) :- not check(0), not check(1), not check(2), not check(3), not check(4), not check(5), not check(6), not check(8), not check(9).
check(8) :- not check(0), not check(1), not check(2), not check(3), not check(4), not check(5), not check(6), not check(7), not check(9).
check(9) :- not check(0), not check(1), not check(2), not check(3), not check(4), not check(5), not check(6), not check(7), not check(8).

member(D, 0) :- digit(0, N1), digit(1, N2), digit(2, N3), check(D), D != N1, D != N2, D != N3.
member(D, 1) :- check(D), not member(D, 0).
