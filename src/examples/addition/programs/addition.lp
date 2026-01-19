%===========================================
% MNIST ADDITION (single-digit numbers) — NORMAL
%===========================================

% Two positions: 0 for the first digit, 1 for the second digit.
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

% Deterministic sum (single-digit addition).
sum(S) :- digit(0, A), digit(1, B), S = A + B, S = 0..18.
