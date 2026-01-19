%===========================================
% MNIST ADDITION (2-digit numbers) — NORMAL
%===========================================

% Four positions: 0 and 1 for the first number (tens, ones),
% 2 and 3 for the second number.
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

digit(3, 0) :- not digit(3, 1), not digit(3, 2), not digit(3, 3), not digit(3, 4), not digit(3, 5), not digit(3, 6), not digit(3, 7), not digit(3, 8), not digit(3, 9).
digit(3, 1) :- not digit(3, 0), not digit(3, 2), not digit(3, 3), not digit(3, 4), not digit(3, 5), not digit(3, 6), not digit(3, 7), not digit(3, 8), not digit(3, 9).
digit(3, 2) :- not digit(3, 0), not digit(3, 1), not digit(3, 3), not digit(3, 4), not digit(3, 5), not digit(3, 6), not digit(3, 7), not digit(3, 8), not digit(3, 9).
digit(3, 3) :- not digit(3, 0), not digit(3, 1), not digit(3, 2), not digit(3, 4), not digit(3, 5), not digit(3, 6), not digit(3, 7), not digit(3, 8), not digit(3, 9).
digit(3, 4) :- not digit(3, 0), not digit(3, 1), not digit(3, 2), not digit(3, 3), not digit(3, 5), not digit(3, 6), not digit(3, 7), not digit(3, 8), not digit(3, 9).
digit(3, 5) :- not digit(3, 0), not digit(3, 1), not digit(3, 2), not digit(3, 3), not digit(3, 4), not digit(3, 6), not digit(3, 7), not digit(3, 8), not digit(3, 9).
digit(3, 6) :- not digit(3, 0), not digit(3, 1), not digit(3, 2), not digit(3, 3), not digit(3, 4), not digit(3, 5), not digit(3, 7), not digit(3, 8), not digit(3, 9).
digit(3, 7) :- not digit(3, 0), not digit(3, 1), not digit(3, 2), not digit(3, 3), not digit(3, 4), not digit(3, 5), not digit(3, 6), not digit(3, 8), not digit(3, 9).
digit(3, 8) :- not digit(3, 0), not digit(3, 1), not digit(3, 2), not digit(3, 3), not digit(3, 4), not digit(3, 5), not digit(3, 6), not digit(3, 7), not digit(3, 9).
digit(3, 9) :- not digit(3, 0), not digit(3, 1), not digit(3, 2), not digit(3, 3), not digit(3, 4), not digit(3, 5), not digit(3, 6), not digit(3, 7), not digit(3, 8).

% Deterministic sum without carrying across positions (matches NeurASP mnistAdd2).
sum(S) :- digit(0, A), digit(1, B), digit(2, C), digit(3, D), S = (A + C) * 10 + B + D, S = 0..198.
