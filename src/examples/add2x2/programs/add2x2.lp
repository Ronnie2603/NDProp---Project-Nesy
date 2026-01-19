%===============================
% 2x2 DIGIT GRID — NORMAL ONLY
% (no choice rules, only default negation)
%===============================

% ---- Exactly-one digit per cell (I = 0..3) via exclusive defaults ----
% digit(I,D) holds iff all other digit(I,D2) (D2 != D) are false
digit(I,D) :- I = 0..3, D = 0..9,
              not digit(I,D2) : D2 = 0..9, D2 != D.

% ---- Deterministic sums (no carry) ----
row(0,S) :- digit(0,A), digit(1,B), S = A + B, S = 0..18.
row(1,S) :- digit(2,A), digit(3,B), S = A + B, S = 0..18.

col(0,S) :- digit(0,A), digit(2,B), S = A + B, S = 0..18.
col(1,S) :- digit(1,A), digit(3,B), S = A + B, S = 0..18.