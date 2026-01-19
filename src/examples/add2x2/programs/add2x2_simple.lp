%===============================
% 2x2 DIGIT GRID — NORMAL ONLY
% (no choice rules, only default negation)
%===============================

%------------------------------------------------
% For each position (0–3), choose exactly one digit
%------------------------------------------------
1 { digit(0,V) : V=0..9 } 1.
1 { digit(1,V) : V=0..9 } 1.
1 { digit(2,V) : V=0..9 } 1.
1 { digit(3,V) : V=0..9 } 1.



% ---- Deterministic sums (no carry) ----
row(0,S) :- digit(0,A), digit(1,B), S = A + B, S = 0..18.
row(1,S) :- digit(2,A), digit(3,B), S = A + B, S = 0..18.

col(0,S) :- digit(0,A), digit(2,B), S = A + B, S = 0..18.
col(1,S) :- digit(1,A), digit(3,B), S = A + B, S = 0..18.