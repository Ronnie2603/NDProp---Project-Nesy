%------------------------------------------------
% For each position (0–3), choose exactly one digit
%------------------------------------------------
1 { digit(0,V) : V=0..9 } 1.
1 { digit(1,V) : V=0..9 } 1.


% Deterministic sum without carrying across positions (matches NeurASP mnistAdd2).
sum(S) :- digit(0, A), digit(1, B), S = A + B, S = 0..18.
