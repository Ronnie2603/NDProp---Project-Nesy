%===============================
% MNIST ADDITION — NORMAL ONLY
% (no choice, no constraints, no built-ins)
%===============================

% ---- Exactly-one a_digit(D) via explicit defaults ----

a_digit(0) :- not a_digit(1), not a_digit(2), not a_digit(3), not a_digit(4),
              not a_digit(5), not a_digit(6), not a_digit(7), not a_digit(8),
              not a_digit(9).

a_digit(1) :- not a_digit(0), not a_digit(2), not a_digit(3), not a_digit(4),
              not a_digit(5), not a_digit(6), not a_digit(7), not a_digit(8),
              not a_digit(9).

a_digit(2) :- not a_digit(0), not a_digit(1), not a_digit(3), not a_digit(4),
              not a_digit(5), not a_digit(6), not a_digit(7), not a_digit(8),
              not a_digit(9).

a_digit(3) :- not a_digit(0), not a_digit(1), not a_digit(2), not a_digit(4),
              not a_digit(5), not a_digit(6), not a_digit(7), not a_digit(8),
              not a_digit(9).

a_digit(4) :- not a_digit(0), not a_digit(1), not a_digit(2), not a_digit(3),
              not a_digit(5), not a_digit(6), not a_digit(7), not a_digit(8),
              not a_digit(9).

a_digit(5) :- not a_digit(0), not a_digit(1), not a_digit(2), not a_digit(3),
              not a_digit(4), not a_digit(6), not a_digit(7), not a_digit(8),
              not a_digit(9).

a_digit(6) :- not a_digit(0), not a_digit(1), not a_digit(2), not a_digit(3),
              not a_digit(4), not a_digit(5), not a_digit(7), not a_digit(8),
              not a_digit(9).

a_digit(7) :- not a_digit(0), not a_digit(1), not a_digit(2), not a_digit(3),
              not a_digit(4), not a_digit(5), not a_digit(6), not a_digit(8),
              not a_digit(9).

a_digit(8) :- not a_digit(0), not a_digit(1), not a_digit(2), not a_digit(3),
              not a_digit(4), not a_digit(5), not a_digit(6), not a_digit(7),
              not a_digit(9).

a_digit(9) :- not a_digit(0), not a_digit(1), not a_digit(2), not a_digit(3),
              not a_digit(4), not a_digit(5), not a_digit(6), not a_digit(7),
              not a_digit(8).

% ---- Exactly-one b_digit(D) via explicit defaults ----

b_digit(0) :- not b_digit(1), not b_digit(2), not b_digit(3), not b_digit(4),
              not b_digit(5), not b_digit(6), not b_digit(7), not b_digit(8),
              not b_digit(9).

b_digit(1) :- not b_digit(0), not b_digit(2), not b_digit(3), not b_digit(4),
              not b_digit(5), not b_digit(6), not b_digit(7), not b_digit(8),
              not b_digit(9).

b_digit(2) :- not b_digit(0), not b_digit(1), not b_digit(3), not b_digit(4),
              not b_digit(5), not b_digit(6), not b_digit(7), not b_digit(8),
              not b_digit(9).

b_digit(3) :- not b_digit(0), not b_digit(1), not b_digit(2), not b_digit(4),
              not b_digit(5), not b_digit(6), not b_digit(7), not b_digit(8),
              not b_digit(9).

b_digit(4) :- not b_digit(0), not b_digit(1), not b_digit(2), not b_digit(3),
              not b_digit(5), not b_digit(6), not b_digit(7), not b_digit(8),
              not b_digit(9).

b_digit(5) :- not b_digit(0), not b_digit(1), not b_digit(2), not b_digit(3),
              not b_digit(4), not b_digit(6), not b_digit(7), not b_digit(8),
              not b_digit(9).

b_digit(6) :- not b_digit(0), not b_digit(1), not b_digit(2), not b_digit(3),
              not b_digit(4), not b_digit(5), not b_digit(7), not b_digit(8),
              not b_digit(9).

b_digit(7) :- not b_digit(0), not b_digit(1), not b_digit(2), not b_digit(3),
              not b_digit(4), not b_digit(5), not b_digit(6), not b_digit(8),
              not b_digit(9).

b_digit(8) :- not b_digit(0), not b_digit(1), not b_digit(2), not b_digit(3),
              not b_digit(4), not b_digit(5), not b_digit(6), not b_digit(7),
              not b_digit(9).

b_digit(9) :- not b_digit(0), not b_digit(1), not b_digit(2), not b_digit(3),
              not b_digit(4), not b_digit(5), not b_digit(6), not b_digit(7),
              not b_digit(8).

% ---- Deterministic sum (no carry), NLP only (no arithmetic) ----

% sum(0)
sum(0) :- a_digit(0), b_digit(0).

% sum(1)
sum(1) :- a_digit(0), b_digit(1).
sum(1) :- a_digit(1), b_digit(0).

% sum(2)
sum(2) :- a_digit(0), b_digit(2).
sum(2) :- a_digit(1), b_digit(1).
sum(2) :- a_digit(2), b_digit(0).

% sum(3)
sum(3) :- a_digit(0), b_digit(3).
sum(3) :- a_digit(1), b_digit(2).
sum(3) :- a_digit(2), b_digit(1).
sum(3) :- a_digit(3), b_digit(0).

% sum(4)
sum(4) :- a_digit(0), b_digit(4).
sum(4) :- a_digit(1), b_digit(3).
sum(4) :- a_digit(2), b_digit(2).
sum(4) :- a_digit(3), b_digit(1).
sum(4) :- a_digit(4), b_digit(0).

% sum(5)
sum(5) :- a_digit(0), b_digit(5).
sum(5) :- a_digit(1), b_digit(4).
sum(5) :- a_digit(2), b_digit(3).
sum(5) :- a_digit(3), b_digit(2).
sum(5) :- a_digit(4), b_digit(1).
sum(5) :- a_digit(5), b_digit(0).

% sum(6)
sum(6) :- a_digit(0), b_digit(6).
sum(6) :- a_digit(1), b_digit(5).
sum(6) :- a_digit(2), b_digit(4).
sum(6) :- a_digit(3), b_digit(3).
sum(6) :- a_digit(4), b_digit(2).
sum(6) :- a_digit(5), b_digit(1).
sum(6) :- a_digit(6), b_digit(0).

% sum(7)
sum(7) :- a_digit(0), b_digit(7).
sum(7) :- a_digit(1), b_digit(6).
sum(7) :- a_digit(2), b_digit(5).
sum(7) :- a_digit(3), b_digit(4).
sum(7) :- a_digit(4), b_digit(3).
sum(7) :- a_digit(5), b_digit(2).
sum(7) :- a_digit(6), b_digit(1).
sum(7) :- a_digit(7), b_digit(0).

% sum(8)
sum(8) :- a_digit(0), b_digit(8).
sum(8) :- a_digit(1), b_digit(7).
sum(8) :- a_digit(2), b_digit(6).
sum(8) :- a_digit(3), b_digit(5).
sum(8) :- a_digit(4), b_digit(4).
sum(8) :- a_digit(5), b_digit(3).
sum(8) :- a_digit(6), b_digit(2).
sum(8) :- a_digit(7), b_digit(1).
sum(8) :- a_digit(8), b_digit(0).

% sum(9)
sum(9) :- a_digit(0), b_digit(9).
sum(9) :- a_digit(1), b_digit(8).
sum(9) :- a_digit(2), b_digit(7).
sum(9) :- a_digit(3), b_digit(6).
sum(9) :- a_digit(4), b_digit(5).
sum(9) :- a_digit(5), b_digit(4).
sum(9) :- a_digit(6), b_digit(3).
sum(9) :- a_digit(7), b_digit(2).
sum(9) :- a_digit(8), b_digit(1).
sum(9) :- a_digit(9), b_digit(0).

% sum(10)
sum(10) :- a_digit(1), b_digit(9).
sum(10) :- a_digit(2), b_digit(8).
sum(10) :- a_digit(3), b_digit(7).
sum(10) :- a_digit(4), b_digit(6).
sum(10) :- a_digit(5), b_digit(5).
sum(10) :- a_digit(6), b_digit(4).
sum(10) :- a_digit(7), b_digit(3).
sum(10) :- a_digit(8), b_digit(2).
sum(10) :- a_digit(9), b_digit(1).

% sum(11)
sum(11) :- a_digit(2), b_digit(9).
sum(11) :- a_digit(3), b_digit(8).
sum(11) :- a_digit(4), b_digit(7).
sum(11) :- a_digit(5), b_digit(6).
sum(11) :- a_digit(6), b_digit(5).
sum(11) :- a_digit(7), b_digit(4).
sum(11) :- a_digit(8), b_digit(3).
sum(11) :- a_digit(9), b_digit(2).

% sum(12)
sum(12) :- a_digit(3), b_digit(9).
sum(12) :- a_digit(4), b_digit(8).
sum(12) :- a_digit(5), b_digit(7).
sum(12) :- a_digit(6), b_digit(6).
sum(12) :- a_digit(7), b_digit(5).
sum(12) :- a_digit(8), b_digit(4).
sum(12) :- a_digit(9), b_digit(3).

% sum(13)
sum(13) :- a_digit(4), b_digit(9).
sum(13) :- a_digit(5), b_digit(8).
sum(13) :- a_digit(6), b_digit(7).
sum(13) :- a_digit(7), b_digit(6).
sum(13) :- a_digit(8), b_digit(5).
sum(13) :- a_digit(9), b_digit(4).

% sum(14)
sum(14) :- a_digit(5), b_digit(9).
sum(14) :- a_digit(6), b_digit(8).
sum(14) :- a_digit(7), b_digit(7).
sum(14) :- a_digit(8), b_digit(6).
sum(14) :- a_digit(9), b_digit(5).

% sum(15)
sum(15) :- a_digit(6), b_digit(9).
sum(15) :- a_digit(7), b_digit(8).
sum(15) :- a_digit(8), b_digit(7).
sum(15) :- a_digit(9), b_digit(6).

% sum(16)
sum(16) :- a_digit(7), b_digit(9).
sum(16) :- a_digit(8), b_digit(8).
sum(16) :- a_digit(9), b_digit(7).

% sum(17)
sum(17) :- a_digit(8), b_digit(9).
sum(17) :- a_digit(9), b_digit(8).

% sum(18)
sum(18) :- a_digit(9), b_digit(9).