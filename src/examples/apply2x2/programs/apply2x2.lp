%===========================================
% APPLY 2x2 (operators on 3 digits)
%===========================================

#const min=-100.
#const max=1000.
1 { digits(D1, D2, D3) : D1=0..10, D2=0..10, D3=0..10 } 1.
1 { op(0,O) : O=0..2 } 1.
1 { op(1,O) : O=0..2 } 1.
1 { op(2,O) : O=0..2 } 1.
1 { op(3,O) : O=0..2 } 1.

apply(D1,0,D2,0,D3,R) :- digits(D1,D2,D3), R = (D1 + D2) + D3, R = min..max.
apply(D1,0,D2,1,D3,R) :- digits(D1,D2,D3), R = (D1 + D2) - D3, R = min..max.
apply(D1,0,D2,2,D3,R) :- digits(D1,D2,D3), R = (D1 + D2) * D3, R = min..max.

apply(D1,1,D2,0,D3,R) :- digits(D1,D2,D3), R = (D1 - D2) + D3, R = min..max.
apply(D1,1,D2,1,D3,R) :- digits(D1,D2,D3), R = (D1 - D2) - D3, R = min..max.
apply(D1,1,D2,2,D3,R) :- digits(D1,D2,D3), R = (D1 - D2) * D3, R = min..max.

apply(D1,2,D2,0,D3,R) :- digits(D1,D2,D3), R = (D1 * D2) + D3, R = min..max.
apply(D1,2,D2,1,D3,R) :- digits(D1,D2,D3), R = (D1 * D2) - D3, R = min..max.
apply(D1,2,D2,2,D3,R) :- digits(D1,D2,D3), R = (D1 * D2) * D3, R = min..max.

row(0,R) :- digits(D1,D2,D3), op(0,O1), op(1,O2), apply(D1,O1,D2,O2,D3,R).
row(1,R) :- digits(D1,D2,D3), op(2,O3), op(3,O4), apply(D1,O3,D2,O4,D3,R).

col(0,R) :- digits(D1,D2,D3), op(0,O1), op(2,O3), apply(D1,O1,D2,O3,D3,R).
col(1,R) :- digits(D1,D2,D3), op(1,O2), op(3,O4), apply(D1,O2,D2,O4,D3,R).
