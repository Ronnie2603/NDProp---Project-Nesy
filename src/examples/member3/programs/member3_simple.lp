1 { digit(0,V) : V=0..9 } 1.
1 { digit(1,V) : V=0..9 } 1.
1 { digit(2,V) : V=0..9 } 1.

1 { check(V) : V=0..9 } 1.

member(D, 0) :- digit(0, N1), digit(1, N2), digit(2, N3), check(D), D != N1, D != N2, D != N3.
member(D, 1) :- check(D), not member(D, 0).

