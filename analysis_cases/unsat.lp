a :- not b.
b :- not a.
t1 :- a.
bad1 :- t1, not bad1.
t2 :- b.
bad2 :- t2, not bad2.
