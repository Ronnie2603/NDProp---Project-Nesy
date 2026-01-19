power.
a :- power, not b.
b :- power, not a.
c :- power, not d.
d :- power, not c.
t1 :- a, c.
bad1 :- t1, not bad1.
t2 :- b, d.
bad2 :- t2, not bad2.