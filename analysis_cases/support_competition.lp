% Competing support with mixed positive/negative influences.
base.
a :- base.
b :- base, not inhibit.
inhibit :- trigger.
trigger :- not stable.
stable :- not trigger.
goal :- a, b.

