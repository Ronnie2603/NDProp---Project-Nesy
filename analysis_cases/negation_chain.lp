% exactly_one_3_unique.lp
alice :- not bob, not carol.
bob   :- not alice, not carol.
carol :- not alice, not bob.

% Guarded odd loops eliminate two branches (no constraints)
bad_bob   :- bob,   not bad_bob.
bad_carol :- carol, not bad_carol.