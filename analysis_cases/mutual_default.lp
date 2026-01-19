% Symmetric mutual default-negation cycle probing tie-breaking.
p :- not q.
q :- not p.
decided :- p.
decided :- q.

