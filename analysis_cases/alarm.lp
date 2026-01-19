% Deterministic motion + light; alarm depends on a disable/arm choice.
motion.
light.
alarm    :- motion, not disabled.
disabled :- not armed.
armed    :- light, not disabled.