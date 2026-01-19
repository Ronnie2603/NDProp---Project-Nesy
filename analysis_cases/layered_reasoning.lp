% Deterministic facts feed into a default-negation choice.
sunny.
sprinkler :- sunny.
wet_grass :- sprinkler.
wet_grass :- rain.
rain :- not forecast_clear.
forecast_clear :- not rain.

