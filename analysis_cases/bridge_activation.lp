% bridge_activation_no_constraint.lp
% Mutually default-choice switch
switch_on  :- not switch_off.
switch_off :- not switch_on.

% Positive loop that should run only when "on"
engine_running :- fuel_flow.
fuel_flow      :- engine_running.
fuel_flow      :- switch_on.   % external support enables the loop

% Eliminate the "off" branch without constraints:
dead_end :- switch_off, not dead_end.  % odd self-loop guarded by switch_off