function PICP = calc_picp(y_up, y_lw, y_t)
% CALC_PICP  Prediction Interval Coverage Probability.
% Fracción de valores reales que caen dentro del intervalo [y_lw, y_up].
 
y_up = y_up(:);
y_lw = y_lw(:);
y_t  = y_t(:);
 
cubiertos = (y_t >= y_lw) & (y_t <= y_up);
PICP = mean(cubiertos);
end