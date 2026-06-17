function PINAW = calc_pinaw(y_up, y_lw, y_t)
% Ancho promedio del intervalo, normalizado por el rango de y_t.
%
% PINAW = mean(y_up - y_lw) / (max(y_t) - min(y_t))
%

y_up = y_up(:);
y_lw = y_lw(:);
y_t  = y_t(:);

R = max(y_t) - min(y_t);         % rango de la salida real
if R < eps
    R = 1;                        % evitar división por cero
end

PINAW = mean(y_up - y_lw) / R;
end