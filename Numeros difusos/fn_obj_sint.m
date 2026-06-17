function J = fn_obj_sint(spreads, data_y, y_p, zin, model, param)
% fn_obj_sint Calcula la funcion objetivo del metodo de numeros difusos
% Se asume que los spreads son conocidos en esta etapa
% Variables
%   - spreads: vector que contiene todos los spreads (primero los superiores
%              y luego los inferiores
%   - data_y: vector de datos de salida reales
%   - y_p: vector con la prediccion asociada a data_y
%   - zin: matriz de regresores siendo el primer regresor data_y(k-1)
%   - model: estructura que contiene los parametros del modelo difuso
%   - param: estructura con parametros para la optimizacion


% Separar entre spreads superiores e inferiores
[N_reglas, N_regresores]=size(model.g);
n_spreads = N_reglas*N_regresores;
s_up = spreads(1:n_spreads);
s_lw = spreads(n_spreads+1:end);

% Transformar vector en matriz
s_up = reshape(s_up, N_reglas, N_regresores); % Matriz de spreads (N_reglas x N_regresores)
s_lw = reshape(s_lw, N_reglas, N_regresores); % Matriz de spreads (N_reglas x N_regresores)


% Calcular intervalos de prediccion
% Aqui se asume que s_up, s_inf son conocidos
[int_up,int_lw] = calc_intervals(zin, model.a, model.b , ...
    s_up, s_lw);

y_up = y_p + int_up; % Intervalo superior
y_lw = y_p - int_lw; % Intervalo inferior

% Calcular el costo a optimizar
PINAW = calc_pinaw(y_up, y_lw, data_y);
PICP = calc_picp(y_up, y_lw, data_y);

J = param.eta1*PINAW + exp(-param.eta2*(PICP - param.coverage));

end