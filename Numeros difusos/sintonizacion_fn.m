function [s_up, s_lw] = sintonizacion_fn_multistep(data_y, y_p, zin, model, CP)
% SINTONIZACION_FN_MULTISTEP  Versión adaptada de sintonizacion_fn.m del
% toolbox. La diferencia: recibe y_p y zin como input externo, en vez de
% generarlos internamente con ysim. Esto permite usar predicciones
% recursivas a cualquier horizonte.
%
% Entradas:
%   data_y : vector N x 1 de valores reales
%   y_p    : vector N x 1 de predicciones del modelo (puede ser recursiva)
%   zin    : matriz N x p de regresores (los mismos que generaron y_p)
%   model  : struct del modelo T&S (debe contener .a, .b, .g)
%   CP     : cobertura deseada en decimal (ej: 0.95)
%
% Salidas:
%   s_up, s_lw : matrices R x (p+1) de spreads superiores e inferiores

%% Configuración del optimizador PSO (idéntica al toolbox original)
options = optimoptions('particleswarm','Display','iter');
options.SwarmSize           = 100;
options.FunctionTolerance   = 1e-4;
options.MaxStallIterations  = 30;
options.UseParallel         = false;   % cambiar a true si tienes parallel toolbox

%% Número de variables a optimizar
[R_n, R_p1] = size(model.g);
nvars = 2 * R_n * R_p1;                % spreads up + spreads lw

%% Ponderadores y cobertura (mismos defaults del toolbox)
param.eta1     = 100;
param.eta2     = 200;
param.coverage = CP;

%% Función de costos
fun = @(spreads) fn_obj_sint(spreads, data_y, y_p, zin, model, param);

%% Límites para el optimizador
lb = zeros(nvars, 1);
ub = 100 * ones(nvars, 1);

%% Optimización
[sol, ~, ~, ~] = particleswarm(fun, nvars, lb, ub, options);

%% Separar en superiores e inferiores, reformar como matrices
n_spreads = R_n * R_p1;
s_up_vec = sol(1:n_spreads);
s_lw_vec = sol(n_spreads+1:end);

s_up = reshape(s_up_vec, R_n, R_p1);
s_lw = reshape(s_lw_vec, R_n, R_p1);
end