function [int_up, int_lw] = calc_intervals(zin, a, b, s_up, s_lw)
% CALC_INTERVALS  Calcula el spread superior e inferior del intervalo
% de predicción por números difusos, para un modelo T&S.
%
% La salida es el INCREMENTO a sumar/restar al valor central, NO el
% intervalo absoluto. Es decir:
%     y_up = y_p + int_up
%     y_lw = y_p - int_lw
%
% Entradas:
%   zin  : N x p  matriz de regresores (cada fila una muestra)
%   a,b  : R x p  parámetros (desviaciones inv. y centros) de los clusters
%   s_up : R x (p+1) matriz de spreads superiores (afines)
%   s_lw : R x (p+1) matriz de spreads inferiores (afines)
%
% Salidas:
%   int_up, int_lw : vectores N x 1

[N, p] = size(zin);
R = size(a, 1);

%% --- 1) Grados de activación normalizados (igual que ysim) ------------
mu = zeros(R, p, N);
for r = 1:R
    mu(r, :, :) = exp(-0.5 * (a(r,:) .* (zin - b(r,:))).^2)';
end
W  = squeeze(prod(mu, 2))' + 1e-15;     % N x R
Wn = W ./ sum(W, 2);                     % N x R (normalizado)

%% --- 2) Salida de cada regla con los spreads (afín) -------------------
% s_up es R x (p+1): [s_0, s_1, ..., s_p] para cada regla.
% La salida de la regla r con spread superior en el punto k es:
%     s_up_r(k) = s_up(r,1) + s_up(r,2)*zin(k,1) + ... + s_up(r,p+1)*zin(k,p)
%               = [1, zin(k,:)] * s_up(r,:)'
% Vectorizado sobre todos los puntos:
zin_aug = [ones(N,1), zin];              % N x (p+1)
s_up_per_rule = zin_aug * s_up';         % N x R
s_lw_per_rule = zin_aug * s_lw';         % N x R

%% --- 3) Agregación ponderada -----------------------------------------
% Igual estructura que la salida del T&S pero usando los spreads
int_up = sum(s_up_per_rule .* Wn, 2);    % N x 1
int_lw = sum(s_lw_per_rule .* Wn, 2);    % N x 1
end