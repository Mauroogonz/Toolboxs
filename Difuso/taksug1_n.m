%% g = taksug1(y,Z,a,b,opcion)
%
% Identificación de Parámetros de las consecuencias usando LMS global
%
% Inputs:
%   * y: Vector de salidas del modelo TS
%   * Z: Matriz de entradas del modelo TS
%   * a: (Std)^-1 de los clusters del modelo TS
%   * b: Centros de los clusters del modelo TS
%   * opcion: [Tipo de identificación, Tipo de normalización]
%       * opcion(1) = 1: LMS con todos los datos
%       * opcion(1) = 4: LMS con todos los datos para consecuencias
%                        lineales
% Ouputs:
%   * g: Matriz de parámetros de las consecuencias del modelo TS

function [g, P, h] = taksug1_n(y,Z,a,b,opcion)

[N,n] = size(Z);    % N: Número de datos. n: Número de variables
Nr = size(a,1);     % Número de reglas
h = zeros(N,Nr);  	% Grados de activación normalizados
%keyboard
%% Cálculo de grados de activación
mu = zeros(Nr,n,N);
for r = 1:Nr
    mu(r,:,:)=exp(-0.5*(a(r,:).*(Z-b(r,:))).^2)';
end
W = squeeze(prod(mu,2))' + 1e-15;
h =  W./sum(W,2);
%keyboard
%% Formulación de Z_fuzzy
Z_fuzzy = zeros(N,Nr*n);  % y_fuzzy = Z_Fuzzy*theta (theta: vector de parámetros)
for i = 1:Nr:n*Nr
     Z_fuzzy(:,i:i+Nr-1) = h.*kron(ones(1,Nr),Z(:,ceil(i/Nr)));
end

if opcion(1) == 1           % Consecuencias afines
    Z_fuzzy = [h,Z_fuzzy];
end

%% Parámetros
theta = Z_fuzzy\y;          % Parámetros

if opcion(1) == 4           % Consecuencias lineales
    g = reshape(theta,[Nr,n]);
elseif opcion(1) == 1     	% Consecuencias afines
    g = reshape(theta,[Nr,n+1]);
end

%% Formulación Covarianza
Z=[ones(N,1) Z];
size_z = size(Z);
Z_cov = zeros(N,size_z(2),Nr);
size_z_cov = size(Z_cov);
P = zeros(size_z_cov(2), size_z_cov(2), Nr);
% for j=1:Nr
%     for k=1:N
%         Z_cov(k,:,j)=h(k,j).*Z(k,:);
%     end
%     P(:,:,j)=inv(Z_cov(:,:,j)'*Z_cov(:,:,j));
% end

end

