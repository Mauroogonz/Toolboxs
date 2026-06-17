%% [model,GK] = TakagiSugeno(y,Z,reglas,opcion,varargin)
%
% Entrenar modelo TS
%
% Inputs:
%   * y: Vector de salida
%   * Z: Matriz de datos de entrada
%   * reglas: Número de reglas (clusters)
%   * opcion: [Tipo de identificación, Tipo de normalización, Tipo de Clustering]
%       * opcion(1) = 1: LMS con todos los datos
%       * opcion(1) = 2: LMS para cada regla (dejando fuera puntos con w bajo)
%       * opcion(1) = 3: Identificación de cada regla
%       * opcion(1) = 4: LMS con todos los datos para consecuencias
%                        lineales
%       * opcion(1) = 5: LMS para cada regla (dejando fuera ptos con w bajo)
%                        para consecuencias lineales
%       * opcion(1) = 6: Identificación de cada regla para consecuencias
%                        lineales
%       * opcion(1) = 7: Identificación con controlabilidad asegurada
%       * opcion(1) = 8: Identificación con estabilidad asegurada
%       * opcion(1) = 9: Identificación a N pasos con consecuencias
%       lineales
%       * opcion(2) = 1: Normalización lineal
%       * opcion(2) = 2: Normalización gaussiana
%       * opcion(2) = 3: 1ro Normalización gaussiana, 2do Normalización lineal
%       * opcion(2) = 4: Sin normalización
%       * opcion(3) = 1: Clustering con Gustafson-Kessel
%       * opcion(3) = 2: Clustering con Fuzzy C-Means
%   * varargin: (ny,nu,lambda,check). Usar sólo en caso de que opcion(1) =
%   7, 8 o 9
%       * ny: Retardos de y
%       * nu: Retardos de u
%       * lambda: Penalización por no ser controlable/estable
%       * check (Controlabilidad): [Solver,Verificación condición de controlabilidad,...
%            Inicialización de parámetros,Tipo de Controlabilidad,Identificación a N pasos]
%           * check(1) = 1: Utilizar fminunc (gradient based)
%           * check(1) = 2: Utilizar fminsearch (derivative-free based)
%           * check(1) = 3: Utilizar PSO
%           * check(2) = 1: Utilizar det(C)
%           * check(2) = 2: Utilizar rank(C)
%           * check(3) = 1: Inicialización aleatoria de parámetros
%           * check(3) = 2: Inicialización de parámetros usando LMS Global
%           * check(4) = 1: Utilizar C_fuzzy
%           * check(4) = 2: Utilizar C_lin
%           * check(5) = 1: Identificación a 1 paso
%           * check(5) = 2: Identificación a N pasos
%       * check (Estabilidad): [Solver,Inicialización de parámetros,Identificación a N pasos]
%           * check(1) = 1: Utilizar fminunc (gradient based)
%           * check(1) = 2: Utilizar fminsearch (derivative-free based)
%           * check(1) = 3: Utilizar PSO
%           * check(2) = 1: Inicialización aleatoria de parámetros
%           * check(2) = 2: Inicialización de parámetros usando LMS Global
%           * check(3) = 1: Identificación a 1 paso
%           * check(3) = 2: Identificación a N pasos
%       * check (N pasos): [Solver,Inicialización de parámetros]
%           * check(1) = 1: Utilizar fminunc (gradient based)
%           * check(1) = 2: Utilizar fminsearch (derivative-free based)
%           * check(1) = 3: Utilizar PSO
%           * check(2) = 1: Inicialización aleatoria de parámetros
%           * check(2) = 2: Inicialización de parámetros usando LMS Global
% Outputs:
%   * model: Estructura con el modelo TS
%       * model.a: (Std^-1) de los clusters
%       * model.b: Centros de los clusters
%       * model.g: Parámetros de las consecuencias
%       * model.exitflag: Condición de término del solver empleado (Sólo para opción(1) == {7,8,9})
%   * GK: Resultado del Clustering GK

function [model,GK] = TakagiSugeno(y,Z,reglas,opcion,varargin)

%% Normalización
data.X = [y,Z];   % Todos los datos

if opcion(2) == 1
    data = clust_normalize(data,'range');   % (data.min,data.max) para desnormalizar
elseif opcion(2) == 2
    data = clust_normalize(data,'var');     % (data.mean,data.std) para desnormalizar
elseif opcion(2) == 3
    data = clust_normalize(data,'var2');    % (data.mean,data.std) para desnormalizar
else
    error('Use a valid normalization option');
end
% data.Xold: Data original
% data.X: Data normalizada

%% Identificación de Parámetros de las Premisas
%keyboard
% Parámetros GK
n           = length(Z(1,:));       % n: Número de variables
param.m     = 2;                    % Exponente de peso
% param.e     = 1e-5;                 % Tolerancia
param.c     = reglas;               % Número de clusters
% param.ro    = ones(1,param.c);     	% det(Ai) = ro_i = 1
% param.gamma = 0.5;                 	% Ponderador [0,1]

if opcion(3)==1
    param.DistanceMetric = "mahalanobis";
    GK = FCMclustering(data,param);      % FCM clustering
    a = GK.cluster.std1(:,2:end);   % (Std)^-1 de los clusters normalizados
    b = GK.cluster.v(:,2:end);      % Centros de los clusters normalizados
elseif opcion(3)==2
    GK = FCMclustering(data,param);      % FCM clustering
    a = GK.cluster.std1(:,2:end);   % (Std)^-1 de los clusters normalizados
    b = GK.cluster.v(:,2:end);      % Centros de los clusters normalizados
else
    error('Use a valid clustering option');
end

Fn = GK.cluster.F(2:end,2:end,:);

%% Denormalización

if opcion(2) == 1       % Lineal
    for i = 1:n
        xmax = data.max(1,i+1);
        xmin = data.min(1,i+1);
        dx = xmax - xmin;
        a(:,i) = a(:,i)*(1/dx);
        b(:,i) = xmin + dx*b(:,i);
    end
end

if opcion(2) == 2       % Gaussiana
    for i = 1:n
        xmean = data.mean(1,i+1);
        %dx = 2*data.std(1,i+1);
        dx = data.std(1,i+1);
        a(:,i) = a(:,i)*(1/dx);
        b(:,i)  = xmean + dx*b(:,i);
    end
end
%keyboard

if opcion(2) == 3       % Gaussiana + Lineal
    for i = 1:n         % Lineal
        xmax = data.max(1,i+1);
        xmin = data.min(1,i+1);
        dx = xmax - xmin;
        a(:,i) = a(:,i)*(1/dx);
        b(:,i) = xmin + dx*b(:,i);
    end
    for i = 1:n         % Gaussiana
        xmean = data.mean(1,i+1);
        dx = 2*data.std(1,i+1);
        a(:,i) = a(:,i)*(1/dx);
        b(:,i) = xmean + dx*b(:,i);
    end
end

for c = 1:reglas
    for i=1:n
        for j=1:n
            F(i,j,c) = Fn(i,j,c)*(a(c,i)^-1)*(a(c,j)^-1);
        end
    end
end

model.a = a;    % (Std^-1) de los clusters desnormalizadas
model.b = b;    % Centros de los clusters desnormalizados
model.F = F;

%% Identificación de Parámetros de las Consecuencias
% (Consecuencias afines || lineales)

% if opcion(1) == 1 || opcion(1) == 4
%     g = taksug1(y,Z,a,b,opcion);    % LMS con todos los datos
% elseif opcion(1) == 2 || opcion(1) == 5
%     g = taksug2(y,Z,a,b,opcion);    % LMS para cada regla (dejando fuera puntos con w bajo)
% elseif opcion(1) == 3 || opcion(1) == 6
%     g = taksug3(y,Z,a,b,opcion);    % Identificación de cada regla
% elseif opcion(1) == 7      % Identificación con controlabilidad asegurada
%     params = cell2mat(varargin);         % params = [ny,nu,lambda,check]
%     ny = params(1);
%     nu = params(2);
%     lambda = params(3);
%     check = [params(4),params(5),params(6),params(7),params(8)];
%     [g,exitflag] = taksug4(y,Z,a,b,opcion,ny,nu,lambda,check);
%     model.exitflag = exitflag;
% elseif opcion(1) == 8      % Identificación con estabilidad asegurada
%     params = cell2mat(varargin);         % params = [ny,nu,lambda,check]
%     ny = params(1);
%     nu = params(2);
%     lambda = params(3);
%     check = [params(4),params(5),params(6)];
%     [g,exitflag] = taksug5(y,Z,a,b,opcion,ny,nu,lambda,check);
%     model.exitflag = exitflag;
% elseif opcion(1) == 9       % Identificación a N pasos con Consecuencias Lineales
%     params = cell2mat(varargin);         % params = [ny,nu,check]
%     ny = params(1);
%     nu = params(2);
%     check = [params(3),params(4)];
%     [g,exitflag] = taksug6(y,Z,a,b,opcion,ny,nu,check);
%     model.exitflag = exitflag;
% end
P = []; h = [];
% STEP 4: Calculation of the consequences
if(opcion(1)==1)
    %g=taksug2(y,Z,a,b);% %Global LMS with all the data
    [g, P, h]=taksug1_n(y,Z,a,b,1); %Global LMS with all the data

elseif(opcion(1)==2)
    [g, P, h]=taksug1_n(y,Z,a,b,1);
    [g]=taksug3(y,Z,a,b);%Identification of each rule separately

elseif (opcion(1)==3)
    g=taksug1(y,Z,a,b);%Identification of each rule, in a fast way

elseif (opcion(1)==4)
    g=taksug2(y,Z,a,b);% %Global LMS with all the data

else
    error('Use a valid identification option');
end

model.g=g;
model.P=P;
model.h=h;


end

