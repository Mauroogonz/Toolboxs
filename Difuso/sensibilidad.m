function [p, indice, model]=sensibilidad(yent,Xent,reglas, options)
%se ingresa número de reglas
%la sálida del modelo: yent
% la matriz con las variables candidatas: Xent
%i: variable de entrada xi, r:numero de regla, j:numero de datos

%se genera el modelo takagi sugeno con el set de entradas cadidatas
[model, ~]=TakagiSugeno(yent,Xent,reglas,options);
% se definen los parametros del modelo obtenido   
a=model.a;% rxi
b=model.b; %rxi
g=model.g; %r x i+1

%se obtienen los valores de  grado de pertenencia mu
NR=size(a,1);
[Nd,n]=size(Xent);
muu = zeros(n,NR,Nd);
for r = 1:NR
    muu(:,r,:)=exp(-0.5*(a(r,:).*(Xent-b(r,:))).^2)';
end
w = squeeze(prod(muu,1))';

%c ixrxj
%se obtiene el c asociado a las derivadas
c = zeros(n,NR,Nd);
for r = 1:NR
    c(:,r,:)=(-(a(r,:).*(Xent-b(r,:))).*a(r,:))';
end

%se obtienen las salidas para cada regla
%yr jxr
yr=[ones(length(Xent(:,1)),1),Xent]*g';

%se obtienen los chi para cada candidata
%chi jxi
suma1 = 0;
suma2 = 0;
suma3 = 0;
suma4 = 0;

for r=1:NR
    c_aux = reshape(squeeze(c(:,r,:)),[Nd,n]);
    suma1 = suma1 + w(:,r).*c_aux.*yr(:,r)+g(r,2:end).*w(:,r);
    suma2 = suma2 + w(:,r);
    suma3 = suma3 + w(:,r).*c_aux;
    suma4 = suma4 + w(:,r).*yr(:,r);
end

chi = (suma1.*suma2 - suma3.*suma4)./((suma2+1e-10).^2);

%indice 1 x i
%se obtiene finalmente el indice para cada candidata
indice = mean(chi).^2+std(chi).^2;
[~, p] = min(indice);

%se grafican los indices obtenidos para compararse
figure ()
bar(indice)
ylabel('Índice de sensibilidad')
xlabel('Entrada del modelo')  
end