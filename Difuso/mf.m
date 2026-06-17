function Wn=mf(X,a,b)

% y is the vector of outputs when evaluating the TS defined by a,b,g
% X is the data matrix

% Nd number of point we want to evaluate
% n is the number of regressors of the TS model


[Nd,n]=size(X);

% NR is the number of rules of the TS model
NR=size(a,1);
mu = zeros(NR,n,Nd);

for r = 1:NR
    mu(r,:,:)=exp(-0.5*(a(r,:).*(X-b(r,:))).^2)';
end
W = squeeze(prod(mu,2))' + 1e-15;
Wn =  W./sum(W,2);



end