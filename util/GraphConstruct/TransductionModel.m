function graph=TransductionModel(W)

%%% Graph construction
%%% Input:
%%%       W: NxN symmetric affinity matrix
%%% Output:
%%%        graph
%%%
%%% Copyrights @ QILIN LI, 19/04/2018


%%% Parameter
mu = 0.01;        %%% GGMC
alpha = 0.99;     %%% LGC

W = full(W);
D = diag(sum(W)+eps);
D1 = D^(-1/2);
I = eye(length(W));

%%% LGC
S = D1*W*D1;
IS = (I - alpha*S)^(-1);

%%% GGMC
L = D1*(D - W)*D1;
P = (L/mu + I)^(-1);
A = P'*L*P + (P'-I)*(P-I);

%%% Graph construction
graph.W = W;
graph.L = L;
graph.IS = IS;
graph.P = P;
graph.A = A;

