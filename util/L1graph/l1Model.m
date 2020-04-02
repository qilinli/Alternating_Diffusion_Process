function graph=l1Model(data, alpha)

affine = 0;
CMat = admmOutlier_mat_func(data',affine,alpha);
N = size(data,1);
C = CMat(1:N,:);
W = abs(C) + abs(C');

%%% for GTAM parameter
mu=0.01;
alpha=1/(1+mu);
graph.W=W;

W=full(W);

%%% for LGC
D=full(diag(sum(W)));
if length(W)<4000
    D1=D^(-0.5);
else
    d=diag(D);
    D1=diag(d.^(-0.5));
end

S=D1*W*D1;
I=eye(length(W),length(W));
IS=(I-alpha*S)^(-1);
graph.IS=IS;

%%% for GTAM
L=D1*(D-W)*D1;
P=(L/mu+I)^(-1);
A=P'*L*P+(P'-I)*(P-I);

graph.A=A;
graph.P=P;
graph.L=L;
graph.IS=IS;

