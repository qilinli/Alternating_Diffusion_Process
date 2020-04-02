function [data, gnd] = dataGenerator_subspaceData(n, d, k)

%%% Synthetic Subspace Data Generator %%%%
%%% Input:
%%%       n: the numbner of data points
%%%       d: the dimensionality
%%%       k: the number of class
%%%
%%% Output:
%%%       data: nxd data matrix
%%%       gnd: nx1 label vector
%%%
%%% Copyrights @ QILIN LI, 04/05/2018
% 


U = randn(d, k);
U = orth(U);
data = []; gnd=[];
for i = 1:k
    Ti = randn(d, d);
    Ui = Ti*U;
    Qi = 0 + 1*randn(k, floor(n/k));
    Xi = Ui*Qi;
    data = [data Xi];
    gnd = [gnd ones(1,floor(n/k))*i];
end
data = data';
gnd = gnd';
    

        