function [A, error, F]= ADP(W, K, gnd, labeled_ind, adjacency)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% This code use a label information guided diffusion process to propagate
%%% label, with a learned affinity matrix as the by-product.
%%% Input:
%%%       --- W:              intitial weight matrix (symmetric and non-negtive);
%%%       --- K:              size of k-nearest neighbor
%%%       --- gnd:            Ground truth label;
%%%       --- labeled_ind:    the index of labeled samples;
%%%       --- adjecency:      NxN pre-defined adjecency matrix
%%% Output:
%%%       --- A:              learned affinity matrix;
%%%       --- F:              Classification function
%%%       --- error  :        prediction error;
%%% by QILIN LI (li.qilin@postgrad.curtin.edu.au)
%%% Last update 02/04/2020
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

if nargin < 5
    adjacency = W;
end

%% Pre-processing of weight matrix W
d = sum(W, 2);
D = diag(d + eps);
D_normalized = D ./ max(max(D));     %%% To be used in the iteration
n = size(W, 1);          %%% number of data points
W = W - diag(diag(W)) + D;   %%% use node degree as self-affinity

%%% Normalization  %%%%%%%%%%%%%%%%%%
% S = W ./ repmat(sum(W, 2)+eps, 1, n);

d = sum(W,2);
D = diag(d + eps);
S = D^(-1/2)*W*D^(-1/2);      %%% Symmetric normalization is better
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%% KNN sparse
S = knnSparse(S, K, adjacency);

%%% One-hot-encoded label matrix Y
num_class = length(unique(gnd));
Y = zeros(n, num_class);
for i = 1:length(labeled_ind)
    Y(labeled_ind(i), gnd(labeled_ind(i))) = 1;
end

%%% Main iteration
maxIter = 10;
epsilon = 1e-2;
alpha = 0.99;

%%% Initialization
F = Y;
A = S;

for t = 1:maxIter
    %%% Update F
    [F_new, error] = updateF(gnd, labeled_ind, A, 1e-3);
    err = norm(F_new - F, 'fro');
   
    if err < epsilon, break; end
    
    F = F_new;
    Z = F * F';
    DD = diag(sum(Z,2)+eps)^(-1/2);
    Z =  DD * Z * DD;
    Z = knnSparse(Z, K, adjacency); 
    
    %%% Update W
    for ii = 1:10 
        A_new = alpha*S*(A+Z)*S' + (1-alpha)*(D_normalized);
        A = A_new;
    end
    DD = diag(sum(A_new,2)+eps)^(-1/2);
    A_new =  DD * A_new * DD;
    A = knnSparse(A_new, K, adjacency);
       
    %fprintf('%.3f...', 1-error);
end
A = A_new;

fprintf("ADP converged at iteration %d.\n", t)

%% label correction
A = label_correction(A, gnd, labeled_ind);
A = (A+A')/2;
