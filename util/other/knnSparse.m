function W_knn = knnSparse(W, K, adjecency)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Sparse the affinity matrix by k-nearst neighbor %%%
%%% Input:
%%%       W: NxN affinity matrix
%%%       K: number of neighbors, ignored if adjecency exists
%%%       adjecency: NxN pre-defined adjecency matrix
%%% Output:
%%%       W_knn: NxN sparsed affinity matrix
%%%
%%% by QILIN LI (li.qilin@postgrad.curtin.edu.au)
%%% Last update 29/06/2018
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

if nargin == 3
    W_knn = W .* adjecency;
else
    n = size(W,1);
    [~, idx_knn] = sort(W, 2, 'descend');
    W_knn = zeros(n, n);
    for ii = 1:n
        W_knn(ii, idx_knn(ii, 1:K)) = W(ii, idx_knn(ii, 1:K));
    end
    W_knn = (W_knn+W_knn')/2;
end
