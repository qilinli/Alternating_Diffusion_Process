function [X_sub, Y_sub] = random_data_subset(X, Y, n)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% This function randomly sample a subset of data.
%%% Input:
%%%       --- X:              N*D original data
%%%       --- Y:              N*1 label
%%%       --- n:              n samples per class
%%% Output:
%%%       --- X:              sampled subset data
%%%       --- Y:              sampled subset label
%%% by QILIN LI (li.qilin@postgrad.curtin.edu.au)
%%% Last update 04/05/2019
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 

% random permutation
num_sample = size(X, 1);
idx = randperm(num_sample);
X = X(idx,:);
Y = Y(idx);

idx_sub = [];
num_class = length(unique(Y));
for i = 1:num_class
    idx_i = find(Y == i);
    idx_sub = [idx_sub; idx_i(1:n)];
end
X_sub = X(idx_sub,:);
Y_sub = Y(idx_sub);