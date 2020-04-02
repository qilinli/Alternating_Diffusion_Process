function [F, error] = updateF(gnd, labeled_ind, W, lambda)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Update the classification function F
%%% Input:
%%%       --- gnd:            Ground truth label;
%%%       --- labeled_ind:    the index of labeled samples;
%%%       --- W:              affinity matrix;
%%% Output:
%%%       --- F:              NxC classification function;
%%%       --- error  :        prediction error;
%%% by QILIN LI (li.qilin@postgrad.curtin.edu.au)
%%% Last update 26/03/2020
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

n = length(gnd);
nl = length(labeled_ind);    %%% number of labeled points
c = length(unique(gnd));

unlabeled_ind = setdiff(1:n, labeled_ind);
gnd_unlabeled = gnd(unlabeled_ind);
gnd = [gnd(labeled_ind); gnd(unlabeled_ind)];

%%% Initial label matrix Y (one-hot encoded)
Y = zeros(n, c);
for i = 1:length(labeled_ind)
    Y(labeled_ind(i), gnd(i)) = 1;
end

Y_labeled = Y(labeled_ind,:);   %% Y matrix for labeled points

%%% Re-order W so that the labeled points come first
W_ll = W(labeled_ind, labeled_ind);
W_lu = W(labeled_ind, unlabeled_ind);
W_ul = W(unlabeled_ind, labeled_ind);
W_uu = W(unlabeled_ind, unlabeled_ind);
WW = [W_ll W_lu; W_ul W_uu];

%%% Update F for unlabeled points (Fu = -Luu^-1*Lul*Yl)
D = diag(sum(WW,2)+eps);
DD = D^(-1/2);
L = DD*(D - WW)*DD;    %%% normalized graph laplacian

L_uu = L(nl+1:n, nl+1:n);
%%%% For numberic stablity
L_uu = L_uu + lambda * eye(length(L_uu));
L_ul = L(nl+1:n, 1:nl);
F_unlabeled = -pinv(L_uu) * L_ul * Y_labeled;

%%% Arrange back to the order
%%% F = Y for the labeled points
F = zeros(n, c);
for i = 1:length(labeled_ind)
    F(labeled_ind(i), gnd(i)) = 1;
end

for i = 1:length(unlabeled_ind)
    F(unlabeled_ind(i), :) = F_unlabeled(i, :);
end

%%% Calculate the error
[~, y_unlabeled] = max(F_unlabeled, [], 2);

%% Use gnd = 0 to supress certain points (e.g. noise & outlier)
if isempty(gnd_unlabeled)
    error = 0;
else
    error = sum((y_unlabeled ~= gnd_unlabeled) & (gnd_unlabeled ~= 0)) / ...
    sum(unlabeled_ind ~= 0);
end

