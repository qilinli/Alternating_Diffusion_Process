function [Y, error]=grf(gnd,labeled_ind,W)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% This code is the implementation of the graph transductive learning algorithm in the paper
%%% "Learning with Local and Global Consistency", by D. Zhou et al, Proc. NIPS 2004
%%% implemented by Jun Wang EE@Columbia (jwang@ee.columbia.edu);
%%% Dependency [neighborhood, weights] = LLE_Weights(X, K, WeightModel);
%%% Input:
%%%       --- data:           Input Dataset (nXd);
%%%       --- gnd:            Ground truth label;
%%%       --- labeled_ind:    the index of labeled samples;
%%%       --- graph:          Graph built for inference;
%%%       --- graph.W         Weight matrix (symetric and non-negative, diagnal elements are zeros);
%%%       --- graph.L         Graph Laplacian (normalized);
%%%       --- graph.IS        Prediction matrix IS=(I-\alpha L)^-1;
%%% Output:
%%%       --- predict:        predicted label;
%%%       --- error  :        prediction error;
%%% by Jun Wang (jwang@ee.columbia.edu)
%%% Last update Dec 8, 2008

gnd=reshape(gnd,1,length(gnd));
data_num=length(gnd);
class_num=length(unique(gnd(gnd~=0)));

unlabeled_ind=setdiff([1:data_num], labeled_ind);

gnd_unlabeled=gnd(unlabeled_ind);

gnd=[gnd(labeled_ind) gnd(unlabeled_ind)];

fl=zeros(length(labeled_ind),class_num);

for i=1:length(labeled_ind)
    fl(i,gnd(i))=1;    
end

W=full(W);
W_ll=W(labeled_ind,labeled_ind);
W_lu=W(labeled_ind,unlabeled_ind);    
W_ul=W(unlabeled_ind,labeled_ind);
W_uu=W(unlabeled_ind,unlabeled_ind); 
WW=[W_ll W_lu; W_ul W_uu];

[fu, ~] = harmonic_function(WW, fl);
[~, b] = max(fu,[],2);

if isempty(gnd_unlabeled)
    error = 0;
else
    error=sum((gnd_unlabeled~=b')&(gnd_unlabeled~=0))/(sum(gnd~=0)-length(labeled_ind));
end

num_class = length(unique(gnd));
Y = zeros(data_num, num_class);
for i = 1:length(labeled_ind)
    Y(labeled_ind(i), gnd(i)) = 1;
end

for i = 1:length(unlabeled_ind)
    Y(unlabeled_ind(i), :) = fu(i,:);
end



