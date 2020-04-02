clear;
clc;
close all;

addpath('util/GraphConstruct/');
addpath('util/other/');
addpath('util/L1graph/');
warning('off', 'MATLAB:singularMatrix');
warning('off', 'MATLAB:nearlySingularMatrix');

load data/five_circle_N500_D2_V2.mat

%%%%%%%%%%%%%%%%%%%%%%%%%%%%% parameters %%%%%%%%%%%%%%%%%%%%%%%%%%
K                       = 5;       % k-nearst neighbor
alpha                   = 10;       % sparsity regularizer in lasso
lambda                  = 1e-4;     % for numeric stablity in graph laplacian inverse;
replicates              = 1;        % # of replicates experiment
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

class_num = length(unique(Y));
data_num = length(Y);
Labels_num = class_num * [1];

%%%%%%%%%% Choice of initial W %%%%%%%%%%%%%%%%%
[W_full,W_knn] = adaptiveGaussian(X, 10);

% C = admmOutlier_mat_func(X', 0, alpha);
% C = C(1:data_num,:);
% W_knn = abs(C) + abs(C');
% W_full = W_knn;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%% KNN graph
graph_knn = TransductionModel(W_knn);
graph_knn.prior = ones(1, class_num);

%% RDP graph
W_rdp = RDP(W_full, K);
graph_rdp = TransductionModel(W_rdp);
graph_rdp.prior = ones(1, class_num);

for labels_i=1:length(Labels_num)
    labnum=Labels_num(labels_i);
    for test_j=1:replicates
        [labeled_ind, ~] = pickLabels(Y, labnum);
        
        %% Uncomment to let all labeled points at the right-most point
%         ll = [];
%         for i = 1:5
%             cIdx = find(Y==i);
%             xi = X(cIdx,1);
%             [~,idx]=sort(xi);
%             ll =[ll;cIdx(idx(end))];
%         end
%         labeled_ind = ll;
         
        %%%%% KNN + label_correction graph
        W_knn_lc = label_correction(W_knn, Y, labeled_ind);
        graph_knn_lc = TransductionModel(W_knn_lc);
        graph_knn_lc.prior = ones(1, class_num);
        
        %%%%% KNN + label prapagation
        [F_knn, error] = updateF(Y, labeled_ind, graph_knn.W, lambda);
        acc_knn(labels_i, test_j) = 100 - error * 100;
        
        %%%%% KNN + label correction + label propagation
        [F_knn_lc, error] = updateF(Y, labeled_ind, graph_knn_lc.W, lambda);
        acc_knn_noDiffusion(labels_i, test_j) = 100 - error * 100;
        
        %%%%% KNN + affinity diffusion + label propagation        
        [F_rdp, error] = updateF(Y, labeled_ind, graph_rdp.W, lambda);
        acc_knn_noLabel(labels_i, test_j) = 100 - error * 100;
        
        %% KNN + label correction + affinity diffusion + label propagation 
        [A_jdp, error_knn_withALL(labels_i, test_j), F_jdp] = JDP(W_full, K, Y, labeled_ind);
        graph_jd = TransductionModel(A_jdp);
        graph_jd.prior = ones(1, class_num);
        acc_knn_withALL(labels_i, test_j) = 100 - error_knn_withALL(labels_i, test_j) * 100;
        
        
        fprintf('========================labels/c=%d   KNN=%d   Round %d/%d ==========================\n', ...
            labnum/class_num, K, test_j, replicates);
        fprintf('KNN: %.2f, KNN_noDiffusion: %.2f, KNN_Nolabel: %.2f, KNN_withALL: %.2f\n', acc_knn(labels_i, test_j), ...
            acc_knn_noDiffusion(labels_i, test_j), acc_knn_noLabel(labels_i, test_j), ...
            acc_knn_withALL(labels_i, test_j));
        fprintf('==================================================================================\n')
        
        [~, Y_knn] = max(F_knn, [], 2);
        [~, Y_noDiffusion] = max(F_knn_lc, [], 2);
        [~, Y_noLabel] = max(F_rdp, [], 2);
        [~, Y_withALL] = max(F_jdp, [], 2);
        createfigure_fiveCircle(X, Y, Y_knn, Y_noDiffusion, Y_noLabel, Y_withALL, labeled_ind)
        
        
    end
    %%%%% Mean results of all replicates
    fprintf('####################Summay --- labels=%d   KNN=%d ####################\n', labnum, K);
    fprintf('KNN: %.2f, KNN_noDiffusion: %.2f, KNN_Nolabel: %.2f, KNN_withALL: %.2f\n', ...
        mean(acc_knn(labels_i,:)), mean(acc_knn_noDiffusion(labels_i,:)),mean(acc_knn_noLabel(labels_i,:)), ...
        mean(acc_knn_withALL(labels_i,:)));
    fprintf('####################End --- labels=%d   KNN=%d ####################\n\n', labnum, K);
end
