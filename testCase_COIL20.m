clear;
clc;

addpath('util/GraphConstruct/');
addpath('util/other/');
addpath('util/L1graph/');
warning('off', 'MATLAB:singularMatrix');
warning('off', 'MATLAB:nearlySingularMatrix');

load data/COIL20Resized_N1440_D1024.mat;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%% parameters %%%%%%%%%%%%%%%%%%%%%%%%%%
K                       = 10;       % k-nearst neighbor
alpha                   = 20;       % sparsity regularizer in lasso
lambda                  = 1e-4;     % for numeric stablity in graph laplacian inverse;
replicates              = 5;       % # of replicates experiment
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

class_num = length(unique(Y));
data_num = length(Y);
Labels_num = class_num * [1:2:11];

%%%%%%%%%% Choice of initial W %%%%%%%%%%%%%%%%%
%  [W_full,W_knn] = adaptiveGaussian(X, 27);

C = admmOutlier_mat_func(X', 0, 20);
C = C(1:data_num,:);
W_knn = abs(C) + abs(C');
W_full = W_knn;
 
% W_knn = weight_b_g_ag(X, knn, sigma_k, weight_type);
% W_full = weight_b_g_ag(X, data_num, sigma_k, weight_type);
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
        
        %% Semi-supervised diffusion (SRD)
        W_srd = SRD(W_full, K, Y, labeled_ind);
        graph_srd = TransductionModel(W_srd);
        graph_srd.prior = ones(1, class_num);
        
        %% Joint diffusion process (JDP)
        [W_jd, error_jdp(labels_i, test_j), ~] = JDP(W_full, K, Y, labeled_ind);
        graph_jd = TransductionModel(W_jd);
        graph_jd.prior = ones(1, class_num);
        acc_jdp(labels_i, test_j) = 100 - error_jdp(labels_i, test_j) * 100;
        
        %% Alternating diffusion process (ADP)
        [W_jd1, error_adp(labels_i, test_j),~] = ADP(W_full, K, Y, labeled_ind);
        graph_jd1 = TransductionModel(W_jd1);
        graph_jd1.prior = ones(1, class_num);
        acc_adp(labels_i, test_j) = 100 - error_adp(labels_i, test_j) * 100;
        
        %%%%% with knn graph
        [~, error_knn] = updateF(Y, labeled_ind, graph_knn.W, lambda);
        acc_knn(labels_i, test_j) = 100 - error_knn * 100;
        
        %%%%% with rdp graph
        [~, error_rdp] = updateF(Y, labeled_ind, graph_rdp.W, lambda);
        acc_rdp(labels_i, test_j) = 100 - error_rdp * 100;
        
        %%%%% with srd graph
        [~, error_srd] = updateF(Y, labeled_ind, graph_srd.W, lambda);
        acc_srd(labels_i, test_j) = 100 - error_srd * 100;
        
        fprintf('========================labels/c=%d   KNN=%d   Round %d/%d ==========================\n', ...
            labnum/class_num, K, test_j, replicates);
        fprintf('KNN: %.2f, RDP: %.2f, SRD: %.2f, JDP: %.2f, ADP: %.2f \n', acc_knn(labels_i, test_j), ...
            acc_rdp(labels_i, test_j), acc_srd(labels_i, test_j), ...
            acc_jdp(labels_i, test_j), acc_adp(labels_i, test_j));
        fprintf('==================================================================================\n')
    end
    %%%%% Mean results of all replicates
    fprintf('####################Summay --- labels=%d   KNN=%d ####################\n', labnum, K);
    fprintf('KNN-Mean: %.2f, RDP-Mean: %.2f, SRD-Mean: %.2f, JDP-Mean: %.2f, ADP-Mean: %.2f \n', ...
        mean(acc_knn(labels_i,:)), mean(acc_rdp(labels_i,:)),mean(acc_srd(labels_i,:)), ...
        mean(acc_jdp(labels_i,:)), mean(acc_adp(labels_i,:)));
    fprintf('####################End --- labels=%d   KNN=%d ####################\n\n', labnum, K);
end

Y = [mean(acc_knn,2) mean(acc_rdp,2) mean(acc_srd,2) mean(acc_jdp,2) mean(acc_adp,2)];
X = Labels_num/class_num;
STD = [std(acc_knn,[],2) std(acc_rdp,[],2) std(acc_srd,[],2) std(acc_jdp,[],2) std(acc_adp,[],2)];
createfigure(X, Y, 'COIL20');