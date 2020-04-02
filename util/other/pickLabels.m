function [labeled_ind, unlabeled_ind] = pickLabels(label, L, mode)

%%% Random pick label points %%%
%%% Input:
%%%       label: 1xN ground truth
%%%       L: number of labeled points
%%%       mode(optional): 'imbalance' or 'balance'
%%% Output:
%%%        labeled_ind: index of labled points
%%%        unlabeled_ind: index of unlabeled points
%%%
%%% By QILIN LI (li.qilin@postgrad.curtin.edu.au) 
%%% Last updates 11/07/2018

if ~exist('mode','var'), mode = 'balance'; end

%% label has to be 1xN.
[m, n] = size(label);
if m > n, label = label'; end

N = length(label);
nClass = length(unique(label));

%%% Check if L > nClass.
if L < nClass
    newL = nClass;
    warning('Labels cannot be smaller than the number of class,\n the number of labeled points is changed from %d to %d', L, newL);  
    L = newL;
end

%%% Check if L is dividable by nCLass.
if strcmp(mode, 'balance') && mod(L, nClass) ~= 0
    newL = floor(L/nClass) * nClass; 
    warning('Labels cannot be balanced as mod(%d, %d) ~= 0,\n the number of labeled points is changed from %d to %d', L, nClass, L, newL);
    L = newL;
end
    
labeled_ind = [];
%%% First make sure each class has at least a labeled point.
for i = 1:nClass
    ind_class = find(label==i);
    rand_ind = randperm(length(ind_class));
    ind_class = ind_class(rand_ind);
    labeled_ind = [labeled_ind ind_class(1)];
end

%%% For L > nClass, divide into two modes.
if L > nClass
    if strcmp(mode, 'balance')
        LperClass = L/nClass - 1;
        for i = 1:nClass
            ind_class = find(label==i);
            ind_class = setdiff(ind_class, labeled_ind);
            rand_ind = randperm(length(ind_class));
            ind_class = ind_class(rand_ind);
            labeled_ind = [labeled_ind ind_class(1:LperClass)];
        end
    else
        unlabeled_ind = setdiff(1:N, labeled_ind);
        rand_ind=randperm(length(unlabeled_ind));
        labeled_ind=[labeled_ind unlabeled_ind(rand_ind(1:L-nClass))];
    end
end

unlabeled_ind = setdiff(1:N, labeled_ind);

    
        


