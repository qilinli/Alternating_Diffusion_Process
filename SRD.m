%% Li, Qilin, Wanquan Liu, and Ling Li. "Self-reinforced diffusion for graph-based semi-supervised learning." Pattern Recognition Letters 125 (2019): 439-445.
function W_srd = SRD(W, K, gnd, labeled_ind)

%% Pre-processing of affinity matrix W
d = sum(W, 2);
D = diag(d + eps);
D_normalized = D ./ max(max(D));
n = size(W, 1);          %%% number of data points
W = W - diag(diag(W)) + D;   %%% use node degree as self-affinity

%% Normalization  %%%%%%%%%%%%%%%%%%
%  W = W ./ repmat(sum(W, 2)+eps, 1, n);

d = sum(W,2);
D = diag(d + eps);
W = D^(-1/2)*W*D^(-1/2);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%% Sparse of affinity matrix W
W_knn = knnSparse(W, K);

%%% Add true neighbors if they have same label
Y = zeros(n, n);
for i = 1 : length(labeled_ind)
    for j = i+1 : length(labeled_ind)
        if gnd(labeled_ind(i)) == gnd(labeled_ind(j)) && W_knn(labeled_ind(i), labeled_ind(j)) == 0  
            %fprintf("Supervised neighbors found!%d and %d.\n", labeled_ind(i), labeled_ind(j));
            W_knn(labeled_ind(i), labeled_ind(j)) = W(labeled_ind(i), labeled_ind(j));
            W_knn(labeled_ind(j), labeled_ind(i)) = W(labeled_ind(j), labeled_ind(i));
                       
            Y(labeled_ind(i), labeled_ind(j)) = W(labeled_ind(i), labeled_ind(j));
            Y(labeled_ind(j), labeled_ind(i)) = W(labeled_ind(j), labeled_ind(i));

        elseif gnd(labeled_ind(i)) ~= gnd(labeled_ind(j)) && W_knn(labeled_ind(i), labeled_ind(j)) ~= 0
            %fprintf("Knn removed by supervision!%d and %d.\n", labeled_ind(i), labeled_ind(j));
            W_knn(labeled_ind(i), labeled_ind(j)) = 0;
            W_knn(labeled_ind(j), labeled_ind(i)) = 0;
        end
    end
end

%%% Main iteration of Semi-supervised diffusion (SSD)
maxIter = 30;
epsilon = 1e-2;
W_srd = W_knn;

mu = 0.18;
alpha = 1 / (1+mu);
for t = 1:maxIter
    W_srd_new = alpha*W_knn*W_srd*W_knn' + (1-alpha)*(D_normalized + Y);
    error = norm(W_srd_new-W_srd, 'fro');
    if error < epsilon, break; end
    W_srd = W_srd_new;
end
fprintf("SRD converged at iteration %d.\n", t);

for i = 1 : length(labeled_ind)
    for j = i+1 : length(labeled_ind) 
        if gnd(labeled_ind(i)) ~= gnd(labeled_ind(j)) && W_srd(labeled_ind(i), labeled_ind(j)) ~= 0
            %fprintf("After diffusion, Knn removed by supervision!%d and %d.\n", labeled_ind(i), labeled_ind(j));
            W_srd(labeled_ind(i), labeled_ind(j)) = 0;
            W_srd(labeled_ind(j), labeled_ind(i)) = 0;
        end
    end
end
W_srd = (W_srd+W_srd')/2;


