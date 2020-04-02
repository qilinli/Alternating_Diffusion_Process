function [W, W_knn] = adaptiveGaussian_singleX(data, x, K)

n = size(data, 1);
D = EuDist2(x, data);
[T, idx] = sort(D, 2);

W = zeros(1, n);
sigma = mean(T(2:K+1));
W = normpdf(D, 0, 0.35*sigma);

P = zeros(1, n);
P(1, idx(1:K+1)) = 1;

W_knn = W.*P;



