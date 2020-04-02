function [W,W_knn]=adaptiveGaussian(data,K)

n = size(data, 1);
D = EuDist2(data);
D = D - diag(diag(D));   %%% Zero distance to itself
[T, idx] = sort(D, 2);

W = zeros(n,n);
for i = 1:n
    for j = 1:n
        sigma = mean([T(i,2:K+1), T(j,2:K+1)]);
        W(i,j) = normpdf(D(i,j), 0, 0.35*sigma);
    end
end
W = (W+W')/2;

P = zeros(n, n);
for i = 1:n
    P(i, idx(i,1:K+1)) = 1;
end

W_knn = W.*P;
W_knn = (W_knn+W_knn')/2;


