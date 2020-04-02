function [W_full,W_knn] = D2W(D, K, weight_type)

n = size(D, 1);
[T, idx] = sort(D, 2);

P = zeros(n, n);
for i = 1:n
    P(i, idx(i,1:K+1)) = 1;
end

switch weight_type
    case 2
        sigma = mean(T(:, K+1));
        W_full = exp(-D.^2/(2*sigma^2));
        W_knn = W_full.*P;
    case 3
        W = zeros(n,n);
        for i = 1:n
            for j = 1:n
                sigma = mean([T(i,2:K+1), T(j,2:K+1)]);
                W(i,j)=normpdf(D(i,j), 0, 0.35*sigma);
            end
        end
        W_full = W;
        W_knn = W.*P;
end

W_knn = (W_knn+W_knn')/2;
        

