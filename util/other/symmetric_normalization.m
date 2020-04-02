function W = symmetric_normalization(W)

d = sum(W, 2);
D = diag(d + eps);
W = D^(-1/2)*W*D^(-1/2);      %%% Symmetric normalization is better
