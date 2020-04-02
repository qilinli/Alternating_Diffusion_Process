%% Bai, Song, et al. "Regularized diffusion process for visual retrieval." Thirty-First AAAI Conference on Artificial Intelligence. 2017.
function [WW] = RDP(W, K)

d = sum(W, 2);
D = diag(d + eps);
W = D^(-0.5)*W*D^(-0.5);

S = knnSparse(W, K);

WW = S;
mu = 0.18;
epsilon = 1e-2;
alpha = 1/(1+mu);
maxIter = 30;
for t = 1:maxIter
    temp = alpha*S*WW*S' + (1-alpha)*eye(length(WW));
    error = norm(temp-WW,'fro');
    if error < epsilon, break; end  
    WW = temp;   
end
fprintf("RDP onverged at iteration %d.\n", t);
