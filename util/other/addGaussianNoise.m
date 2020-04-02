function x=addGaussianNoise(x, p, r)

[n,d]=size(x);
N=ceil(n*p);
idx=ceil(rand(1,N)*n);
if (~isempty(idx))
    for i=1:length(idx)
        x(idx(i),:)=x(idx(i),:)+normrnd(0,r*norm(x(idx(i),:)),d,1)';
    end
end
return