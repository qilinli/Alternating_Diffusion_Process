function [idx_tr, idx_te] = dataset_split(Y, test_ratio)

n_sample = length(Y);
n_class = length(unique(Y));
n_sample_te = floor(n_sample * test_ratio);
n_per_class = floor(n_sample_te / n_class);

idx_te = [];
for i = 1:n_class
    ind_class = find(Y==i);
    ind_class = setdiff(ind_class, idx_te);
    rand_ind = randperm(length(ind_class));
    ind_class = ind_class(rand_ind);
    idx_te = [idx_te ind_class(1:n_per_class)];
end

idx_tr = setdiff(1:n_sample, idx_te);