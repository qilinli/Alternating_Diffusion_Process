function two_moon_scatter(X, Y, labeled_idx)

% scatter(X(Y==1,1), X(Y==1,2), [], 'r', 'filled')
% hold on
% scatter(X(Y==2,1), X(Y==2,2), [], 'b', 'filled')
% scatter(X(labeled_idx,1), X(labeled_idx,2), 100, 'g', 'filled', 'h')


scatter(X(:,1), X(:,2), 5, Y, 'filled')
hold on
scatter(X(labeled_idx,1), X(labeled_idx,2), 50, 'g', 'filled', 'h')