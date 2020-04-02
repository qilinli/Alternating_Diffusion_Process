function two_moon_scatter(X, Y, labeled_idx)

scatter(X(:,1), X(:,2), 5, Y, 'filled')
hold on
scatter(X(labeled_idx,1), X(labeled_idx,2), 100, 'g', 'filled', 'h')