function createfigure_fiveCircle(X, Y, Y_knn, Y_noDiffusion, Y_noLabel, Y_jd, labeled_idx)

unlabeled_idx = setdiff(1:length(Y), labeled_idx);
u = unlabeled_idx;

figure1 = figure('Color',[1 1 1]);

% Create subplot for original dataset
subplot1 = subplot(1,5,1,'Parent',figure1);
%xlabel('(1)','FontWeight','bold');
box(subplot1,'on');
set(subplot1,'FontSize',8);
hold(subplot1,'on');

% Create scatter for labeled points
scatter(X(u, 1),X(u, 2),5,'black','Parent',subplot1,'MarkerFaceColor','flat',...
    'MarkerEdgeColor','none');
subScatter(X(labeled_idx,:), Y(labeled_idx,:), 50, subplot1);

% Create subplot for knn
subplot2 = subplot(1,5,2,'Parent',figure1);
%xlabel('(2)','FontWeight','bold');
box(subplot2,'on');
set(subplot2,'FontSize',8);
hold(subplot2,'on');
subScatter(X, Y_knn, 5, subplot2);
title('KNN')

% Create subplot
subplot3 = subplot(1,5,3,'Parent',figure1);
%xlabel('(3)','FontWeight','bold');
box(subplot3,'on');
set(subplot3,'FontSize',8);
hold(subplot3,'on');
subScatter(X, Y_noDiffusion, 5, subplot3);
title({'KNN +', 'label correction'})

% Create subplot
subplot4 = subplot(1,5,4,'Parent',figure1);
%xlabel('(4)','FontWeight','bold');
box(subplot4,'on');
set(subplot4,'FontSize',8);
hold(subplot4,'on');
subScatter(X, Y_noLabel, 5, subplot4);
title({'KNN +', 'affinitu diffusion'})

% Create subplot
subplot5 = subplot(1,5,5,'Parent',figure1);
%xlabel('(5)','FontWeight','bold');
box(subplot5,'on');
set(subplot5,'FontSize',8);
hold(subplot5,'on');
subScatter(X, Y_jd, 5, subplot5);
title({'KNN + label correction +', 'affinity diffusion'})


function subScatter(X, Y, mkr_size, subplot)
    
scatter(X(Y==1,1),X(Y==1,2),mkr_size,'r','Parent',subplot,'MarkerFaceColor','flat',...
    'MarkerEdgeColor','none');
scatter(X(Y==2,1),X(Y==2,2),mkr_size,[0.4660 0.6740 0.1880],'Parent',subplot,'MarkerFaceColor','flat',...
    'MarkerEdgeColor','none');
scatter(X(Y==3,1),X(Y==3,2),mkr_size,[0 0.4470 0.7410],'Parent',subplot,'MarkerFaceColor','flat',...
    'MarkerEdgeColor','none');
scatter(X(Y==4,1),X(Y==4,2),mkr_size,[0.9290 0.6940 0.1250],'Parent',subplot,'MarkerFaceColor','flat',...
    'MarkerEdgeColor','none');
scatter(X(Y==5,1),X(Y==5,2),mkr_size,'m','Parent',subplot,'MarkerFaceColor','flat',...
    'MarkerEdgeColor','none');

