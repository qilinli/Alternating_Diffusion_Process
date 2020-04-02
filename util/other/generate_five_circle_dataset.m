function [X, Y] = generate_five_circle_dataset

X = [];
Y = [];

x_center = [5, 5, 3.5, 6.5, 5];
y_center = [5, 5, 6, 6, 4];

r = [3.5, 3, 1, 1, 1];
th = 0 : pi/50 : 2*pi;
th = th(1:100);
for i = 1:length(r)
    points = [r(i)*cos(th)'+x_center(i), r(i)*sin(th)'+y_center(i)];
    X = [X; points];
    Y = [Y; ones(length(points),1)*i];
end