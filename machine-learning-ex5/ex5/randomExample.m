function [X_rand, y_rand] = randomExample(X, y, i)

X_rand = zeros(i, size(X, 2));
y_rand = zeros(i, 1);
sample = randperm(size(X, 1), i);

for index = 1:i
	X_rand(index, :) = X(sample(index), :);
	y_rand(index) = y(sample(index));
end