function [error_train error_val] = randomLearningCurve(X, y, Xval, yval, MaxIter, lambda)

m = size(X, 1);

error_train = zeros(m, 1);
error_val   = zeros(m, 1);

for iter = 1:MaxIter	
	for i = 1:m
		% Randomly select i examples from X to form X_rand
		[X_rand, y_rand] = randomExample(X, y, i);

		% Randomly select i examples from Xval to form X_rand_val
		[X_rand_val, y_rand_val] = randomExample(Xval, yval, i);
		
		% And learn a theta_train from X_rand using lambda
		theta_train = trainLinearReg(X_rand, y_rand, lambda);

		% compute error_train(i) = error_train(i) + computed_error_train;
		error_train(i) = error_train(i) + linearRegCostFunction(X_rand, y_rand, theta_train, 0);
		% compute error_val(i) = error_val(i) + computed_error_val;
		error_val(i) = error_val(i) + linearRegCostFunction(X_rand_val, y_rand_val, theta_train, 0);
	end
end

error_train = error_train / MaxIter;
error_val   = error_val   / MaxIter;