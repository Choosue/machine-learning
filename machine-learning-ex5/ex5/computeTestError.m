function [error_test] = computeTestError(X, y, Xtest, ytest, lambda)

[theta_train] = trainLinearReg(X, y, lambda);
[error_test grad_test] = linearRegCostFunction(Xtest, ytest, theta_train, 0);