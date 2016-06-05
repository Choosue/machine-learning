function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

% Using Theta1 and Theta2 to compute output units
A1 = [ones(m, 1) X];
A2 = sigmoid(A1 * Theta1');
A2 = [ones(m, 1) A2];
A3 = sigmoid(A2 * Theta2');

% Expand y into y_matrix
% y_matrix = eye(num_labels)(y,:); % This is a really tricky method
y_matrix = zeros(m, num_labels);
for i = 1:num_labels
	y_matrix(:, i) = (y == i);
end

% Code to compute unregularized J, attention the dot multiplication between y_matrix and log
J = sum(sum(((-1 * y_matrix) .* log(A3) - (1 - y_matrix) .* log(1 - A3)) / m)); % Unregularized

% Calculate regularization term
regularization_term = (lambda / (2 * m)) * (sum(sum(Theta1(:, 2:size(Theta1, 2)) .^2)) + sum(sum(Theta2(:, 2:size(Theta2, 2)) .^2)));

% Add the regularization term to the cost
J = J + regularization_term;

% Back propagation
Delta1 = zeros(size(Theta1));
Delta2 = zeros(size(Theta2));
for t = 1:m
	% Step 1: feed forward
	a1 = X(t, :)(:);                                                                     % Get the t-th row of X, and turn it into a column vector (400 * 1)
	a1 = [1 ; a1];                                                                      % Add a1_0 (401 * 1)
	z2 = Theta1 * a1;                                                                % (25 * 401) * (401 * 1) = (25 * 1)
	% z2
	% sigmoidGradient(z2)
	a2 = sigmoid(z2);                                                                % (25 * 1)
	a2 = [1 ; a2];                                                                      % Add a2_0 (26 * 1)
	z3 = Theta2 * a2;                                                                % (10 * 26) * (26 * 1) = (10 * 1)
	a3 = sigmoid(z3);                                                                % (10 * 1) and a3 is the hypothesis h of x
	% Step 2: calculate delta3
	delta3 = a3 - y_matrix(t, :)(:);                                                  % (10 * 1) - (10 * 1) = (10 * 1)
	% Step 3: calculate delta2 (we will ignore delta1)
	delta2 = (Theta2(:, 2:end))' * delta3 .* sigmoidGradient(z2);  % (25 * 10) * (10 * 1) .* (25 * 1) = (25 * 1), for Theta2 we do not count the bias term
	% Step 4
	Delta1 = Delta1 + delta2 * a1';                                              % (25 * 1) * (1 * 401) = (25 * 401), the same size as Theta1
	Delta2 = Delta2 + delta3 * a2';                                              % (10 * 1) * (1 * 26) = (10 * 26), the same size as Theta2
end
% Step 5
Theta1_grad = (1 / m) * Delta1;                                                 % (25 * 401)
Theta2_grad = (1 / m) * Delta2;                                                 % (10 * 26)

% Compute regularization term
Temp1 = Theta1;
Temp1(:, 1) = 0;
Temp2 = Theta2;
Temp2(:, 1) = 0;
Theta1_grad = Theta1_grad + (lambda / m) * Temp1;
Theta2_grad = Theta2_grad + (lambda / m) * Temp2;




% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
