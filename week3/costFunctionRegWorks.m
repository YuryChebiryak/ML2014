function [J, grad] = costFunctionRegWorks(theta, X, y, lambda)
fprintf(" in costFunctionRegWorks \n ");
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta


h = sigmoid(X * theta);
for i = 1 : m
	J = J + ( (1 / m) * (-y(i) * log(h(i)) - ( 1 - y(i)) * log(1 - h(i))));
endfor;
fprintf('Cost at initial theta (zeros): %f\n', J);

% regularization
for j = 2:size(theta)
	J = J + (lambda / ( 2 * m )) * (theta(j) ^ 2)
endfor;
fprintf('Cost after reg: %f\n', J);

n = size(X, 2);
for j = 1:size(theta)
	grad(j,1) = (1 / m) * sum(  X(:,[j])' * (h-y) );
	if (j > 1) %regularization: add (lambda / m) * theta_j
		grad(j,1) = grad(j,1) + (lambda / m) * theta(j);
	endif
endfor;

fprintf('Gradient after reg: \n');
fprintf(' %f \n', grad);



% =============================================================

end
