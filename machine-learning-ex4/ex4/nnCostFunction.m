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

y_vec = zeros(size(X,1),num_labels);
for i = 1:size(y,1)
   k = y(i);
   y_vec(i,k) = 1; %5000 x 10
end

% a1 = [ones(m, 1) X];
% a2 = sigmoid(a1*Theta1'); %5000 x 401 * 25x401 '
% a2 = [ones(m, 1) a2]; %5000 x 26
% a3 = sigmoid(a2*Theta2'); %5000 x 10
% %h_theta(x) = a^(3)
% 
% %non-regularized 5000x10 * 5000 x 10
% 
% J = (1/m).*sum(sum(-y_vec.*log(a3) - (1 - y_vec).*log((1-a3))));
% 
% %Regularized
% 
% temp1 = Theta1;
% temp1(:,1) = 0;
% temp2 = Theta2;
% temp2(:,1) = 0;
% 
% %regularized
% J = J + (lambda/(2*m)).*(sum(sum(temp1.^2)) + sum(sum(temp2.^2)));


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

d3 = zeros(1,num_labels);
delta2 = zeros(size(Theta2));
delta1 = zeros(size(Theta1));

temp1 = Theta1;
temp1(:,1) = 0;
temp2 = Theta2;
temp2(:,1) = 0;

for t = 1:m  
    a_1 = X(t,:)';
    a_1 = [1 ; a_1];
    z_2 = Theta1*a_1;
    a_2 = sigmoid(z_2); 
    a_2 = [1 ; a_2];
    z_3 = Theta2*a_2; 
    a_3 = sigmoid(z_3);
    
    for j = 1:num_labels
        %non-regularized
        J = J + (1/m).*sum(sum(-y_vec(t,j)*log(a_3(j)) - (1 - y_vec(t,j))*log((1-a_3(j)))));
    end
        
    d3 = a_3 - y_vec(t,:)';
    d2 = Theta2(:,2:end)'*d3.*sigmoidGradient(z_2);
    
    delta2 = delta2 + d3*a_2';
    delta1 = delta1 + d2*a_1';
    
end
%Regularized
J = J + (lambda/(2*m)).*(sum(sum(temp1.^2)) + sum(sum(temp2.^2)));
Theta2_grad = (1/m).*delta2 + (lambda/m)*temp2;
Theta1_grad = (1/m).*delta1 + (lambda/m)*temp1;


% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%



















% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];

end
