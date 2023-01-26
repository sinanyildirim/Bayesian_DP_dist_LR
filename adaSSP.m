function [theta] = adaSSP(S, Z, DP_params)

% [theta] = adaSSP(S, Z, DP_params)
%
% Implements adaSSP

bound_X = DP_params.bound_X;
bound_Y = DP_params.bound_Y;
epsilon = DP_params.epsilon;
delta = DP_params.delta;
p = DP_params.p;

J = length(S);
d = size(S{1}, 1);

% calculate the noise std's
sigma0 = analytic_Gaussian_mech(epsilon/3, delta/3);
std_Z = sigma0*bound_X*bound_Y;
std_S = sigma0*bound_X^2;

S_total = 0;
Z_total = 0;

lambda = zeros(1, J);
for j = 1:J
    lambda_min = min(eig(S{j}));
    
    lambda_release = max(lambda_min + sqrt(log(6/delta))/(epsilon/3)*(bound_X^2)*randn - log(6/delta)/(epsilon/3)*(bound_X^2), 0);
    lambda(j) = max(0, sqrt(d*log(6/delta)*log(2*d^2/p))*bound_X^2/(epsilon/3) - lambda_release);

    upper_temp_Z = triu(randn(d), 1);
    lower_temp_Z = upper_temp_Z';
    noise_S = (upper_temp_Z + lower_temp_Z + diag(randn(1, d), 0))*std_S;
    noise_Z = std_Z*randn(d, 1);
    
    % observations
    S_total = S_total + S{j} + noise_S;
    Z_total = Z_total + Z{j} + noise_Z;
end

theta = (S_total + sum(lambda)*eye(d))\Z_total;