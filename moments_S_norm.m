function [mu_S, Cov_S] = moments_S_norm(C, theta, var_y)

% [mu_S, Cov_S] = moments_S_norm(C, theta, var_y)
% 
% Calculates the mean and covariance of the prior distribution of the 
% vectorised form of the sufficient statistics of MCMC-B&S

d = size(C, 1);
% theta = zeros(d, 1);
[~, S11] = fourth_central_moment_norm(C);

Theta_mtx = kron(eye(d), theta);
S12 = S11*Theta_mtx;
S13 = S12*theta;
S13_as_mtx = reshape(S13, d, d);
S22 = var_y*C + Theta_mtx'*S12;
S23 = S13_as_mtx*theta + 2*var_y*C*theta;
S33 = theta'*S13_as_mtx*theta + 2*var_y^2 + 4*var_y*theta'*C*theta;

Cov_S = [[S11 S12 S13]; [S12' S22 S23]; [S13' S23' S33]];
Cov_S = (Cov_S + Cov_S')/2;
mu_S = [C(:); C*theta; (var_y + theta'*C*theta)];