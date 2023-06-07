function [mu_theta, Cov_theta] = Fast_Bayesian_DP_LR(S_obs, Z_obs, DP_params, hyperparams)

% theta = Fast_Bayesian_DP_LR(S_obs, Z_obs, DP_params, hyperparams)
% 
% Implements Bayes-fixedS-fast

var_Z = hyperparams.var_Z;
C = hyperparams.C;
d = size(S_obs{1}, 1);
J = length(S_obs);
bound_Y = DP_params.bound_Y;
var_y = bound_Y/3;

Sigma_inv = inv(C);
Sigma_mean_post_theta = 0;

for j = 1:J    
    % find the nearest psd matrix and construct a pd matrix close to S_obs
    S0 = closest_psd(S_obs{j}); % + eye(d);
    S0 = (S0 + S0')/2;
    
    Sigma_inv = Sigma_inv + S0*((S0*var_y + eye(d)*var_Z)\S0);
    Sigma_mean_post_theta = Sigma_mean_post_theta + S0*((S0*var_y + eye(d)*var_Z)\Z_obs{j});
end

mu_theta = Sigma_inv\Sigma_mean_post_theta;
Cov_theta = eye(d)/Sigma_inv;