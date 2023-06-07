function [S, decision] = MH_S_update(S, N, S_obs, Z_obs, theta, Sigma, var_y, var_S, var_Z, a)

% [S, decision] = MH_S_update(S, N, S_obs, Z_obs, theta, Sigma, var_y, var_S, var_Z, a)
% 
% MH update for S

% prepare the indices of the upper diagonal
d = length(theta);
upper_tria_ind = zeros(d*(d+1)/2, 1);
c = 0;
for i = 1:d
    upper_tria_ind((c+1):(c+i)) = (i-1)*d+(1:i)';
    c = c + i;
end

%% calculate the log posterior for the initial S
% cov_Z, |cov_Z|, |S|, p(Z|S,...), p(S_prime|S), P(S|Sigma)
cov_Z = S*var_y + eye(d)*var_Z;
log_det_cov_Z = 2*sum(log(diag(cholcov(cov_Z))));
log_det_S = 2*sum(log(diag(cholcov(S))));
log_Z_S = -0.5*(log_det_cov_Z + (Z_obs - S*theta)'*(cov_Z\(Z_obs - S*theta)));
log_S_g_Sigma = log_det_S*((N-d-1)/2) - trace(Sigma\S)/2 ;
log_S_noise = -0.5*sum((S_obs(upper_tria_ind) - S(upper_tria_ind)).^2)/var_S;
log_pi = log_Z_S + log_S_noise + log_S_g_Sigma;

%% Proposal
S_prop = wishrnd(S/a, a);

%% calculate the acceptance ratio
% cov_Z_prop, |cov_Z_prop|, |S_prop|, p(Z|S,...), p(S_prime|S), P(S|Sigma)
cov_Z_prop = S_prop*var_y + eye(d)*var_Z;
log_det_cov_Z_prop = 2*sum(log(diag(cholcov(cov_Z_prop))));
log_det_S_prop = 2*sum(log(diag(cholcov(S_prop))));
log_Z_S_prop = -0.5*(log_det_cov_Z_prop + (Z_obs - S_prop*theta)'*(cov_Z_prop\(Z_obs - S_prop*theta)));
log_S_prop_noise = -0.5*sum((S_obs(upper_tria_ind) - S_prop(upper_tria_ind)).^2)/var_S;
log_S_prop_g_Sigma = log_det_S_prop*((N-d-1)/2) - trace(Sigma\S_prop)/2;
log_pi_prop = log_Z_S_prop + log_S_prop_noise + log_S_prop_g_Sigma;

%% log proposal
log_prop = (log_det_S - log_det_S_prop)*(a-(d+1)/2) + (trace(S\S_prop) ...
    - trace(S_prop\S))*(a/2);

%% calculate the log-acceptance rate
log_r = log_pi_prop - log_pi + log_prop;
decision = rand < exp(log_r);

if decision == 1
    S = S_prop;
end