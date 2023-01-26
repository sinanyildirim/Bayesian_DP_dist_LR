function [outputs] = MCMC_DP_LR_BS(SS_obs, init_vars, hyperparams, N_node, K)

% [outputs] = MCMC_DP_LR_BS(SS_obs, init_vars, hyperparams, N_node, K)
% 
% Implements MCMC-BS

J = length(N_node);

Lambda = hyperparams.Lambda;
kappa = hyperparams.kappa;
var_SS = hyperparams.var_SS;

theta = init_vars.theta;
d = length(theta);
var_y = init_vars.var_y;
SS = init_vars.SS;

mu_0 = hyperparams.m;
lambda_0 = hyperparams.C;
a_0 = hyperparams.a;
b_0 = hyperparams.b;

D = d^2 + d + 1; %noise covariance dimension

% construct the vector of indices of unique elements in S
ind_S_unique = zeros(1, d^2);
d_S = 0;
for i = 1:d
    c_vec = d_S+(1:(d-i+1));
    ind_S_unique(c_vec) = (i-1)*d + (i:d);
    d_S = d_S + (d-i+1);
end
ind_S_unique = ind_S_unique(1:d_S);
ind_SS_unique = [ind_S_unique d^2+(1:d+1)];
d_SS = d_S + d+1;

% get the unique parts of the observed statistics
SS_obs_unique = SS_obs(ind_SS_unique, :);
% arrange the noise covariance based on the unique parts
Cov_DP_S = var_SS*eye(d_SS);

% Covariance matrix (S) of Xs.
XXT_total  = reshape(sum(SS(1:d^2, :), 2), d, d);
XXT_total = (XXT_total + XXT_total)/2;
XXT_total = closest_psd(XXT_total) + eye(d);

%% Make inference with MCMC
theta_vec = zeros(d, K);
var_y_vec = zeros(1, K);

for k = 1:K
    
    % project to space of psd matrices and make pd
    Sigma_X = iwishrnd(XXT_total + Lambda, kappa + sum(N_node)); 
    Sigma_X = (Sigma_X + Sigma_X')/2;

    % Update S
    [mu_SS, Cov_SS] = moments_S_norm(Sigma_X, theta, var_y);
    
    % Get the mean and covariance of unique elements    
    mu_SS_unique = mu_SS(ind_SS_unique);
    Cov_SS_unique = Cov_SS(ind_SS_unique, ind_SS_unique);    

    SS = zeros(D, J);
    for j=1:J
        Cov_SSj = Cov_SS_unique*N_node(j);
        
        %%%%% update S
        Cov_SS_post = Cov_DP_S - var_SS*Cov_DP_S/(Cov_DP_S + Cov_SSj);
        Cov_SS_post = (Cov_SS_post + Cov_SS_post')/2; % to ensure symmetry        
        mu_SS_post = Cov_SS_post*(Cov_SS_unique\mu_SS_unique + SS_obs_unique(:, j)/var_SS);

        SSj = N_node(j)*mvnrnd(mu_SS_post/N_node(j), Cov_SS_post/N_node(j)^2)'; 
        
        % convert the XTX part of Sj to a vectorised conv matrix
        Sj_unique_S = SSj(1:d_S);
        Sj_mtx = zeros(d);
        Sj_mtx(ind_S_unique) = Sj_unique_S;
        Sj_mtx = Sj_mtx + Sj_mtx' - diag(diag(Sj_mtx));

        SS(:, j) = [Sj_mtx(:); SSj(d_S+(1:d+1))];
    end

    %%%%% calculate params of the normal-inverse wishard posterior 
    % for theta, sigma
    SS_sum = sum(SS, 2);
    XXT_total = reshape(SS_sum(1:d^2), d, d);
    % project to space of psd matrices and make pd
    XXT_total = closest_psd(XXT_total) + eye(d);

    Xy_total = SS_sum(d^2 + (1:d));
    yy_total = SS_sum(end);

    lambda_n = XXT_total + lambda_0;
    lambda_n = (lambda_n + lambda_n')/2;

    mu_n = lambda_n\(Xy_total + lambda_0*mu_0);

    a_n = a_0 + sum(N_node)/2;
    b_n = b_0 + 0.5*(yy_total + mu_0'*lambda_0*mu_0 - mu_n'*lambda_n*mu_n);
    % Ensure it is at least b_0
    b_n = max(b_n, b_0);

    % now sample from this posterior distribution
    var_y = 1/gamrnd(a_n, 1/b_n);
    C_post = var_y*(lambda_n\eye(d));
    C_post = (C_post+C_post')/2;
    theta = mvnrnd(mu_n, C_post)';

    % Store the variables
    theta_vec(:, k) = theta;
    var_y_vec(k) = var_y;

end

%% Store outputs
outputs.theta_vec = theta_vec;
outputs.var_y_vec = var_y_vec;