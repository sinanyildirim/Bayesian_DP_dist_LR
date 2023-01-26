function [outputs] = MCMC_DP_LR(S_obs, Z_obs, init_vars, hyperparams, prop_params, N_node, K, update_S)

% [outputs] = MCMC_DP_LR(S_obs, Z_obs, init_vars, hyperparams, prop_params, N_node, K, update_S)
% 
% Implements MCMC-normalX and MCMC-fixedS

% hyperparameters
m = hyperparams.m;
C = hyperparams.C;
a = hyperparams.a;
b = hyperparams.b;
var_S = hyperparams.var_S;
var_Z = hyperparams.var_Z;
Lambda = hyperparams.Lambda;
kappa = hyperparams.kappa;

% proposal parameters
sigma_q_y = prop_params.sigma_q_y;
a_MH = prop_params.a_MH;

J = length(N_node);
d = length(m);

% initial variables
theta = init_vars.theta;
var_y = init_vars.var_y;
S = init_vars.S;
S_total = zeros(d);
for j = 1:J
    S_total = S_total + S{j};
end

% initialize the arrays
theta_vec = zeros(d, K);
var_y_vec = zeros(1, K);

if update_S == 1
    S_vec = repmat({zeros(d, d, K)}, 1, J);
    Sigma_vec = zeros(d, d ,K);
    decision_rate_vecs = zeros(1, K);

    update_a_burn_in = K/10;
    S_acc = zeros(1, J);
    a_MH_vec = zeros(K, J);
    alpha_ast = 0.1;
end

% Calcualte the inverses needed in the iterations
I_d = eye(d);
C_inv = I_d/C;

for i = 1:K
    gamma_s = i^(-0.6);
    gamma_a = i^(-0.6);

    % update Sigma %%
    if update_S == 1
        Sigma = iwishrnd(S_total + Lambda, kappa + sum(N_node));
        decision_vec = zeros(1, J);
        S_total = zeros(d);
    end

    % update Ss
    Sigma_inv = 0;
    Sigma_mean_post_theta = 0;

    for j = 1:J
        if update_S == 1
            % wishart proposal
            [S{j}, decision] = MH_S_update(S{j}, N_node(j), S_obs{j}, ...
                Z_obs{j}, theta, Sigma, var_y, var_S, var_Z, a_MH(j));

            % update a_MH
            S_acc(j) = (1 - gamma_s)*S_acc(j) + gamma_s*decision;
            if i > update_a_burn_in
                a_MH(j) = exp(log(a_MH(j)) - gamma_a*(S_acc(j) - alpha_ast));
            end
            decision_vec(j) = decision;
            
            S_total = S_total + S{j};
        end
        Sigma_inv = Sigma_inv + S{j}'*((S{j}*var_y + eye(d)*var_Z)\S{j});
        Sigma_mean_post_theta = Sigma_mean_post_theta ...
            + S{j}'*((S{j}*var_y + eye(d)*var_Z)\Z_obs{j});
    end

    % update theta
    Sigma_inv = Sigma_inv + C_inv;
    Sigma_post_theta = I_d/Sigma_inv;
    Sigma_post_theta = (Sigma_post_theta + Sigma_post_theta')/2;
    mean_post_theta = Sigma_post_theta*(Sigma_mean_post_theta + C_inv*m);
    theta = mvnrnd(mean_post_theta, Sigma_post_theta)';

    % update var_y
    var_y_prop = var_y + sigma_q_y*randn;

    if var_y_prop > 0
        % log-likelihood
        log_Z_S = 0;
        log_Z_S_prop = 0;
        for j = 1:J
            cov_Z = S{j}*var_y + eye(d)*var_Z;
            cov_Z_prop = S{j}*var_y_prop + eye(d)*var_Z;

            log_det_cov_Z = 2*sum(log(diag(cholcov(cov_Z))));
            log_det_cov_Z_prop = 2*sum(log(diag(cholcov(cov_Z_prop))));

            u = Z_obs{j} - S{j}*theta;
            log_Z_S = log_Z_S - 0.5*(log_det_cov_Z + u'*(cov_Z\u));
            log_Z_S_prop = log_Z_S_prop - 0.5*(log_det_cov_Z_prop + u'*(cov_Z_prop\u));
        end

        % prior density
        log_prior = -(a+1)*log(var_y) - b/var_y;
        log_prior_prop = -(a+1)*log(var_y_prop) - b/var_y_prop;

        log_r = log_Z_S_prop - log_Z_S + log_prior_prop - log_prior;

        decision = rand < exp(log_r);
        if decision == 1
            var_y = var_y_prop;
        end
    end

    % store the variables
    theta_vec(:, i) = theta;
    var_y_vec(i) = var_y;
    
    if update_S == 1
        Sigma_vec(:, :, i) = Sigma;
        for j = 1:J
            S_vec{j}(:, :, i) = S{j};
        end
        a_MH_vec(i, :) = a_MH;
        decision_rate_vecs(i) = mean(decision_vec);
    end
end

% store the outputs
outputs.theta_vec = theta_vec;
outputs.var_y_vec = var_y_vec;
if update_S == 1
    outputs.a_MH_vec = a_MH_vec;
    outputs.S_vec = S_vec;
    outputs.acceptance_rate_vec = decision_rate_vecs;
    outputs.Sigma_vec = Sigma_vec;
end