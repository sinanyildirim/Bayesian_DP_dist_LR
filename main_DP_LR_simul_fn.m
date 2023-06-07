function outputfile = main_DP_LR_simul_fn(simul_or_real, dataname, ...
    N_train, d, L_avg, M, K, algo_to_run, J_vec, epsilon_vec, rng_no, run_empty)

% outputfile = main_DP_LR_simul_fn(simul_or_real, dataname, ...
%    N_train, d, L_avg, M, K, algo_to_run, J_vec, epsilon_vec, rng_no, run_empty)
%
% Runs the experiments

%% outputfile name
outputfile = [sprintf('DP_LR_%s_%s_n_%d_d_%d_M_%d_K_%d_L_%d', simul_or_real, dataname, N_train, d, M, K, L_avg) ...
    '_alg_', sprintf('%d', algo_to_run), ...
    '_J', sprintf('_%d', J_vec)...
    '_eps', sprintf('_%d', 10*epsilon_vec)];

%% If run empty, exit here with the name of the outputfile (for plotting purposes)
if run_empty == 1
    return;
end

% If not run empty, run experiments
rng(rng_no);

% Num of nodes: Distributed for J > 1.
n_alg = length(algo_to_run);
L_J = length(J_vec);
L_eps = length(epsilon_vec);

if strcmp(simul_or_real, 'real') == 1
    data = load([dataname, '.csv']);
    % preprocess
    data(isnan(data))=0;
    N_all = size(data, 1); 
    d = size(data, 2) - 1;
    test_ratio = 0.2;
    N_train = ceil(N_all*(1 - test_ratio));
    % Get all X and all Y
    X_all = data(:,1:end-1);
    X_all = zscore(X_all);
    Y_all = data(:, end);
    Y_all = Y_all - mean(Y_all);
    Y_all = Y_all/max(abs(Y_all));
    
    % training data
    X_train = X_all(1:N_train, :);
    Y_train = Y_all(1:N_train);
    % test data
    X_test = X_all(N_train + 1: end, :);
    Y_test = Y_all(N_train + 1: end);

    % OLS solution using the training data
    theta_OLS = (X_train'*X_train)\(X_train'*Y_train);

    % This will be used as a hyperparameter for the covariance of X
    V = randn(d); 
    Lambda = V'*V;

elseif strcmp(simul_or_real, 'simul') == 1
    V = randn(d); Lambda = V'*V; 
    kappa = d + 1;
    var_y = 1;
    % sample covariance for X
    Sigma_X = iwishrnd(Lambda, kappa);
    % sample theta
    theta_true = mvnrnd(zeros(1, d), eye(d))';
    % sample X
    X_train = mvnrnd(zeros(1, d), Sigma_X, N_train);
    % sample Y
    Y_train = X_train*theta_true + sqrt(var_y)*randn(N_train, 1);
end
% set the delta values based on the training data size
delta = 1/N_train;
D = d^2 + d + 1; % dimension of SS =  [X'X, X'Y, Y'Y]

%% Determine bound_X and bound_Y
bound_X = max(sum(X_train.^2, 2))^0.5;
bound_Y = max(abs(Y_train));
bound_SS = sqrt(bound_X^4 + (bound_X*bound_Y)^2 + bound_Y^4);
bound_SZ = sqrt(bound_X^4 + (bound_X*bound_Y)^2);

%% make rows approximately normal (appendix C.1)
if L_avg > 1
    N_train_new = floor(N_train/L_avg);
    N_floor = N_train_new*L_avg;
    X_train = reshape(mean(reshape(X_train(1:N_floor, :), L_avg, d*N_train_new), 1), N_train_new, d)/sqrt(L_avg);
    Y_train = reshape(mean(reshape(Y_train(1:N_floor), L_avg, N_train_new), 1), N_train_new, 1)/sqrt(L_avg);
    N_train = N_train_new;
end

%% model hyperparameters

% Model hyperparameters
a_0 = 20; b_0 = 0.5; lambda_0 = b_0/(a_0-1)*eye(d); C = (a_0-1)/b_0*eye(d); m = zeros(d, 1);
hyperparams = struct('Lambda', Lambda, 'kappa', d+1, 'a', a_0, 'b', b_0, ...
    'm', m, 'C', C, 'lambda_0', lambda_0);


%%  non-private estimation
S_true = X_train'*X_train;
Z_true = X_train'*Y_train;
Y2_true = Y_train'*Y_train;

lambda_n = S_true + lambda_0;
lambda_n = (lambda_n + lambda_n')/2;

mu_n = lambda_n\Z_true;
a_n = a_0 + N_train/2;
b_n = b_0 + 0.5*(Y2_true + m'*lambda_0*m -  mu_n'*lambda_n*mu_n);
% Ensure it is at least b_0
b_n = max(b_n, b_0);

% now calculate the moments
mu_theta_nonDP = mu_n;
Cov_theta_nonDP = (b_n/(a_n-1))*eye(d)/lambda_n;
if strcmp(simul_or_real, 'real') == 1
    MSE_nonDP = mean((mu_theta_nonDP - theta_OLS).^2);
    Pred_nonDP = mean((Y_test - X_test*mu_theta_nonDP).^2);
elseif strcmp(simul_or_real, 'simul') == 1
    MSE_nonDP = mean((mu_theta_nonDP - theta_true).^2);
    Pred_nonDP = trace(Sigma_X*(mu_theta_nonDP - theta_true)*(mu_theta_nonDP - theta_true)');
end
%%
% The parameters of adaSSP
DP_params = struct('bound_X', bound_X, 'bound_Y', bound_Y, 'p', 0.05, 'delta', delta);

% Number of MCMC iterations
t_burn = K/2;

% initialize the errors
MSE_reps = zeros(L_J, L_eps, M, n_alg);
Pred_reps = zeros(L_J, L_eps, M, n_alg);
Time_reps = zeros(L_J, L_eps, M, n_alg);
mu_thetas = cell(L_J, L_eps, M, n_alg);
cov_thetas = cell(L_J, L_eps, M, n_alg);
Theta_samps = cell  (L_J, L_eps, M, n_alg);
    
for i1 = 1:L_J

    % get the number of nodes
    J = J_vec(i1);

    % distribute the data into nodes
    N_node0 = floor(N_train/J);
    N_last = N_train - (J-1)*N_node0;
    N_node = [N_node0*ones(1, J-1) N_last];
    N_node_cumul = [0 cumsum(N_node)];

    %% Generate the distributed data
    Sd = cell(1, J);
    Zd = cell(1, J);
    Y2d = cell(1, J);
    
    for j = 1:J
        node_rows = N_node_cumul(j)+1:N_node_cumul(j+1);
    
        X_node = X_train(node_rows, :);
        Y_node = Y_train(node_rows);
    
        % construct S = X'X
        Sd{j} = X_node'*X_node;
        % construct Z = X'y
        Zd{j} = X_node'*Y_node;
        % construct Y^TY
        Y2d{j} = Y_node'*Y_node;        
    end

    % The proposal parameters for MCMC
    MCMC_prop_params.a_MH = 10.^(2*(log10(N_node)-1));
    MCMC_prop_params.sigma_q_y = 0.1;
            
    for i2 = 1:L_eps
        epsilon = epsilon_vec(i2);

        % Set DP parameters
        DP_params.epsilon = epsilon;

        % arrange the DP parameters
        if epsilon == inf
            sigma_unit1 = 0;
            sigma_unit2 = 0;
        else
            sigma_unit1 = analytic_Gaussian_mech(epsilon, delta);
            sigma_unit2 = analytic_Gaussian_mech(epsilon/2, delta/2);
        end

        std_SS = sigma_unit1*bound_SS;
        std_SZ = sigma_unit1*bound_SZ;
        std_S1 = sigma_unit2*bound_X^2;
        std_Z1 = sigma_unit2*bound_X*bound_Y;
                
        if std_S1^2 + std_Z1^2 > std_SZ^2
            std_S = std_SZ;
            std_Z = std_SZ;
        else
            std_S = std_S1;
            std_Z = std_Z1;
        end
        disp([std_S, std_Z, std_SS]);

        % Set the hyperparameters
        hyperparams.var_S = std_S^2;
        hyperparams.var_Z = std_Z^2;        
        hyperparams.var_SS = std_SS^2;        

        % repeat experiments to reduce the effect of randomness
        for i3 = 1:M
            
            fprintf('\nReplication %d for J = %d and epsilon = %.2f \n', i3, J, epsilon);
            
            % Generate the noisy observations
            Sd_obs = cell(1, J);
            Zd_obs = cell(1, J);
            SSd_obs = zeros(D, J);

            for j = 1:J
                %%% A. sample the noisy versions of S and Y
                % sample the noisy versions of S and Y
                upper_temp_Z = triu(randn(d),1);
                lower_temp_Z = upper_temp_Z';
                noise_S = std_S*(upper_temp_Z + lower_temp_Z + diag(randn(1, d)));
                noise_Z = std_Z*randn(d,1);

                Sd_obs{j} = Sd{j} + noise_S;
                Zd_obs{j} = Zd{j} + noise_Z;
                
                %%% B. construct noisy version for Bernstein&Sheldon
                SSd = [Sd{j}(:); Zd{j}(:); Y2d{j}];

                % construct the noisy version of SS = [S, Z, YTY]
                upper_temp_Z = triu(randn(d), 1);
                lower_temp_Z = upper_temp_Z';
                noise_S = std_SS*(upper_temp_Z + lower_temp_Z + diag(randn(1, d)));
                noise_Z = std_SS*randn(d,1);
                noise_Y2 = std_SS*randn;

                SSd_obs(:, j) = SSd + [noise_S(:); noise_Z(:); noise_Y2];
            end

            %% initial variables and algorithm parameters
            S0 = cell(1, J);
            SS0 = zeros(D, J);
            for j = 1:J
                % find the nearest psd matrix and construct a pd matrix
                % close to S_obs
                % For MCMC
                S0{j} = closest_psd(Sd_obs{j}) + eye(d);                
                
                % For MCMC-BS
                SS0j = closest_psd(reshape(SSd_obs(1:d^2), d, d)) + eye(d);
                SS0(:, j) = [SS0j(:); SSd_obs(d^2+(1:d), j); SSd_obs(end, j)];
            end
            init_vars.S = S0;
            init_vars.theta = zeros(d, 1);
            init_vars.var_y = bound_Y/3;
            init_vars.SS = SS0; % For MCMC-BS

            %% Run the methods to compare
            %%%%%%%%%%%%%%%%%%%%%%%% Method 1: MCMC-normX %%%%%%%%
            if algo_to_run(1) == 1
                fprintf('MCMC with S update Wishart proposal for S... \n');
                tic;
                outputs = MCMC_DP_LR(Sd_obs, Zd_obs, init_vars, hyperparams, MCMC_prop_params, N_node, K, 1);
                Time_reps(i1, i2, i3, 1) = toc;
                theta_samps{1} = outputs.theta_vec(:, t_burn+1:end);

                mu_thetas{i1, i2, i3, 1} = mean(theta_samps{1}, 2);
                cov_thetas{i1, i2, i3, 1} = cov(theta_samps{1}');
                Theta_samps{i1, i2, i3, 1} = theta_samps{1};

            end

            %%%%%%%%%%%%%%%%%%%%%%%% Method 2: MCMC-fixedS %%%%%
            if algo_to_run(2) == 1
                fprintf('Implementing MCMC without S update... \n');
                tic;
                outputs = MCMC_DP_LR(Sd_obs, Zd_obs, init_vars, hyperparams, MCMC_prop_params, N_node, K, 0);
                Time_reps(i1, i2, i3, 2) = toc;
                theta_samps{2} = outputs.theta_vec(:, t_burn+1:end);

                mu_thetas{i1, i2, i3, 2} = mean(theta_samps{2}, 2);
                cov_thetas{i1, i2, i3, 2} = cov(theta_samps{2}');
                Theta_samps{i1, i2, i3, 2} = theta_samps{2};
            end

            %%%%%%%%%%%%%%%%%%%%%%%% Method 3: ADASSP %%%%%%%%%%%%%%%%%%%%
            if algo_to_run(3) == 1
                fprintf('Implementing adaSSP... \n');
                tic;
                theta_est = adaSSP(Sd, Zd, DP_params);
                Time_reps(i1, i2, i3, 3) = toc;
                theta_samps{3} = theta_est;

                mu_thetas{i1, i2, i3, 3} = mean(theta_samps{3}, 2);
                cov_thetas{i1, i2, i3, 3} = cov(theta_samps{3}');
                Theta_samps{i1, i2, i3, 3} = theta_samps{3};
            end

            %%%%%%%%%%%%%%%%%%%%%%%% Method 4: Bayes-fixedS-fast %%%%%%%%
            if algo_to_run(4) == 1
                fprintf('Implementing Bayesian adaSSP v2... \n');
                tic;
                [theta_est, theta_cov_est] = Fast_Bayesian_DP_LR(Sd_obs, Zd_obs, DP_params, hyperparams);
                Time_reps(i1, i2, i3, 4) = toc;
                theta_samps{4} = theta_est;
                mu_thetas{i1, i2, i3, 4} = theta_est;
                cov_thetas{i1, i2, i3, 4} = theta_cov_est;
                Theta_samps{i1, i2, i3, 4} = theta_est;
            end

            %%%%%%%%%%%%%%%%%%%%%%%% Method 5: MCMC-B&S %%%%%
            if algo_to_run(5) == 1
                fprintf('Implementing Bernstein&Sheldon... \n')
                tic;
                [outputs] = MCMC_DP_LR_BS(SSd_obs, init_vars, hyperparams, N_node, K);
                Time_reps(i1, i2, i3, 5) = toc;
                theta_samps{5} = outputs.theta_vec(:,t_burn+1:end);
                mu_thetas{i1, i2, i3, 5} = mean(theta_samps{5}, 2);
                cov_thetas{i1, i2, i3, 5} = cov(theta_samps{5}');
                Theta_samps{i1, i2, i3, 5} = theta_samps{5};
                
            end

            %% Calculate the errors
            for i_alg = 1:n_alg
                if algo_to_run(i_alg) == 1
                    theta_samp = theta_samps{i_alg};
                    theta_point_est = mean(theta_samp, 2);
                    K_samp = size(theta_samp, 2);

                    if strcmp(simul_or_real, 'real') == 1
                        MSE_reps(i1, i2, i3, i_alg) = mean((theta_point_est - theta_OLS).^2);
                        Pred_reps(i1, i2, i3, i_alg) = mean((Y_test - X_test*theta_point_est).^2);
                    elseif strcmp(simul_or_real, 'simul') == 1
                        MSE_reps(i1, i2, i3, i_alg) = mean((theta_point_est - theta_true).^2);
                        Pred_reps(i1, i2, i3, i_alg) = trace(Sigma_X*(theta_samp - theta_true)*(theta_samp - theta_true)'/K_samp);
                    end
                end
            end

        end
    end
end

% mean MSE, Pred, Times
MSE = squeeze(mean(MSE_reps, 3));
Pred = squeeze(mean(Pred_reps, 3));
Times = squeeze(mean(Time_reps, [2, 3]));

save(outputfile);