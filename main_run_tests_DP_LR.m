% run this file for the experiments in the paper
% 
% Differentially Private Distributed Bayesian Linear Regression with MCMC 
% Baris Alparslan, Sinan Yildirim, Ilker Birbil
% ICML 2023
%
% choose exp_no = 1 for the comparison of running times
% choose exp_no = 2 for the accuracy experiments on simulated data
% choose exp_no = 3 for the experiments on real data sets.
% 
% choose show_results = 0 to actually run the experiments,
% choose show_results = 1 to load the already-run experiments and show
% their results.
% 
% choose plot_CI = 1 to plot to calculate the CI of MSEs for exp_no = 2.

clear; clc; close all;
exp_no = 3; show_results = 1;   
plot_CI = 0;
alpha_CI = 0.10; % 1 - this is the confidence level 

% dataname = 'Sydney_data.csv'; K_avg = 12; d = 49, n = 72000
% dataname = 'power_plant_energy'; d = 5, n = 9568;
% dataname = '3droad.csv'; d = 4; n = 434874;
% dataname = 'bike_sharing.csv'; d = 16; n = 17379;
dataname = 'air_quality.csv'; d = 13; n = 9357;
datafilename = 'air_quality';

switch exp_no
    case 1 
        %% Tests for running time
        algo_to_run = [1 1 0 0 1];
        d_vec = 1:20; L_d = length(d_vec);
        J_vec = [1 5 10];
        epsilon_vec = 1;
        outputfile = cell(1, L_d);
        M = 1;
        TIMES = zeros(3, 5, L_d);
        rng_no = 1;
        K = 10^4;
        n = 10^4;
        L_avg = 1;
        for i = 1:L_d
            d = d_vec(i);
            outputfile{d} = main_DP_LR_simul_fn('simul', '', n, d, L_avg, ...
                M, K, algo_to_run, J_vec, epsilon_vec, rng_no, show_results);
            load(outputfile{i}, 'Times');
            TIMES(:, :, d) = Times;
        end
        if show_results == 1
            color_codes = ["black" "blue" "red","[0.6350 0.0780 0.1840]","magenta"];
            style_codes = ["*-k" "-ok" ".-k" "-squarek" "-xk"];

            for j = 1:3
                J = J_vec(j);
                subplot(1,3,j);

                hold on
                for i=[1 2 5]
                    plot(d_vec,squeeze(TIMES(j, i,:))/K, style_codes(i),"Color",color_codes(i));
                end
                hold off;

                xlabel('$d$', 'Interpreter', 'Latex');
                if j == 1
                    legend('MCMC-normalX','MCMC-fixedS','MCMC-B&S');
                end
                title(sprintf('$J=%d$', J), 'Interpreter', 'Latex');
            end
        end
    case 2
        %% Tests for MSE and prediction
        algo_to_run = [1 1 1 1 1];
        epsilon_vec = [0.1 0.2 0.5 1 2 5 10]; L_eps = length(epsilon_vec);
        d = 2;
        M = 50;
        rng_no = 1;
        K = 10^4;
        n = 10^5;
        L_avg = 1;

        J_vec_cell = {1, 5, 10}; L_J = length(J_vec_cell);
        outputfile = cell(1, L_J);
        MSEs = cell(1, L_J);
        Preds = cell(1, L_J);

        for j_cell = 1:L_J
            J = J_vec_cell{j_cell};

            outputfile{j_cell} = main_DP_LR_simul_fn('simul', '', n, d, L_avg, ...
                M, K, algo_to_run, J, epsilon_vec, rng_no, show_results);

            load(outputfile{j_cell}, 'MSE', 'Pred', 'Pred_reps', 'MSE_reps');
            MSEs{j_cell} = MSE;
            Preds{j_cell} = Pred;

            pred_CI_mean = squeeze(mean(Pred_reps, 3));
            pred_CI_std = squeeze(std(Pred_reps, [], 3));
            MSE_CI_mean = squeeze(mean(MSE_reps, 3));
            MSE_CI_std = squeeze(std(MSE_reps, [], 3));
            
            % plot results
            if show_results == 1
                color_codes = ["black" "blue" "red","[0.6350 0.0780 0.1840]","magenta"];
                style_codes = ["*-k" "-ok" ".-k" "-squarek" "-xk"];
                alg_names = ["MCMC-normalX" "MCMC-fixedS" "adaSSP" "Bayes-fixedS-fast" "MCMC-B&S"];
                Alg_vec = [1 2 4 5 3];
                figure(1);
                subplot(2, L_J, j_cell);
                for k = Alg_vec
                    if plot_CI == 1
                        t_c = tinv(1-alpha_CI/2, M-1);
                        Hw = t_c*pred_CI_std(:, k)/sqrt(M);
                        
                        x = 1:L_eps;
                        x2 = [x, fliplr(x)];
                        curve1 = pred_CI_mean(:, k) + Hw;
                        curve2 = max(pred_CI_mean(:, k) - Hw, 0);
                        inBetween = [log(curve1'), log(fliplr(curve2'))];
                        fill(x2, inBetween, 'g', 'HandleVisibility','on', 'FaceAlpha',0.5);
                        hold on;
                    end
                    plot(1:L_eps, log(Pred(:, k)), style_codes(k),"Color",color_codes(k));
                    hold on;
                end
                hold off;
                set(gca, 'Xtick', 1:L_eps, 'XtickLabel', epsilon_vec);
                temp_mtx = log(Pred(:, [1 2 4 5]));
                set(gca, 'ylim',  [min(log(Pred(:)))-0.25 max(temp_mtx(:))+0.25]);

                title(sprintf('(log-)MSE: prediction, $J=%d$', J), 'interpreter', 'Latex');

                xlabel('$\epsilon$', 'Interpreter','Latex');
                % ylabel('$\log$(MSE)', 'Interpreter','Latex');
                if j_cell == L_J
                    legend(alg_names(Alg_vec));
                end
                
                subplot(2, L_J, L_J + j_cell);
                for k= Alg_vec
                    if plot_CI == 1
                        t_c = tinv(1-alpha_CI/2, M-1);
                        Hw = t_c*MSE_CI_std(:, k)/sqrt(M);

                        x = 1:L_eps;
                        x2 = [x, fliplr(x)];
                        curve1 = MSE_CI_mean(:, k) + Hw;
                        curve2 = max(MSE_CI_mean(:, k) - Hw, 0);
                        inBetween = [log(curve1'), log(fliplr(curve2'))];
                        fill(x2, inBetween, 'y', 'HandleVisibility','off', 'FaceAlpha',0.5);
                        hold on;
                    end
                    plot(1:L_eps,log(MSE(:,k)),style_codes(k),"Color",color_codes(k))
                    hold on;
                end
                hold off;
                set(gca, 'Xtick', 1:L_eps, 'XtickLabel', epsilon_vec);
                temp_mtx = log(MSE(:, [1 2 4 5]));
                set(gca, 'ylim',  [min(log(MSE(:)))-0.25 max(temp_mtx(:))+0.25]);                
                xlabel('$\epsilon$', 'Interpreter','Latex');
                title(sprintf('(log-)MSE: estimation $J=%d$', J), 'interpreter', 'Latex');
            end
        end

    case 3
        %% Tests for real data        
        algo_to_run = [1 1 1 1 1];
        epsilon_vec = 1;
        M = 50;
        rng_no = 1;
        K = 10^4;
        L_avg = 1;
        
        J_vec_cell = {1, 5, 10}; L_J = length(J_vec_cell);
        outputfile = cell(1, L_J);
        MSEs = cell(1, L_J);
        Preds = cell(1, L_J);
        
        % Initialise for Confidence Intervals
        t_c = tinv(1-alpha_CI/2, M-1);
        CI_pred = cell(1, L_J);
        Pred_Reps =cell(1, L_J);

        for j_cell = 1:L_J
            J_vec = J_vec_cell{j_cell};

            outputfile{j_cell} = main_DP_LR_simul_fn('real', datafilename, 0, 0, L_avg, ...
                M, K, algo_to_run, J_vec, epsilon_vec, rng_no, show_results);
            load(outputfile{j_cell}, 'Pred', 'Pred_reps');
            Preds{j_cell} = Pred;
            Pred_Reps{j_cell} = Pred_reps;
            CI_pred{j_cell} = zeros(5, 2);
        
            for i = 1:5
                Hw = t_c*std(Pred_reps(:, :, :, i), [], 3)/sqrt(M);
                CI_mean = mean(Pred_reps(:, :, :, i), 3);
                CI_pred{j_cell}(i, :) = [CI_mean - Hw CI_mean + Hw];
            end
        end
end