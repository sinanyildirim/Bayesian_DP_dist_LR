% main function for MMD calculations.
% 
clear; clc; close all; fc = 0;

Js_vec = [1 5 10]; L_Js = length(Js_vec);
Alg_vec = [1 2 4 5];
d_vec = [2]; L_d = length(d_vec);
h = 0.1;
MMD_cell = cell(L_d, L_Js);
for is0 = 1:L_d
    d = d_vec(is0);
    for is1 = 1:L_Js
        Js = Js_vec(is1);

        filenametoload = sprintf('DP_LR_simul__n_100000_d_%d_M_50_K_10000_L_1_alg_11111_J_%d_eps_1_2_5_10_20_50_100.mat', d, Js);
        load(filenametoload);

        if is1 == 1
            K_true = 2000;
            theta_nonDP_samps = zeros(d, K_true);
            for i = 1:K_true
                var_y = 1/gamrnd(a_n, 1/b_n, 1);
                C_post = var_y*(lambda_n\eye(d));
                C_post = (C_post + C_post')/2;
                theta_nonDP_samps(:, i) =  mvnrnd(mu_n, C_post)';
            end

            c = 0;
            Term1 = zeros(1, K_true*(K_true-1));
            for i = 1:K_true
                for j = 1:K_true
                    if i ~= j
                        c = c + 1;
                        Term1(c) = -norm(theta_nonDP_samps(:, i) - theta_nonDP_samps(:, j))^2/(2*h);
                    end
                end
            end
            D1 = mean(exp(Term1));
        end

        MMD_array = zeros(L_eps, M, 4);

        for is2 = 1:L_eps
            for is3 = 1:M
                for is4 = Alg_vec
                    disp([is0 is1 is2 is3 is4]);
                    if is4 == 4
                        cov_theta = cov_thetas{1, is2, is3, is4};
                        cov_theta = (cov_theta + cov_theta')/2;
                        theta_samps = mvnrnd(mu_thetas{1, is2, is3, is4}, cov_theta, 2000)';
                    else
                        theta_samps = Theta_samps{1, is2, is3, is4}(:, 50:50:end);
                    end
                    MMD_array(is2, is3, is4) = MMD(theta_nonDP_samps, theta_samps, h, D1);
                       
                end
            end
        end
        MMD_cell{is0, is1} = MMD_array;

    end
end

%% Plot
for is0 = 1:L_d
    d = d_vec(is0);
    for is1 = 1:L_Js
        Js = Js_vec(is1);

        color_codes = ["black" "blue" "red","[0.6350 0.0780 0.1840]","magenta"];
        style_codes = ["*-k" "-ok" ".-k" "-squarek" "-xk"];
        alg_names = ["MCMC-normalX" "MCMC-fixedS" "adaSSP" "Bayes-fixedS-fast" "MCMC-B&S"];

        % fc = fc + 1; figure(fc);
        subplot(L_d, L_Js, (is0-1)*L_Js + is1);
        for k = Alg_vec
            temp_MMD = squeeze(MMD_cell{is0, is1}(:, k, :));
            plot(1:L_eps, mean(temp_MMD, 2), style_codes(k), "Color",color_codes(k));
            % boxplot(temp_MMD');
            hold on;
        end
        hold off;
        set(gca, 'Xtick', 1:L_eps, 'XtickLabel', epsilon_vec);
        title(sprintf('MMD$^2$, $d = %d$ $J=%d$', d, Js), 'interpreter', 'Latex');
        xlabel('$\epsilon$', 'Interpreter','Latex');
        if is0 == L_d && is1 == L_Js
            legend(alg_names(Alg_vec));
        end

    end
end
