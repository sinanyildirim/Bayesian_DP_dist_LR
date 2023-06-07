# Differentially Private Distributed Bayesian Linear Regression with MCMC

This work focuses on differentially private Bayesian inference and linear regression using Markov chain Monte Carlo methods. We propose a distributed setting where each party share its own noisy sufficient statistics for the inference. Given noisy statistics, we develop several algorithms and compare them with state-of-art methods. 

#### 
The main file for conducting the experiments is `main_run_tests_DP_LR`. In this file, `exp_no` controls the type of experiments and `algo_to_run` decides on the algorithms to be run for that particular experiment. 
- Use `exp_no = 1` for measuring the run times of MCMC algorithms.
- Use `exp_no = 2` for measuring the errors of the algorithms when the data is artificial.
- Use `exp_no = 3` for measuring the errors of the algorithms when the data is real. 
- Use `show_results = 0` to save the output file without plotting.
- Use `show_results = 1` to plot/show the results of that specific experiment.
- Options for `algo_to_run`: [MCMC-normalX, MCMC-fixedS, adaSSP, Bayes-fixedS-fast, MCMC-B\&S].

#### 
The code `main_run_tests_DP_LR.m` runs the function `main_DP_LR_simul_fn.m` to conduct the experiments. This file runs the experiments and outputs the name of the relevant output file. If the option show_results = 1 is chosen, it skips running the experiments.
The other key parameters are as follows: 
- $n$ - number of rows
- $d$ - number of features
- $J$ - number of parties.
- $\epsilon$ - differential privacy parameter.
- $M$ - number of Monte Carlo runs for averaging performances.
- $K$ - number of iterations for the MCMC algorithms.

#### 
The code main_MMD_calculations calculates and plots the squared MMD estimates between the DP posterior and the true posterior. This code is relevant to simulated-data experiments where the true posterior is known. It uses the results of the simulated-data experiments.

## Functions for Algorithms

- `MCMC_DP_LR`: This function implements MCMC-normalX (when 'update_S' = 1) and MCMC-fixedS (when 'update_S = 0'). 
  - Input - $S_{obs}$, $Z_{obs}$ (noisy versions of sufficient statistics), initial variables, proposal parameters, `update_S`.
  - Output - samples of $\theta$ (regression coefficients), samples of $\sigma_y$ (residual variance), samples of $S = X^TX$ if `update_S = 1`, samples of $\Sigma_x$ (covariance matrix of $X$) if `update_S = 1`. 

- `Fast_Bayesian_DP_LR`: This function implements Bayes-fixedS-fast in the paper. 
  - Input - $S_{obs}$, $Z_{obs}$ (noisy versions of sufficient statistics), privacy parameters.
  - Output - point estimation of $\theta$. 

- `MCMC_DP_LR_BS`: This function implements hierarchical MCMC algorithm from Bernstein & Sheldon (2019) by extending the methodology for distributed setting as in Appendix B.
  - Input - $SS_d$ (noisy version sufficient statistics vector), initial values.
  - Output - samples of $\theta$ (regression coefficients), samples of $\sigma_y$ (residual variance).

- `adaSSP`: This function implements `adaSSP` from Wang (2018) by extending the methodology for distributed setting as in Appendix C.
  - Input - $S$, $Z$ (sufficient statistics without noise), privacy parameters.
  - Output - $\theta$ (estimation for the regression coefficients)

## Other functions

- `MH_S_update`: Implements a Metropolis-Hastings move for $S$ update using Wishart proposal. 
  - Input - $S$ (current value), $S_{obs}$, $Z_{obs}$ (noisy version of $S$ and $Z$), $\theta$ (current value), $\Sigma$ (current value), $\sigma_y$ (current value of residual variance), privacy and proposal parameters
  - Output - The updated value of $S$ 

- `analytic_Gauss_mech`: Implements analytic gauss mechanism algorithm from Balle & Wang (2018, Algorithm 1) for unit sensitivity.
  - Input - $\epsilon$, $\delta$
  - Output - noise standard deviation

 - 'fourth_central_moment_norm.m':  Calculates fourth-order central moments of the normal distribution with covariance $C$ using the method from Triantafyllopoulos (2002, Moments and cumulants of multivariate real and complex Gaussian distributions)
   - Input - $C$, the covariance matrix
   - Output - $M$, fourth moments of the distribution, $S$ the covariance of the vectorised form of $XX^T$ when $X \sim \mathcal{N}(0, C)$.

- `moments_S_norm`: Calculates the mean and covariance of the prior distribution of the vectorised form of the sufficient statistics of MCMC-B\&S
  - Input - $\Sigma_x$ (covariance matrix of $X$), $\theta$ (current value), $\sigma_y$ (current value). 
  - Output - The moments $\mu_{ss}$, $\Sigma_{ss}$

- `closest_psd`: This function finds the closest positive semi-definite matrix to $X$ in terms of the Frobenous norm.
  - Input - matrix
  - Output - closest positive semi-definite matrix of the input matrix
  
-  `MMD`: This function estimates the squared Maximum mean discrepancy value between two distributions when samples from those distributions are provided as inputs. 
	- Input - $x$, $y$ (samples from the two distributions between which MMD is to be estimated), $h$ (bandwidth parameter) $D1$, $D2$, $D3$ (the three terms (if their values are known) whose sum gives the answer - all of them are optional). 
	- Output - squared MMD estimate.

