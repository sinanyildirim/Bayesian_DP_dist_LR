function [sigma] = analytic_Gaussian_mech(epsilon, delta)

% [sigma] = analytic_Gaussian_mech(epsilon, delta)
%
% A crude implementation of the analytic Gaussian mechanism of 
% (Algorithm 1) Balle and Wang (2018), 
% "Improving the Gaussian Mechanism for Differential Privacy: Analytical 
% Calibration and Optimal Denoising"

delta_0 = normcdf(0) - exp(epsilon)*normcdf(-2*sqrt(epsilon));
if delta >= delta_0
    v_max = 0;
    c = 0;
    while c==0
        v = v_max:0.001:(v_max+1);
        v_max = v_max + 1;
        B = normcdf(sqrt(epsilon)*v) - exp(epsilon)*normcdf(-sqrt(epsilon*(v+2)));
        c = (max(B) > delta)*(min(B) < delta);
    end
    v_sol = v(sum(B <= delta));
    alpha = sqrt(1 + v_sol/2) - sqrt(v_sol/2);
else
    u_max = 0;
    c = 0;
    while c==0
        u = u_max:0.001:(u_max+1);
        u_max = u_max + 1;
        B = normcdf(-sqrt(epsilon)*u) - exp(epsilon)*normcdf(-sqrt(epsilon*(u+2)));
        c = (min(B) < delta)*(max(B) > delta);
    end
    u_sol = u(sum(B > delta) + 1);
    alpha = sqrt(1 + u_sol/2) + sqrt(u_sol/2);
end
sigma = alpha/sqrt(2*epsilon);
    





