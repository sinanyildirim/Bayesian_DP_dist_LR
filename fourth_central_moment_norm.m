function [M, S] = fourth_central_moment_norm(C)

% [M, S] = fourth_central_moment_norm(C)
%
% Calculates fourth the central moments of a normal distribution with
% covariance C

d = size(C, 1);

M = zeros(d, d, d, d);
S = zeros(d^2, d^2);
for i = 1:d
    for j = 1:d
        for k = 1:d
            for l = 1:d                
                term1 = C(i, j)*C(k, l);
                term2 = C(i, k)*C(j, l);
                term3 = C(i, l)*C(j, k);
                M(i, j, k, l) = term1 + term2 + term3;
                S((i-1)*d+j, (k-1)*d+l) = term2 + term3;
            end
        end
    end
end
                
                
   

