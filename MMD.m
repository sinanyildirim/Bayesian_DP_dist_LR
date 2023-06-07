function [D] = MMD(x, y, h, D1, D2, D3)


% [D] = MMD(x, y, h, D1, D2, D3)
% 
% Maximum mean discrepancy

m = size(x, 2);
n = size(y, 2);
d = size(x, 1);
if nargin < 4
    c = 0;
    Term1 = zeros(1, m*(m-1));
    for i = 1:m
        for j = 1:m
            if i ~= j
                c = c + 1;
                Term1(c) = -norm(x(:, i) - x(:, j))^2/(2*h);
            end
        end
    end
    D1 = mean(exp(Term1));
end


if nargin < 5
    Term2 = zeros(1, n*(n-1));
    c = 0;
    for i = 1:n
        for j = 1:n
            if i ~= j
                c = c + 1;
                Term2(c) = -norm(y(:, i) - y(:, j))^2/(2*h);
            end
        end
    end
    D2 = mean(exp(Term2));
end

if nargin < 6
    c = 0;
    Term3 = zeros(1, n*m);
    for i = 1:m
        for j = 1:n
            c = c + 1;
            Term3(c) = -norm(x(:, i) - y(:, j))^2/(2*h);
        end
    end
    D3 = mean(exp(Term3));
end
       
D = D1 + D2 - 2*D3;
D = D*((2*pi*h)^(-d/2));

