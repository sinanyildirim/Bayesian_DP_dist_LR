function [Y] = closest_psd(X)

% [Y] = closest_psd(X)
%
% Finds the closes psd matrix to X in terms of frobenous norm
%
[~, D, E] = eig(X);
v = diag(D) > 0;
Y = E(:, v)*D(v, v)*E(:, v)';
Y = real(Y);