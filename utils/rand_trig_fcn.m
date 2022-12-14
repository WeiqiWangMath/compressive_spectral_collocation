%--- Description ---%
%
% Filename: iso_exp.m
% Authors: Ben Adcock, Simone Brugiapaglia and Clayton Webster
% Part of the book "Sparse Polynomial Approximation of High-Dimensional
% Functions", SIAM, 2021
%
% Description: computes the isotropic exponential function f_1 defined in Appendix A
%
% Input:
% y - m x d array of sample points
%
% Output:
% b - m x 1 array of function values at the sample points

function y = rand_trig_fcn(k,x)

[m,d] = size(x);
y = ones(m,1);

for i = 1 : d
    y = y .* cos((2*k(i)+1)*pi*x(:,i));
end

end