function [ K ] = TCK(GMM, Xte, C, G)
% TCK -  compute TCK kernel matrix between training data and test data Xte
%
% INPUTS
% GMM : Cell output from the function trainTCK
% Xte: data array of size Nte x T x V, where Nte is the number of multivariate time series, T the length and V the number of attributes.

%
% OUTPUTS
% K: kernel matrix

%
% Reference: "Time Series Cluster Kernel for Learning Similarities between Multivariate Time Series with Missing Data", 2017 Pattern Recognition, Elsevier.
% Authors: "Karl �yvind Mikalsen, Filippo Maria Bianchi"

nan_idx = isnan(Xte);
if(sum(sum(sum(nan_idx)))>0)
    missing = 1;
else
    missing = 0;
end


K = zeros(size(GMM{1,1},1),size(Xte,1));
parfor i=1:G*(C-1)
    c= floor((i-1)/G) + 2;
    K = K + GMM{i,1}*GMMposterior( Xte, c, GMM{i,2}, GMM{i,3}, GMM{i,4}, GMM{i,5}, GMM{i,6}, missing )';
end



