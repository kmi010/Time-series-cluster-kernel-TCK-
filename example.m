close all
clear

% load synthetic two-variate VAR dataset
load('x_VAR.mat');
X = reshape(x,[200,50,2]);
Y = [ones(100,1); 2*ones(100,1)];
load('xte_VAR.mat');
Xte = reshape(xte,[200,50,2]);
Yte = Y;

% add missing data to the synthetic data
Z = ones(size(X));
R = rand(size(X));
Z(R>0.5) = 0;
X(Z==0) = nan;
Zte = ones(size(Xte));
Rte = rand(size(Xte));
Zte(Rte>0.5) = 0;
Xte(Zte==0) = nan;


% Train GMM models
[GMMpar,C,G]  = trainTCK(X);

%Compute in-sample kernel matrix
K = TCK(GMMpar,C,G);

% Compute similarity between Xte and the training points
Kte = TCK(GMMpar,C,G,Xte);

% 1NN -classifier
Nte = length(Yte);
[C,I] = max(Kte);    % find training series with maximum similarity
pred_Y = Y(I);     % 1NN classification
accuracy=sum(pred_Y==Yte)/Nte
