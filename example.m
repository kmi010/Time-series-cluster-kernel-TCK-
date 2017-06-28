close all
clear

% load synthetic two-variate VAR dataset
load('x_VAR.mat');
X = reshape(x,[200,50,2]);
Y = [ones(100,1); 2*ones(100,1)];
load('xte_VAR.mat');
Xte = reshape(xte,[200,50,2]);
Yte = Y;

C = 40;
G = 30;

% Train GMM models
[GMMpar,C,G]  = trainTCK(X,'C',C,'G',G);

% Compute TCK kernel
Kte = TCK(GMMpar,Xte,C,G);

% 1NN -classifier
Nte = length(Yte);
[C,I] = max(Kte);    % find training series with maximum similarity
pred_Y = Y(I);     % 1NN classification
accuracy=sum(pred_Y==Yte)/Nte
