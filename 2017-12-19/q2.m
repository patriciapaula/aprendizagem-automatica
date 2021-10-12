%%% *** q2 - SVM

%iniciando
close all; clc; clear all; 

%-matriz X é composta de 863 linhas e 2 colunas 
%-vetor y dá a classe a qual pertence cada vetor
load('ex6data2.mat');

%-Apresentar: Figura com o conjunto de dados

%figure(01);
%plotData(X, y);

%-Kernel Gaussian RBF
%-sigma = 0.1 e sigma = 0.2
%-Utilize C = 1

C = 1; 

%sigma = 0.1;
sigma = 0.2;

model = svmTrain(X, y, C, @(x1, x2) gaussianKernel(x1, x2, sigma));
visualizeBoundary(X, y, model);

