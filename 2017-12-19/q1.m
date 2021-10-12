%%% *** q1 - SVM

%iniciando
close all; clc; clear all; 

%-matriz X é composta de 51 linhas e 2 colunas
%-vetor y dá a classe a qual pertence cada vetor
load('ex6data1.mat');

%-Apresentar: Figura com o conjunto de dados

figure(01);
plotData(X, y);

%-usar função svmTrain e função visualizeBoundaryLinear
%-utilize o Kernel linear
%-Utilize valores de C = 1 e C = 100

C = 1;
model = svmTrain(X, y, C, @linearKernel, 1e-5, 20);
figure(02)
visualizeBoundaryLinear(X, y, model);
disp('w para C=1')
disp(model.w)

C = 100;
model = svmTrain(X, y, C, @linearKernel, 1e-5, 20);
figure(03)
visualizeBoundaryLinear(X, y, model);
disp('w para C=100')
disp(model.w)

%-Utilize C=0.001 e refaça o experimento

C = 0.001;
model = svmTrain(X, y, C, @linearKernel, 1e-5, 20);
figure(04)
visualizeBoundaryLinear(X, y, model);
disp('w para C=0.001')
disp(model.w)

%roda indefinidamente e nao consegue convergir
%y(37) = 1;
%C = 1e9;
%model = svmTrain(X, y, C, @linearKernel, 1e-5, 20);
%figure(05)
%visualizeBoundaryLinear(X, y, model);
%disp('w para C=1e9')
%disp(model.w)


