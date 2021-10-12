%%% *** q2 pca

%PCA:
%Toma os dados de entrada (a)
%Subtrai as médias
%Calcula a matriz de covariância
%Calcula os autovalores e autovetores
%Escolhe os k maiores autovalores
%Utiliza os k autovetores correspondentes para criar k novos atributos

%iniciando
close all; clc; clear all; 

%dados de entrada
data= load('ex5data1.data');
X = data(:,1:4);

%subtrai medias
c = mean(X);
X= X-repmat(c, size(X,1), 1);

%matriz de covariancia
covar = cov(X);

%- Utilizando a mesma base de dados da questão anterior, 
%  aplique o algoritmo PCA e reduza a dimensão de modo a preservar 99% da variância.

%autovalores e autovetores (eigenvalues, eigenvectors)
%d recebe autovalores de covar
d = eigs(covar);
%Autovalores d = [4.2248     0.2422     0.0785     0.0237]

%verificando com 2 atributos mais relevantes:
s = sum(d);
% 4.5693

s2 = sum(d(1:2,:));
%s = 4.4671
var2 = s2 / s;
%97.76%

%verificando com 3 atributos mais relevantes:
s3 = sum(d(1:3,:));
%s = 4.5456
var3 = s3 / s;
%99.48%

%- Reduza a dimensão da base de dados original para 2.

%LA = Largest real
opt.disp = 0;
[p, D] = eigs(covar, 2, 'LA', opt);

reduced = X*p;

%Apresentar: Figura em 2 dimensões com os dados. 
%Utilize cores diferentes para cada classe.

figure(2)
hold on;
plot(reduced(1:50, 1), reduced(1:50, 2), '+');
plot(reduced(51:100, 1), reduced(51:100, 2), '*');
plot(reduced(101:150, 1), reduced(101:150, 2), 'o');
hold off;
legend('Setosa', 'Versicolor','Virginica');

