%%% *** q1a naive bayes
%Implemente os seguintes classificadores: Naive Bayes e Discriminante Quadrático Gaussiano
%Apresentar: As matrizes de confusão para os dois classificadores.

%iniciando
close all; clc; clear all; 

%-A matriz de dados (Dados) é composta de 1500 linhas e 2 colunas. 
%-A matriz dos rótulos (y) apresenta os rótulos das classes. 
load('DadosLista4.mat');
x=Dados;

%normalizacao
for i=1:2
  med1=mean(x(:,i));
  d1=max(x(:,i))-min(x(:,i));
  x(:,i) = (x(:,i)-med1)/d1;
end

%-Nestes dados, existem 3 classes, sendo 500 exemplos de cada classe.

xc1=x(1:500,:);
xc2=x(501:1000,:);
xc3=x(1001:1500,:);
yc1=y(1:500);
yc2=y(501:1000,:);
yc3=y(1001:1500,:);

hold all;
plot(xc1(:,1),xc1(:,2),'bo');
plot(xc2(:,1),xc2(:,2),'ro');
plot(xc3(:,1),xc3(:,2),'go');

%-Divida aleatoriamente o conjunto de dados entre treino e teste. 

%embaralha
I1=randperm(500); %500 eh o tam cada classe
xc1=xc1(I1,:);
yc1=yc1(I1,:);
I2=randperm(500);
xc2=xc2(I2,:);
yc2=yc2(I2,:);
I3=randperm(500);
xc3=xc3(I3,:);
yc3=yc3(I3,:);

%-Para este problema, utilize 600 (200 de cada classe) dados para treino, 
%900 (300 de cada classe) dados para teste.

%treinamento - entrada e saida
xtreino = [xc1(1:200,:); xc2(1:200,:); xc3(1:200,:)];
ytreino = [yc1(1:200,:); yc2(1:200,:); yc3(1:200,:)];

%teste
xteste = [xc1(201:end,:); xc2(201:end,:); xc3(201:end,:)];
yteste = [yc1(201:end,:); yc2(201:end,:); yc3(201:end,:)];

yu = unique(yteste); %C = unique(A) returns the same data as in A, but with no repetitions. C is in sorted order
numclasses = length(yu);   %classes - podia colocar 3
numvarx = size(xtreino,2); %variaveis - podia colocar 2
tamteste = length(yteste); %tam do teste - podia colocar 900

%probabilidade de cada classe
for i=1:numclasses
    py(i)=sum(double(ytreino==yu(i)))/length(ytreino); %mesmo efeito de atribuir 1/3 
end

%parametros do treinamento
for i=1:numclasses
    xi=xtreino((ytreino==yu(i)),:);
    media(i,:)=mean(xi,1); %media
    desviop(i,:)=std(xi,1);%desvio padrao
end

%probabilidade para o conjunto de teste
%normcdf - normal cumulative distribution function
for j=1:tamteste
    Pt=normcdf(ones(numclasses,1)*xteste(j,:),media,desviop);
    P(j,:)=py.*prod(Pt,2)';
end

%resultado para o conjunto de teste
[p,id]=max(P,[],2);
for i=1:length(id)
    pr(i,1)=yu(id(i));
end

yu=unique(yteste);   %podia colocar 3
MC=zeros(length(yu));% 3x3
for i=1:length(yu)
    for j=1:length(yu)
        MC(i,j)=sum(yteste==yu(i) & pr==yu(j));
    end
end

disp('---');

disp('Matriz de Confusao Naive Bayes:');
disp(MC);
conf=sum(pr==yteste)/length(pr);
disp(['Accuracy = ',num2str(conf*100),'%']);

%%% *** q1b distribuição quad gaussiana
%Implemente os seguintes classificadores: Naive Bayes e Discriminante Quadrático Gaussiano
%Apresentar: As matrizes de confusão para os dois classificadores.

tamtreino = length(ytreino); %tam do treino
yu = unique(y);
tamtreino = 200;
tamteste = 300;

%parametros do treinamento
MC = [];
for i=1:numclasses
    Xi=xtreino((ytreino==yu(i)),:);
    Md(i,:)=mean(Xi,1); % media
    mco = cov(Xi,1); % matriz diagonal de covariancia
    MC = [MC; mco];
end

%probabilidade para o conjunto de teste - MC 6x2
MConf = zeros(numclasses, numclasses);
for j=1:3*tamteste
    PC1=1/(sqrt(norm(MC(1:2,:)))*(2*pi)^(tamtreino/2))*exp(-1/2*(xteste(j,:)-Md(1,:))*inv(MC(1:2,:))*(xteste(j,:)-Md(1,:))');
    PC2=1/(sqrt(norm(MC(3:4,:)))*(2*pi)^(tamtreino/2))*exp(-1/2*(xteste(j,:)-Md(2,:))*inv(MC(3:4,:))*(xteste(j,:)-Md(2,:))');
    PC3=1/(sqrt(norm(MC(5:6,:)))*(2*pi)^(tamtreino/2))*exp(-1/2*(xteste(j,:)-Md(3,:))*inv(MC(5:6,:))*(xteste(j,:)-Md(3,:))');
    
    [me ie] = max([PC1 PC2 PC3]);
    MConf(yteste(j,1),ie) = MConf(yteste(j,1),ie)+1;
end

disp('Matriz de Confusao DQG:')
MConf
tx = sum(diag(MConf))/(3*tamteste);
disp(['Accuracy = ',num2str(tx*100),'%'])

%ref:
%   http://www.mathworks.com/matlabcentral/fileexchange/37737-naive-bayes-classifier?focused=5239664&tab=function
%   https://machinelearningmastery.com/confusion-matrix-machine-learning/

