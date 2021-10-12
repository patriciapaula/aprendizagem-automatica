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


% ----------------------------------------
Mdl = fitcnb(xtreino,ytreino);
isLabels1 = resubPredict(Mdl);
mconf = confusionmat(ytreino,isLabels1);
tx = sum(diag(mconf))/(3*300);
disp(['Precisao = ',num2str(tx*100),'%']);
%precisao do treino: 64.1111%

Mdl = fitcnb(xteste,yteste);
isLabels1 = resubPredict(Mdl);
mconf = confusionmat(yteste,isLabels1)
tx = sum(diag(mconf))/(3*300);
disp(['Precisao = ',num2str(tx*100),'%']);
%precisao do teste: 95.6667%



