%%% RL Regularizacao
%encontrar os coeficientes da regressao utilizando o metodo dos minimos quadrados regularizado 

clear all;
close all;
clc;

%propostos no problema
lambda = [0 1 2 3 4 5];
qtde_treino = 30;
qtde_teste = 17;

%lendo primeiras 30 linhas
data = load('ex1data3.txt');
x = data(1:30,1:5);
y = data(1:30,6);
%x=dados entrada
%y=dados saida

%tamanho da amostra de treino (qtde training examples) 
m=length(y);

#inserindo coluna de 1s no inicio
x = [ones(30,1) x];

%valores "chutados"
w = randn(6,1);
alfa = 0.01;

%matriz identidade (identidade de n+1 por n+1 com (0,0)=0 -> onde n eh num features)
l = eye(6);
l(1,1)=0;

fc = zeros(6,1);

for j=1:6
  eqm =0;
  for i=1:30    
    p1 = x' * x;
    p2 = lambda(j)*l;
    p3 = x' * y;
    w = inv(p1 + p2) * p3;
    
    yi = x(i,:)*w;
    ei = y(i) - yi;
    eqm = eqm + ei *ei;
        
    fc(j) = 0.5 * ((y - yi)' * (y - yi) .+ (lambda(j) * w' * w)); 
  end
  wtreino(:,j)=w;
  eqmtreino(j)=eqm/30;
end

%valor final coeficientes e erros
wtreino
eqmtreino
lambda
fc

%grafico lambda x erros
figure(01)
hold all
plot(lambda, eqmtreino);
title('Grafico EQM x lambda - Treinamento');
ylabel('erro');
xlabel('lambda');

% *******************

%lendo apenas ultimas linhas pra teste
x = data(31:47,1:5);
y = data(31:47,6);
x = [ones(17,1) x];
w = wtreino;

%tamanho da amostra de teste 
m=length(y);

fc = zeros(6,1);
for j=1:6
  eqmt =0;
  for i=1:17
    yit = x(i,:)*w(:,j);
    eit = y(i) - yit;
    eqmt = eqmt + eit *eit;
    %w ja foi calculado!
  end
  eqmteste(j)=eqmt/17;
end

%valor final erros
eqmteste

%grafico lambda x erros
figure(02)
plot(lambda, eqmteste);
title('Grafico EQM x lambda - Teste');
ylabel('erro');
xlabel('lambda');
