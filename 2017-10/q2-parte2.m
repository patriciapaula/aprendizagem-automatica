%%% q2 Regressao logistica regularizada
%usar alg gradiente descendente estocastico para encontrar os coeficientes da regressao

clear all;
close all;
clc;

data = load('ex2data2.txt');
x=data(:,1:2); %col 1 e 2 - entradas
y=data(:,3);   %resultado
m=length(y);

% normalizando entrada - essa normalizacao nao fica ok pra esses dados
med1=mean(x(:,1));
med2=mean(x(:,2));
d1=max(x(:,1))-min(x(:,1));
d2=max(x(:,2))-min(x(:,2));
x(:,1) = (x(:,1)-med1)/d1;
x(:,2) = (x(:,2)-med2)/d2;

alfa = 0.01;
epocas = 1000;

%lambda: 0; 0.01; 0.25
lambda = 0.01;

% utilizando metodo (do prof) para aumentar o num de atributos
x = mapFeature(x(:,1), x(:,2));
[l c] = size(x)

% gera w de acordo com num de atributos de x
w = randn(1, c)';

Xplot = x;
x = x';

eqm = zeros(epocas,2);
for j=1:epocas
  eqm(j,1) = j; %posicao do erro
  eqm(j,2) = 0; %valor do erro inicializado com zero
  
  for i=1:m
    ex = -1 * w' * x(:,i);
    yi = 1 ./ (1 + exp(ex));

    %erro de cada ponto -> dif entre o real e o calculado
    ei = y(i) - yi;
      
    %somatorio dos erros
    eqm(j,2) = eqm(j,2) + ei ^2;
    
    for k=1:c
      if k==1
        %primeiro eh sem lambda
        w(k) = w(k) + alfa * (ei* x(k,i));
      else
        %com lambda
        w(k) = w(k) + alfa * (ei* x(k,i) - lambda * w(k));
      end
    end
  end
  %depois do somatorio, divide por qtde x
	eqm(j,2) = eqm(j,2)/m;
  
  %permuta valores de x e y a cada epoca
  idx = randperm(m);
	x = x(:,idx);
  y = y(idx); 
end

x = Xplot;
figure(01);
% utilizando metodo (do prof) de geracao da superficie de decisao
plotDecisionBoundary(w, x, y);
w
