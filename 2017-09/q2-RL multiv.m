%%% Regressao Linear Multivariada
%usar gradiente descendente estocastico para encontrar os coeficientes da regressao
%encontrar os coeficientes da regressao utilizando o metodo dos mínimos quadrados

clear all;
close all;
clc;

data = load('ex1data2.txt');
x=data(:,1:2);
y=data(:,3);
%x=dados entrada
%y=dados saida

%propostos no problema
alfa=0.01;
epocas=100;

%tamanho da amostra de treino (qtde training examples) 
m=length(y);

#inserindo coluna de 1s no inicio
x = [ones(m,1) x];
x = x';

%valores "chutados" (poderia ser apenas 0)
w = [randn; randn; randn] ; %theta0, theta1, theta2 em vetor col

for j=1:epocas
  eqm(j,1) = j; %posicao do erro
	eqm(j,2) = 0; %valor do erro inicializado com zero
  
	for i=1:m
    %y=wt*xi
		yi = w'*x(:,i);
    
    %erro de cada ponto -> dif entre o real e o calculado
		ei = y(i) - yi;       
    
    %somatorio dos erros -> somat erros ao quadrado=erro acumulado + (erro atual ao quadrado)
		eqm(j,2) = eqm(j,2) + ei^2; 
    
    %atualiza w simultaneamente: 
    %wn=wn+alfa*ei*xi
		w(1) = w(1) + alfa * ei*x(1,i);
		w(2) = w(2) + alfa * ei*x(2,i);
    w(3) = w(3) + alfa * ei*x(3,i);
  end
  
  %depois do somatorio, divide por m
	eqm(j,2) = eqm(j,2)/m;
  
  %permuta valores de x e y a cada epoca
  idx = randperm(m);
	x = x(:,idx);
  y = y(idx);  
end

%valor final dos coeficientes
w

%grafico epocas x erro - eqm guarda par posicao x valor
figure(01)
plot(eqm(:,1),eqm(:,2));
title('Grafico EQM por epoca');
ylabel('erro');
xlabel('epoca');

%%% --- usando metodo dos minimos quadrados

%relendo valores iniciais de x e y
x = data(:,1:2);
y = data(:,3);
x = [ones(m,1) x];

%calculo dos pesos
w_mq = inv(x'*x)*x'*y;
yi = w_mq'*x';
erro = y - yi;

%erro medio dos mínimos quadrados
e = 0;
for i=1:length(erro)
    e = e + erro(i)^2;
end
e = e/m;

%valor final dos coeficientes
w_mq

%grafico epocas x erro - eqm guarda par posicao x valor
figure(02)
hold all
p1 = plot((0:2), w, 's-');   %plota square linha solida
set(p1,'MarkerFaceColor','b');
p2 = plot((0:2), w_mq, 'd-');%plota diamond linha solida
set(p2,'MarkerFaceColor','g');

title('Coeficientes');
ylabel('x');
xlabel('y');
legend('Gradiente descendente', 'Mínimos quadrados');

