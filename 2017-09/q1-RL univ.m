%%% q1 Regressao Linear Univariada
%usar alg gradiente descendente estocastico para encontrar os coeficientes da regressao

clear all;
close all;
clc;

data = load('ex1data1.txt');
x=data(:,1);
y=data(:,2);
%x=dados entrada
%y=dados saida
%(x,y) -> um unico exemplo de treino
%(xi,yi) -> um exemplo de treino especifico i (indice da i-esima linha)
%           um training set é um conjunto de m training examples

%propostos no problema
alfa=0.001;
epocas=1000;

%tamanho da amostra de treino (qtde training examples) 
m=length(y);

%valores "chutados" entre 0 e 1 (poderia ser apenas 0)
w0 = randn; %theta0
w1 = randn; %theta1

for j=1:epocas
  eqm(j,1) = j; %posicao do erro
	eqm(j,2) = 0; %valor do erro inicializado com zero
  
	for i=1:m
    %yi=w1*xi+w0
		yi = w1*x(i)+w0;
    
    %erro de cada ponto -> dif entre o real e o calculado
		ei = y(i) - yi;       
    
    %somatorio dos erros -> somat erros ao quadrado=erro acumulado + (erro atual ao quadrado)
		eqm(j,2) = eqm(j,2) + ei^2; 
    
    %atualiza w0 e w1 simultaneamente: 
    %wo=w0+alfa*ei 
    %w1=w1+alfa*ei*xi
		w0 = w0 + alfa * ei;
		w1 = w1 + alfa * ei*x(i);
  end
  
  %depois do somatorio, divide por m
	eqm(j,2) = eqm(j,2)/m;
  
  %permuta valores de x e y a cada epoca
  idx = randperm(m);
	x = x(idx, :);
  y = y(idx);  
end

%reta de regressao
y0 = w1*min(data(:,1))+w0;
y1 = w1*max(data(:,1))+w0;

figure(1);
%mostra figura com dados iniciais - 1a parte da questao
plot(data(:,1), data(:,2),'o');%mesmo que: scatter(data(:,1), data(:,2));
hold all;

plot([min(data(:,1)), max(data(:,1))], [y0,y1]);
title('Regressao linear');
ylabel('y');
xlabel('x');

%valor final dos coeficientes
w0
w1

%grafico epocas x erro - eqm guarda par posicao x valor
figure(2);
plot(eqm(:,1),eqm(:,2));
title('Grafico EQM por epoca');
ylabel('erro');
xlabel('epoca');

