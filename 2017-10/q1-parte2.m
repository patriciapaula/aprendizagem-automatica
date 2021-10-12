%%% q1 Regressao logistica
%usar alg gradiente descendente estocastico para encontrar os coeficientes do classificador

clear all;
close all;
clc;

data = load('ex2data1.txt');
x=data(:,1:2); %col 1 e 2 - entrada
y=data(:,3);   %resultado
m=length(y);

qtde_treino = 70;
qtde_teste = m - 70;

alfa = 0.01;
epocas = 1000;
w = randn(1,3)'
%[randn; randn; randn];

% normalizando entrada
med1=mean(x(:,1));
med2=mean(x(:,2));
d1=max(x(:,1))-min(x(:,1));
d2=max(x(:,2))-min(x(:,2));
x(:,1) = (x(:,1)-med1)/d1;
x(:,2) = (x(:,2)-med2)/d2;

% separando teste e treino
xtreino = x(1:qtde_treino,:);
ytreino = y(1:qtde_treino);
xtreino = [ones(qtde_treino,1) xtreino]';

xteste = x(qtde_treino+1:m,:);
yteste = y(qtde_treino+1:m);
xteste = [ones(qtde_teste,1) xteste]';

%%% treino ***

eqm = zeros(qtde_treino,2);
for j=1:epocas
  eqm(j,1) = j; %posicao do erro
  eqm(j,2) = 0; %valor do erro inicializado com zero
  
  for i=1:qtde_treino
		ex = -1 .* (w' * xtreino(:,i));
    yi = 1 ./ (1 + exp(ex));

    %erro de cada ponto -> dif entre o real e o calculado
		ei = ytreino(i) - yi; 
    
    %somatorio dos erros
		eqm(j,2) = eqm(j,2) + ei ^ 2;

    w(1) = w(1) + alfa * ei * xtreino(1,i);
    w(2) = w(2) + alfa * ei * xtreino(2,i);
    w(3) = w(3) + alfa * ei * xtreino(3,i);
  end
  %permuta valores de x e y a cada epoca
  idx = randperm(qtde_treino);
	xtreino = xtreino(:,idx);
  ytreino = ytreino(idx);
  
  %depois do somatorio, divide por qtde
	eqm(j,2) = eqm(j,2)/qtde_treino;  
end

%valor final coeficientes e erro
w
eqm(qtde_treino,2)

% epocas x erro
figure(01);
hold all;
plot(eqm(:,1),eqm(:,2));
title('Epocas x Erro');
xlabel('Epoca');
ylabel('Erro');
hold off;

%%% teste ***

y2 = zeros(qtde_teste,1);
for i=1:qtde_teste
  % usando w do treino
  ex = -1 .* (w' * xteste(:,i));
  yi = 1 ./ (1 + exp(ex));

  %monta resultado para comparar com real
  if yi >= 0.5
    y2(i) = 1;
  else
    y2(i) = 0;
  end
end

% calculando acertos
erro = abs(yteste - y2);
total_erro = sum(erro)
perc_sucesso = 1 - (total_erro / qtde_teste)
perc_erro = 1 - perc_sucesso
