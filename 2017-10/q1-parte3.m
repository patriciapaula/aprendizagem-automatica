%%% q1 Regressao logistica
%usar alg gradiente descendente estocastico e utilizar k-fold para validacao cruzada do resultado

clear all;
close all;
clc;

data = load('ex2data1.txt');
x=data(:,1:2); %col 1 e 2 - entrada
y=data(:,3);   %resultado

m=length(y);

alfa = 0.01;
epocas = 1000;

% normalizando entrada
med1=mean(x(:,1));
med2=mean(x(:,2));
d1=max(x(:,1))-min(x(:,1));
d2=max(x(:,2))-min(x(:,2));
x(:,1) = (x(:,1)-med1)/d1;
x(:,2) = (x(:,2)-med2)/d2;

%1: O conjunto de dados de tamanho m(m exemplos) é dividido em k conjuntos disjuntos de tamanho m/k
%2: O algoritmo é treinado k vezes, cada vez com um conjunto diferente sendo deixado de fora 
%   para fazer a validação (1 conj pra teste por vez)
%3: O desempenho é estimado como sendo o erro médio ou taxa de acerto média sobre estes 
%   k conjuntos de validação

x=[ones(m,1) x]';

k = 10;
tamFold = m/k;
w = [randn; randn; randn];

qtde_treino = m - tamFold;
qtde_teste = tamFold;

for z = 1:k
  %%% separando teste e treino
  
  pos1 = (z-1)*tamFold + 1;
  pos2 = z*tamFold;
  
  % separa 1 grupo pra teste - o resto eh pra treino
  xteste = x(:,pos1:pos2);
  yteste = y(pos1:pos2);
  
  % montagem dos conj treino eh dif com z=1, z=k
  if z==1
    %treino eh o q vem depois do teste
    xtreino = x(:,pos2+1:m);
    ytreino = y(pos2+1:m);
  elseif z==k
    %treino eh o q vem antes do teste
    xtreino = x(:,1:pos1-1);
    ytreino = y(1:pos1-1);
  else
    %treino - pega o q vem antes do teste e junta com o q vem depois
    xtreino1 = x(:,1:pos1-1);
    ytreino1 = y(1:pos1-1);
    
    xtreino2 = x(:,pos2+1:m);
    ytreino2 = y(pos2+1:m);
        
    xtreino = [xtreino1 xtreino2];
    ytreino = [ytreino1; ytreino2];
  end
  
  %%% treino ***
  eqm = zeros(tamFold,2);
  for j=1:epocas
    eqm(j,1) = j; %posicao do erro
    eqm(j,2) = 0; %valor do erro inicializado com zero
    
    for i=1:length(ytreino)
      ex = -1 .* (w' * xtreino(:,i));
      yi = 1 ./ (1 + exp(ex));

      %erro de cada ponto -> dif entre o real e o calculado
      ei = ytreino(i) - yi;
      
      %somatorio dos erros
      eqm(j,2) = eqm(j,2) + ei ^2;

      %guarda cada w por fold
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
  % guarda w e erro por fold
  w_final(:,z) = w;
  erro_treino(z) = eqm(epocas,2);
  
  %%% teste ***
  y2 = zeros(k,1);
  for i=1:k
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
  total_erro = sum(erro);
  perc_sucesso(z) = 1 - (total_erro / k);
end

%
%perc_sucesso = 1 - (total_erro / qtde_teste);
%perc_erro = 1 - perc_sucesso;
w_final
sucesso = sum(perc_sucesso)/k*100

figure(01)
hold on;
title('Coeficientes por fold');
plot([1:10],w_final(1,:), 'bo-');
plot([1:10],w_final(2,:), 'ko-');
plot([1:10],w_final(3,:), 'ro-');
legend('w0', 'w1', 'w2');
hold off;

figure(02)
title('% sucesso x kfold');
plot([1:k],perc_sucesso);

figure(03)
title('erro x kfold');
plot([1:k],erro_treino);

