%%% *** q1 kmedias
%- Implemente o k-médias para a base de dados, utilizando somente os 4 primeiros atributos.
%- Varie o número de clusters entre 2 e 5
%- Calcule o somatório dos erros quadráticos em relação aos centroides para cada número de agrupamentos. 

%iniciando
close all; clc; clear all; 

%-matriz com 150 linhas e 5 colunas
%-as 4 primeiras colunas representam 4 atributos (tamanho e espessura da sépala e da pétala de cada flor)
%-a coluna 5 representa a classe a qual pertence o exemplo (1-setosa, 2-versicolor e 3-virginica)
data = load('ex5data1.data');
x = data(:,1:4);
y = data(:,5);
[lin col]=size(x);

%normalizacao 
for i=1:4
  med1=mean(x(:,i));
  d1=max(x(:,i))-min(x(:,i));
  x(:,i) = (x(:,i)-med1)/d1;
end

%embaralha
I=randperm(lin);
x=x(I,:);
y=y(I,:);

for k=2:5
  %dividindo dados entre os clusters
  %k = num clusters  
  if k==2
    c1 = x(1:75,:);
    y1 = y(1:75,:);
    c2 = x(76:150,:);
    y2 = y(76:150,:);
  elseif k==3
    c1 = x(1:50,:);
    y1 = y(1:50,:);
    c2 = x(51:100,:);
    y2 = y(51:100,:);
    c3 = x(101:150,:);
    y3 = y(101:150,:);
  elseif k==4
    c1 = x(1:37,:);
    y1 = y(1:37,:);
    c2 = x(38:75,:);
    y2 = y(38:75,:);
    c3 = x(76:113,:);
    y3 = y(76:113,:);
    c4 = x(114:150,:); %menor
    y4 = y(114:150,:);
  else
    c1 = x(1:30,:);
    y1 = y(1:30,:);
    c2 = x(31:60,:);
    y2 = y(31:60,:);
    c3 = x(61:90,:);
    y3 = y(61:90,:);
    c4 = x(91:120,:);
    y4 = y(91:120,:);
    c5 = x(121:150,:);
    y5 = y(121:150,:);
  end

  %pega aleatoriamente k pontos como centroides
  p = randperm(size(x,1)); 
  for i =1:k
      C(i,:)= x(p(i),:);
  end

  T=zeros(lin,1);
  distancia = zeros(lin,1);
  
  muda=true;
  while muda
    %calcula a distancia dos pontos para os centroides
    
    %varre cada ponto
    for cont1=1:lin
        for cont2=1:k
            %distancia do ponto para cada centroide (dist entre vetores: norm)
            tempdist(cont2) = norm(x(cont1,:)-C(cont2,:));
        end
        %ind vai indicar o grupamento
        %vmax=valor, ind=indice no vetor
        [vmax, ind] = min(tempdist);
        Y(cont1) = ind;
    end

    %se centroide nao mudou
    if (Y==T)
      muda=false;
    else
      %recebe atual
      T=Y;
      
      %atualiza os centroides pela nova distribuicao em classes 
      %Centróides são recalculados como a média dos pontos do grupo
      for i=1:k
        Mt = [x Y']; %une col Y' a x
        Mt = Mt((Mt(:,5)==i),1:4); %deixa onde Y for igual a i
        C(i,:)=mean(Mt, 1);
      end
    end
  end

  %- Calcule o somatório dos erros quadráticos em relação aos centroides para cada número de agrupamentos
  %erro = soma de todas as distâncias
  X = [x Y']; %une col Y' a x
  erro = 0;
  for i=1:k
    Ms = X((X(:,5)==i),:);
    D = norm(Ms(:,1:4)-C(i,:));
    %distancia(Ms(:,1:4),C(i,:))
    erro = erro + sum(D(:));
  end
  
  errok(k-1)=erro^2; %erro quadratico
end

%Apresentar: Gráfico do erro pelo número de agrupamentos
%Apresentar: O número de agrupamentos para este problema, de acordo com a heurística apresentada em aula
figure(01)
plot(errok,2:5)

figure(02)
[H, ax, bigax] = gplotmatrix(x, x, Y);
axes(bigax);

%- Execute o K-médias para o número de agrupamentos obtidos
%- Compare o resultado com o valor real das classes

%(tamanho e espessura da sépala e da pétala de cada flor)
labs = {'Tam sepala', 'Esp sepala', 'Tam petala', 'Esp petala'};
for i = 1:4
    txtax = axes('Position', get(ax(i,i), 'Position'),'units','normalized');
    text(.1, .5, labs{i});
    set(txtax, 'xtick', [], 'ytick', []);
end

%ref:
%http://www.diegonogare.net/2015/08/entendendo-como-funciona-o-algoritmo-de-cluster-k-means/
%http://docplayer.com.br/25697053-Pratica-8-a-distancia-euclidiana-entre-dois-vetores-n-dimensionais-x-e-y-e-definida-como-o-escalar-d-norm-x-y.html
