function [Y C erro] = kmedias(M, k)

[lin, col]= size(M);

%Pega aleatoriamente 3 pontos como centr�ide
p = randperm(size(M,1)); 
for i =1:k
    C(i,:)= M(p( i ),:);
end

T=zeros(lin, 1 );
while 1
    %Calcula a dist�ncia dos pontos para os centr�ides
    D = distancia(M,C); 
    [Z,Y] = min(D,[],2);
    %Se n�o houve mudan�a de grupos, pare
    if Y==T 
        break;
    else %Recebe a configura��o atual
        T=Y; 
    end
    %Atualiza os centr�ides de acordo com a nova configura��o
    for i=1:k
        Mt = [M Y];
        Mt = Mt((Mt(:,5)==i),1:4); 
        C(i,:)=mean(Mt, 1);
    end
end
%Encontrar o erro, que � a soma de todas as dist�ncias
X = [M Y];
erro = 0;
for i=1:k
  Ms = X((X(:,5)==i),:);
  D = distancia(Ms(:,1:4),C(i,:));
  erro = erro + sum(D(:));
end
