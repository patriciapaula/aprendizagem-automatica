function [Y C erro] = kmedias(M, k)

[lin, col]= size(M);

%Pega aleatoriamente 3 pontos como centróide
p = randperm(size(M,1)); 
for i =1:k
    C(i,:)= M(p( i ),:);
end

T=zeros(lin, 1 );
while 1
    %Calcula a distância dos pontos para os centróides
    D = distancia(M,C); 
    [Z,Y] = min(D,[],2);
    %Se não houve mudança de grupos, pare
    if Y==T 
        break;
    else %Recebe a configuração atual
        T=Y; 
    end
    %Atualiza os centróides de acordo com a nova configuração
    for i=1:k
        Mt = [M Y];
        Mt = Mt((Mt(:,5)==i),1:4); 
        C(i,:)=mean(Mt, 1);
    end
end
%Encontrar o erro, que é a soma de todas as distâncias
X = [M Y];
erro = 0;
for i=1:k
  Ms = X((X(:,5)==i),:);
  D = distancia(Ms(:,1:4),C(i,:));
  erro = erro + sum(D(:));
end
