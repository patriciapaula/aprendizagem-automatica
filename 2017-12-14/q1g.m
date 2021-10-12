clc
clear all
close all
%Dados
data= load('ex5data1.data');
X = data(:,1:4);
Y = data(:,5);
k = 5;

%normalizacao 
for i=1:4
  med1=mean(X(:,i));
  d1=max(X(:,i))-min(X(:,i));
  X(:,i) = (X(:,i)-med1)/d1;
end

for i=1:k
  [R C e] = kmedias(X, i);  
  erros(i) = e^2;
end
figure(01)
plot(erros)

figure(02)
[H, ax, bigax] = gplotmatrix(X, X, R);
axes(bigax);

labs = {'Tamanho Sépala', 'Espessura Sépala', 'Tamanho Pétala', 'Espessura Pétala'};
for i = 1:4
    txtax = axes('Position', get(ax(i,i), 'Position'),'units','normalized');
    text(.1, .5, labs{i});
    set(txtax, 'xtick', [], 'ytick', []);
end

