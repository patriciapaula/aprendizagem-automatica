%%% q1 Regressao logistica - exibir dados

clear all;
close all;
clc;

data = load('ex2data1.txt');
x=data(:,1:2); %col 1 e 2 - entrada
y=data(:,3);   %resultado
m=length(y);

%%% normalizando entrada
med1=mean(x(:,1));
med2=mean(x(:,2));
d1=max(x(:,1))-min(x(:,1));
d2=max(x(:,2))-min(x(:,2));
x(:,1) = (x(:,1)-med1)/d1;
x(:,2) = (x(:,2)-med2)/d2;

%%% plotando todos os dados
figure(01);
hold on;
title('Alunos');
for i=1:m
  px = x(i,1);
  py = x(i,2);
  if (y(i))  
    plot (px,py,'bo-');
  else 
    plot (px,py,'ro-');
  end
end
%legend('Admitidos', 'Nao admitidos');
xlabel('Nota1');
ylabel('Nota2');
hold off;
