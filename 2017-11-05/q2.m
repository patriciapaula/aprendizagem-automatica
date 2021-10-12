%%% q2 redes neurais - regressao

clear all; close all; clc;

D=load('ex3data2.data');
[lin col]=size(D);

%embaralha
I=randperm(lin);
D=D(I,:);

%normalizacao
for i=1:13
  med1=mean(D(:,i));
  d1=max(D(:,i))-min(D(:,i));
  D(:,i) = (D(:,i)-med1)/d1;
end
% Normalização dos dados entre 0.1 e 0.9 ***** pq?
%for i=1:13
%       D(:,i)=D(:,i)*0.8/(max(D(:,i)-min(D(:,i))))+ 0.9 -...
%           (0.8 * max(D(:,i)))/(max(D(:,i))-min(D(:,i)));
%end


%%%dividindo conjuntos

%treinamento - entrada e saida
xtreino = D(1:306,1:13); 
ytreino = D(1:306,14); 
[ltreino ctreino]=size(xtreino); %num linhas e colunas

%validação
xvalid = D(307:406,1:13); 
yvalid = D(307:406,14); 
[lvalid cvalid]=size(xvalid); %num linhas e colunas

%teste
xteste = D(407:end,1:13); 
yteste = D(407:end,14);  
[lteste cteste]=size(xteste);  %num linhas e colunas


%%%definindo arquitetura
epocas = 2000; % *** 5000
alfa = 0.001;  % *** 0.001

%num neuronios
qNeuroCO = 30;   %num camada oculta   = 2p+1 -> heuristica de hetch-nielsen no=2ni+1 *** 50
qNeuroCS = 1;   %num camada de saida = num classes

%intervalo de análise de erro    %aumento da media dos erros nos últimos x% da qtde de epocas?
analiseErro = floor(epocas*0.05);

%matrizes de pesos - gera aleatoriamente
WW=0.1*rand(qNeuroCO,ctreino+1);  %pesos entrada para camada oculta
MM=0.1*rand(qNeuroCS,qNeuroCO+1); %pesos camada oculta para camada de saida ***** 0.1* ?


%%%treinamento
for epoca=1:epocas
    %epoca

    %embaralha dados 
    I=randperm(ltreino); 
    xtreino=xtreino(I,:); 
    ytreino=ytreino(I,:);   %embaralha vetores de treinamento e saidas desejadas
    
    %erro quadratico medio
    EQ=0;
    for i=1:ltreino,   % Inicia epocas de treinamento
        % CAMADA OCULTA
        Xe=horzcat(-1, xtreino(i,:)); % Constroi vetor de entrada com adicao da entrada x0=-1
        Ui = WW * Xe';          % Ativacao (net) dos neuronios da camada oculta
        Yi = 1./(1+exp(-Ui));   % Saida entre [0,1] (funcao logistica)

        % CAMADA DE SAIDA 
        Ye=[-1; Yi];            % Constroi vetor de entrada DESTA CAMADA com adicao da entrada y0=-1
        Uk = MM * Ye;           % Ativacao (net) dos neuronios da camada de saida
        Ok = Uk;%1./(1+exp(-Uk));   % Saida entre [0,1] (funcao logistica) ***?

        % CALCULO DO ERRO 
        Ek = ytreino(i,:) - Ok';     % erro entre a saida desejada e a saida da rede
        EQ = EQ +(Ek*Ek');
        
        %%% CALCULO DOS GRADIENTES LOCAIS
        Dk = 1; %Ok.*(1 - Ok);   % derivada da sigmoide logistica (camada de saida) ***?
        DDk = Ek; %Ek.*Dk';       % gradiente local (camada de saida) ***?
        
        Di = Yi.*(1 - Yi); % derivada da sigmoide logistica (camada oculta)
        DDi = Di.*(MM(:,2:end)'*DDk');    % gradiente local (camada oculta)

        % AJUSTE DOS PESOS - CAMADA DE SAIDA
        MM = MM + alfa*DDk'*Ye';
        
        % AJUSTE DOS PESOS - CAMADA OCULTA
        WW = WW + alfa*DDi*Xe;
    end   % Fim de uma epoca
    
    % MEDIA DO ERRO QUADRATICO P/ EPOCA
    EQs(epoca) = EQ/ltreino;
    
    %Erro Quadrático Médio da Validação
    EQ=0;
    
    for i=1:lvalid,
        % CAMADA OCULTA
        Xe = horzcat(-1, xvalid(i,:));      % Constroi vetor de entrada com adicao da entrada x0=-1
        Ui = WW * Xe';          % Ativacao (net) dos neuronios da camada oculta
        Yi= 1./(1+exp(-Ui)); % Saida entre [0,1] (funcao logistica)
        
        % CAMADA DE SAIDA
        Ye=[-1; Yi];           % Constroi vetor de entrada DESTA CAMADA com adicao da entrada y0=-1
        Uk = MM * Ye;          % Ativacao (net) dos neuronios da camada de saida
        Ok = Uk; %1./(1+exp(-Uk)); % Saida entre [0,1] (funcao logistica) ***?
        
        % CALCULO DO ERRO 
        Ek = yvalid(i,:) - Ok';           % erro entre a saida desejada e a saida da rede
        EQ = EQ +(Ek*Ek');    
    end
    
    % MEDIA DO ERRO QUADRATICO P/ EPOCA
    EQVs(epoca) = EQ/lvalid;
    if (epoca == analiseErro)
        em1 = sum(EQVs(1:analiseErro))/analiseErro;
    end
    if(mod(epoca, analiseErro) == 0) 
        em2 = sum(EQVs(epoca-analiseErro+1:epoca))/analiseErro;
        if(em2>em1)
            break;
        end
        em1 = em2;
    end    
end   %fim do treinamento
epoca

%%%etapa de teste
EQ=0;
OUT2=[];
for i=1:lteste  
    %camada oculta (hidden)
    %adiciona x0=-1 na entrada
    Xe = horzcat(-1, xteste(i,:));
    %ativa (net)  neuronios da camada oculta
    Ui = WW * Xe';
    %aplica funcao logistica (saida entre 0 e 1) como funcao de ativacao
    Yi= 1./(1+exp(-Ui));

    %camada de saida
    %adiciona y0=-1 na entrada desssa camada
    Ye=[-1; Yi];
    %ativa (net)  neuronios da camada de saida
    Uk = MM * Ye; 
    
    OUT2=[OUT2 Uk]; %armazena saida da rede
    
    %aplica funcao logistica (saida entre 0 e 1)
    %Ok = 1./(1+exp(-Uk));
    El = xteste(i) - Uk;
    EQ = EQ + (El*El);
end

EQ = EQ/lteste

%figure(01);
%hold all;
%plot([1:epoca], EQs);
%title('Epocas x Erro Quadratico Medio Treinamento');
%xlabel('Epoca');
%ylabel('Erro');
%figure(02);
%plot([1:epoca], EQVs);
%title('Epocas x Erro Quadratico Medio Validacao');
%xlabel('Epoca');
%ylabel('Erro');
%hold off;

figure(01);
hold on;
title('Erro Quadrático Medio - Treinamento x Validacao');
p1 = plot([1:epoca], EQs);
set(p1,'Color',[0.4, 0.7, 0.4]);
set(p1,'LineWidth',2); 
p2 = plot([1:epoca], EQVs);
set(p2,'Color',[0.4, 0.4, 0.7]);
set(p2,'LineWidth',2); 
xlabel('Epoca');
ylabel('Erro');
legend('Treinamento', 'Validacao');

figure(03);
hold on;
title('Precos Treinamento x Teste');
p1 = plot(D(407:end,14));  %treinamento
set(p1,'Color',[0.4, 0.7, 0.4]);
set(p1,'LineWidth',2); 
p2 = plot(OUT2); %teste
set(p2,'Color',[0.4, 0.4, 0.7]);
set(p2,'LineWidth',2); 
ylabel('Preco');
xlabel('Casas');
legend('Treinamento', 'Teste');

