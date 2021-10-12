%%% q1 redes neurais

clear all; close all; clc;

load('ex3data1.mat');
Dn=X;
[LinD ColD]=size(X);

%embaralha atributos e saidas
I=randperm(LinD); %LinD ~ qtde_treino
Dn=Dn(I,:);       %Dn ~ atributos ~ x
alvos=T(I,:);     %alvos ~ saidas ~ y -> embaralha saida p/ manter correspondencia com entrada

%tamanho conj de treinamento, validacao e teste
qtreino=4000;
qvalid=500;
qteste=500;

%%%dividindo conjuntos

%treinamento - entrada e saida
xtreino = Dn(1:qtreino,:);
ytreino = alvos(1:qtreino,:); 
[ltreino ctreino]=size(xtreino);  %num linhas e colunas

%validação
xvalid = Dn(qtreino+1:qtreino+qvalid,:); 
yvalid = alvos(qtreino+1:qtreino+qvalid,:); 
[lvalid cvalid]=size(xvalid);  %num linhas e colunas

%teste
xteste = Dn(qtreino+qvalid+1:end,:); 
yteste = alvos(qtreino+qvalid+1:end,:);  
[lteste cteste]=size(xteste);  %num linhas e colunas


%%%definindo arquitetura
epocas = 500; % *** 1000
alfa = 0.05;

%num neuronios
qNeuroCO = 21;   %num camada oculta   = 2p+1 -> heuristica de hetch-nielsen no=2ni+1 *** 80
qNeuroCS = 10;   %num camada de saida = num classes

%intervalo de análise de erro    %aumento da media dos erros nos últimos x% da qtde de epocas?
analiseErro = floor(epocas*0.05);

%matrizes de pesos - gera aleatoriamente
WW=0.1*rand(qNeuroCO,ctreino+1);  %pesos entrada para camada oculta
MM=0.1*rand(qNeuroCS,qNeuroCO+1); %pesos camada oculta para camada de saida ***** 0.1* ?

%pesos camada oculta
%Ww=rand(qNeuroCO,qNeuroCS+1);% ????? *0.1
%pesos camada de saida
%Mm=rand(qNeuroCS,ctreino+1); % ????? *0.1


%%%etapa de treinamento
for epoca=1:epocas
    %epoca

    %embaralha dados 
    I=randperm(ltreino); 
    xtreino=xtreino(I,:); 
    ytreino=ytreino(I,:);   % Embaralha vetores de treinamento e saidas desejadas
    
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
        Ok = 1./(1+exp(-Uk));   % Saida entre [0,1] (funcao logistica)

        % CALCULO DO ERRO 
        Ek = ytreino(i,:) - Ok';     % erro entre a saida desejada e a saida da rede
        EQ = EQ +(Ek*Ek');
        
        %%% CALCULO DOS GRADIENTES LOCAIS
        Dk = Ok.*(1 - Ok);   % derivada da sigmoide logistica (camada de saida)
        DDk = Ek.*Dk';       % gradiente local (camada de saida)
        
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
        Ok = 1./(1+exp(-Uk)); % Saida entre [0,1] (funcao logistica)
        
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
    %aplica funcao logistica (saida entre 0 e 1)
    Ok = 1./(1+exp(-Uk));
    
    OUT2=[OUT2 Ok];       %armazena saida da rede
end

%taxa acerto
count_OK=0;
for i=1:lteste,
    [T2max iT2max]=max(yteste(i,:));    %indice da saida desejada de maior valor
    [OUT2_max iOUT2_max]=max(OUT2(:,i));%indice do neuronio cuja saida eh a maior
    if iT2max==iOUT2_max,   %se os dois indices coincidem  eh acerto
        count_OK=count_OK+1;
    end
end 
txAcerto=100*(count_OK/lteste)

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

figure(01)
hold on;
title('Erro Quadratico Medio - Treinamento x Validação');
p1 = plot([1:epoca], EQs);
set(p1,'Color',[0.4, 0.7, 0.4]);  %cor azul
set(p1,'LineWidth',2); 
p2 = plot([1:epoca], EQVs);
set(p2,'Color',[0.4, 0.4, 0.7]);  %cor verde
set(p2,'LineWidth',2); 
ylabel('Erro');
xlabel('Epocas');
legend('Treinamento', 'Validacao');
