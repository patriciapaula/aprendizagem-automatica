function [W] = logisticRegressionRegularized(X, Y, W0, alfa, lambda, epocas)
    [v , n] = size(X);    
    %acha os coeficientes por gradiente descendente estocástico
    W = W0;
    for j=1:epocas
        for i=1:n
            %regra de aprendizagem para regressão logística
            yi = 1 ./ ( 1 + exp( -1 * W'*X(:,i)));
            ei = Y(i) - yi;
            %calcula cada um dos coeficientes w0, w1, w2
            for k=1:v
                if (k==0)
                    W(k) = W(k) + alfa * (ei* X(k,i));		
                else
                    W(k) = W(k) + alfa * (ei* X(k,i) - lambda * W(k));		
                end
            end		
        end
        %permutação dos dados
        idx = randperm(n);
        X = X(:,idx);
        Y = Y(idx);
    end
end