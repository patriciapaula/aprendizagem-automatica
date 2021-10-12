function [V newX D] = mPCA(X)
    %Subtraindo a média
    X = bsxfun(@minus, X, mean(X,1));
    %Matriz de Covariância
    C = cov(X);
    %Autovetores e Autovalores
    [V D] = eig(C);
    %Ordenando em ordem decrescente
    [D order] = sort(diag(D), 'descend');
    V = V(:,order);

    newX = X*V;
end
