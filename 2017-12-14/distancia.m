function d=distancia(X,Centroides)
  [linX,colX]= size(X);
  [linCentr,colCentr]= size(Centroides);
    
  for cont=1:colX
      Ce{cont}= repmat (X(:, cont), 1 ,linCentr);
      Di{cont}= repmat (Centroides(:, cont), 1 ,linX);
  end
  
  S=zeros(linX, linCentr);
  for cont=1:colX
      S=S+(Ce{cont}-Di{cont}').^ 2 ;
  end
  d=sqrt(S);
end