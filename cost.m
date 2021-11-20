function c = cost(x,Kmax,Kmin)
    row1 = x - Kmax;
    row2 = Kmin - x; 
    c = max(max(row1,row2), 0);
end