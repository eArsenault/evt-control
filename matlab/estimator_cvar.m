function est = estimator_cvar(Z, eparams)
var = quantile(Z, 1 - eparams.al);
ind = (Z >= var);       % ind(i) = 1 if Z(i) >= var, ind(i) = 0 if Z(i) < var

if eparams.al == 0 
    est = max(Z); 
else 
    est = var + sum((Z-var).*ind) / (eparams.n * eparams.al);
end