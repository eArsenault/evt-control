function val = antiderivative(y,p)
    val  = (p.K * p.a^(1/p.g) / (p.M * (p.g - 1))) * (y + p.a - p.g * p.z) / (p.g * (y - p.z) + p.a) ^ (1/p.g); 
end