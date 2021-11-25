function index = sample_p(p,n)
    y = rand([1 n]);
    cdf = cumsum(p);

    [~,T] = histc(y,cdf);
    index = T+1;
end