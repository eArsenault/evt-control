function est = estimator_evt(Z, eparams)
    X = sort(Z);
    var = quantile(X, 1 - eparams.al);
    [~,i] = min(abs(X-var)); %snap our quantile estimate to an actual value

    K = eparams.n - i;
    z_mk = X(i);

    M1 = (1/K) * sum( (log(X(i:end)) - log(X(i))).^1);
    M2 = (1/K) * sum( (log(X(i:end)) - log(X(i))).^2);
    
    ga = M1 + 1 - (1/2) / (1 - (M1^2)/M2); %tail parameter
    a = z_mk * M1 * (1 - ga + M1); %ga_ = ga - M1

    Z_max =  2 + 2 * (12 - eparams.t);
    aparams.g = ga;
    aparams.a = a;
    aparams.M = eparams.n;
    aparams.K = K;
    aparams.z = z_mk;
    %if eparams.t == 1
    %    display(aparams)
    %end

    if (ga == 0) || (ga == 1) || ((ga < 0) && (z_mk > -1/ga)) || (isnan(ga)) %different antiderivatives for these vals
        %shouldn't make much of a difference for this napkin math
        est = max(X);
    elseif ((ga < 0) && (Z_max < -1/ga)) || (ga > 0) %H_ga defined for y in (0, 1/(max(0,-ga))
        y1 = z_mk;
        y2 = Z_max;
        est = antiderivative(y2,aparams) - antiderivative(y1,aparams);
    else 
        y1 = z_mk;
        y2 = -1/ga;
        est = antiderivative(y2,aparams) - antiderivative(y1,aparams);
    end
end