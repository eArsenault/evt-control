function J_lerp = lerp(x, S, dS, J)
    x_l = cast((floor(x * 10)/10 - (min(S) - dS))/dS, 'uint8'); %lower S indices
    x_u = cast((ceil(x * 10)/10 - (min(S) - dS))/dS, 'uint8'); %upper S indices

    J_l = J(x_l);
    J_u = J(x_u);
    
    %linearly interpolate
    J_lerp = J_l .* (1 - (x - S(x_l))/dS) + J_u .* ((x - S(x_l))/dS);
end