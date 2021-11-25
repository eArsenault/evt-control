function x_tplus = dynamics(x_t, u_t, w_t, params)
    x_tplus = (params.a * x_t) + (1 - params.a) * (params.b - params.nRP * u_t) + w_t;
end

