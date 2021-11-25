function [mean_cost,var_cost] = mc_run(x, m, eparams, sysparams, J)
sp = sysparams;

cost_arr = zeros([1 m]);
for k = 1:m
    total_cost = 0;

    for i = 1:11
        %evaluate current cost, add to total
        total_cost = total_cost + cost(x, max(sp.K), min(sp.K));
        u_min = 1;
        Z_min = 100;
        eparams.t = i;

        %action selection
        for j = 1:10
            u = sp.U(j);

            %sample w_i n times, run through dynamics
            w_mc = sp.w(sample_p(sp.p, eparams.n));
            xplus_mc = dynamics_snap(x, u, w_mc, sp.params, sp.bounds);

            %observe Z_{i+1} via linear interpolation along the grid
            Z = lerp(xplus_mc, sp.S, sp.S_grid, sp.S, max(sp.K), min(sp.K)) + J(i+1,:));

            %apply estimator
            switch eparams.estimator
                case 2
                    Z_eval = estimator_max(Z, eparams);
                case 3
                    Z_eval = estimator_cvar(Z, eparams);
                case 4
                    Z_eval = estimator_evt(Z, eparams);
                otherwise
                    Z_eval = estimator_mean(Z, eparams);
            end

            %evaluate running min
            if Z_eval < Z_min
                u_min = u;
                Z_min = Z_eval;
            end
        end

        %update x according to environment, continue looping
        w_1 = sp.w(sample_p(sp.p,1));
        x = dynamics_snap(x, u_min, w_1, sp.params, sp.bounds);
    end

    cost_arr(k) = total_cost;
end

mean_cost = mean(cost_arr);
var_cost = std(cost_arr)^2;
end