function x_plussnap = dynamics_snap(x, u, w, params, bounds)
    x_plussnap = dynamics(x,u,w,params);
    x_plussnap(x_plussnap > max(bounds)) = max(bounds);
    x_plussnap(x_plussnap < min(bounds)) = min(bounds);
end

