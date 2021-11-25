function state_new = snap_boundary(state, bounds)
    if state > max(bounds)
        state_new = max(bounds);
    elseif state < min(bounds)
        state_new = min(bounds);
    else
        state_new = state;
    end
end