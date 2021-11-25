%% Initialize
%target set, state bounds, time horizon, M size of probability array
K = [20, 21];
bounds = [18, 23];
N = 12;
M = 40;

%thermal properties
dt = 5/60;
eta = 0.7;
R = 2;
P = 14;
C = 2;
params.a = exp(-dt/C*R);
params.b = 32;
params.nRP = eta*R*P;

%possible states, controls
S_grid = 0.1;
U_grid = 0.1;
S = min(bounds):S_grid:max(bounds);
U = 0:U_grid:1;

%% generate truncated normal
mu = 0;
sigma = 1;

p = zeros([1 M]);
ints = -2:(4/M):2; %generates M+1 intervals between -1,1;
for i = 1:M
    if i == 1
        p(i) = normcdf(ints(2),mu,sigma);
    elseif i == M
        p(i) = 1 - normcdf(ints(M), mu, sigma);
    else
        p(i) = normcdf(ints(i+1),mu,sigma) - normcdf(ints(i),mu,sigma);
    end
end

w = (ints(1:M) + ints(2:(M+1)))/2; %average the interval endpoints to get simple approximate value
%% backwards recursion
J = zeros([N length(S)]);
S_mesh = zeros([length(U) length(S) length(p)]);
U_mesh = zeros([length(U) length(S) length(p)]);
w_mesh = repmat(reshape(w,[1 1 M]),[11 51]);
p_mesh = repmat(reshape(p,[1 1 M]),[11 51]);

for i = 1:M
    S_mesh(:,:,i) = meshgrid(S,U);
    U_mesh(:,:,i) = meshgrid(U,S)';
end

J(N,:) = cost(S, max(K), min(K));

for i = (N-1):-1:1
    %run the dynamics, snap to the boundary
    xplus_val = dynamics_snap(S_mesh, U_mesh, w_mesh, params, bounds);
    
    %compute the expectation
    %transform values to get corresponding indices for evaluating J_{i+1}
    Jplus_lerp = lerp(xplus_val, S, S_grid, J(i+1,:));
    E_plus = sum(Jplus_lerp .* p_mesh, 3);

    %finish the recursion
    J(i,:) = cost(S, max(K), min(K)) + min(E_plus, [], 1);
end

%clear the workspace of the large meshes for memory purposes
clear Jplus Jplus_lerp Jplus_u Jplus_l
clear S_mesh U_mesh w_mesh p_mesh
clear xplus_val E_plus

%% Monte Carlo Simulations

m = 20000; %number of total trajectories observed

%eparams.n = 1000; %number of samples of J_{i+1} taken for each u \in U
eparams.al = 0.05; %alpha for CVaR
%eparams.estimator = 1;

sysparams.S = S;
sysparams.U = U;
sysparams.K = K;
sysparams.bounds = bounds;
sysparams.params = params;
sysparams.S_grid = S_grid;

sysparams.p = p;
sysparams.w = w;

x_val = [19.5 20.5 21.5];
e_val = [1 2 3 4];
n_val = [10 20 30 40 50 60 70 80 90 100 ...
         200 300 400 500 600 700 800 900 1000 ...
         2500 5000];

for i = 1:length(x_val)
    %loop over inital conditions
    x = x_val(i);
    mu_mat = zeros([length(e_val) length(n_val)]);
    va_mat = zeros([length(e_val) length(n_val)]);

    for j = 1:length(e_val)
        for k = 1:length(n_val)
           eparams.n = n_val(k);
           eparams.estimator = e_val(j);

           [mu, sigma] = mc_run(x, m, eparams, sysparams, J); %sigma is variance
           mu_mat(j,k) = mu;
           va_mat(j,k) = sigma;
        end
    end

    writematrix(mu_mat, 'mu_mat' + string(x) + '.csv')
    writematrix(va_mat, 'va_mat' + string(x) + '.csv')
end