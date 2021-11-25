mu1 = readmatrix('mu_mat19.5.csv');
mu2 = readmatrix('mu_mat20.5.csv');

va1 = readmatrix('va_mat19.5.csv');
va2 = readmatrix('va_mat20.5.csv');
n = [10 20 30 40 50 60 70 80 90 100 ...
         200 300 400 500 600 700 800 900 1000];

%subplots for each initial condition
figure(1)
subplot(2,1,1);
plot(n, mu1(1,1:19), n, mu1(2,1:19), n, mu1(3,1:19), n, mu1(4,1:19))
ylabel('Mean')
lgd = legend('Mean','Max','CVaR_{0.05}', 'EVT');
title('x_0 = 19.5 C, 20000 trajectories')

subplot(2,1,2); 
plot(n, va1(1,1:19), n, va1(2,1:19), n, va1(3,1:19), n, va1(4,1:19))
xlabel('n - number of samples')
ylabel('Variance')

figure(2)
subplot(2,1,1);
plot(n, mu2(1,1:19), n, mu2(2,1:19), n, mu2(3,1:19), n, mu2(4,1:19))
ylabel('Mean')
lgd = legend('Mean','Max','CVaR_{0.05}', 'EVT');
title('x_0 = 20.5 C, 20000 trajectories')

subplot(2,1,2); 
plot(n, va2(1,1:19), n, va2(2,1:19), n, va2(3,1:19), n, va2(4,1:19))
xlabel('n - number of samples')
ylabel('Variance')