clear; clc;

%% load file
smc_param = csvread('./response/opt_smc_param.csv', 1, 0);
ref_cmd = csvread('./response/ref_cmd.csv', 1, 0);
uav_state = csvread('./response/uav_state.csv', 1, 0);

smc_param(1, :) = [];

%% SMC parameter
shape = size(smc_param);
figure(1);
set(gcf, 'unit', 'centimeters', 'position', [7 6 10 8]);
for i = 1 : shape(2)
    plot((1: length(smc_param(:, i)))/1e2*2, smc_param(:, i)); hold on;
end
% legend('k11', 'k12', 'k13', 'k21', 'k22', 'k23', 'gamma', 'lambda');
grid on;

%% state response
k = 180 / pi;
%% phi
figure(2);
set(gcf, 'unit', 'centimeters', 'position', [7 6 10 8]);
plot(ref_cmd(:, 1), k * ref_cmd(:, 8), 'red', 'linewidth', 2); hold on;
plot(uav_state(:, 1), k * uav_state(:, 8), 'blue', 'linewidth', 2);
grid on;

%% theta
figure(3);
set(gcf, 'unit', 'centimeters', 'position', [7 6 10 8]);
plot(ref_cmd(:, 1), k * ref_cmd(:, 9), 'red', 'linewidth', 2); hold on;
plot(uav_state(:, 1), k * uav_state(:, 9), 'blue', 'linewidth', 2);
grid on;

%% psi
figure(4);
set(gcf, 'unit', 'centimeters', 'position', [7 6 10 8]);
plot(ref_cmd(:, 1), k * ref_cmd(:, 10), 'red', 'linewidth', 2); hold on;
plot(uav_state(:, 1), k * uav_state(:, 10), 'blue', 'linewidth', 2);
grid on;

%% dot_phi
figure(5);
set(gcf, 'unit', 'centimeters', 'position', [7 6 10 8]);
plot(ref_cmd(:, 1), k * ref_cmd(:, 11), 'red', 'linewidth', 2); hold on;
plot(uav_state(:, 1), k * uav_state(:, 14), 'blue', 'linewidth', 2);
grid on;

%% dot_theta
figure(6);
set(gcf, 'unit', 'centimeters', 'position', [7 6 10 8]);
plot(ref_cmd(:, 1), k * ref_cmd(:, 12), 'red', 'linewidth', 2); hold on;
plot(uav_state(:, 1), k * uav_state(:, 15), 'blue', 'linewidth', 2);
grid on;

%% dot_psi
figure(7);
set(gcf, 'unit', 'centimeters', 'position', [7 6 10 8]);
plot(ref_cmd(:, 1), k * ref_cmd(:, 13), 'red', 'linewidth', 2); hold on;
plot(uav_state(:, 1), k * uav_state(:, 16), 'blue', 'linewidth', 2);
grid on;
