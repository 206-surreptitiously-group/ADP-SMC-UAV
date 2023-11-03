clear; clc;

%% load file
smc_param_pos = csvread('./response/opt_smc_param_pos.csv', 1, 0);
smc_param_att = csvread('./response/opt_smc_param_att.csv', 1, 0);
ref_cmd = csvread('./response/ref_cmd.csv', 1, 0);
uav_state = csvread('./response/uav_state.csv', 1, 0);

smc_param_pos(1, :) = [];
smc_param_att(1, :) = [];

%% SMC parameter
shape = size(smc_param_pos);
figure(1);
set(gcf, 'unit', 'centimeters', 'position', [7 6 10 8]);
for i = 1 : shape(2)
    plot((1: length(smc_param_pos(:, i)))/1e2*2, smc_param_pos(:, i), 'linewidth', 2); hold on;
end
set(gca, 'Fontname', 'Times New Roman','FontSize',12);
% legend('k11', 'k12', 'k13', 'k21', 'k22', 'k23', 'gamma', 'lambda');
grid on;

shape = size(smc_param_att);
figure(2);
set(gcf, 'unit', 'centimeters', 'position', [7 6 10 8]);
for i = 1 : shape(2)
    plot((1: length(smc_param_att(:, i)))/1e2*2, smc_param_att(:, i), 'linewidth', 2); hold on;
end
set(gca, 'Fontname', 'Times New Roman','FontSize',12);
% legend('k11', 'k12', 'k13', 'k21', 'k22', 'k23', 'gamma', 'lambda');
grid on;

%% state response
%% x
figure(3);
set(gcf, 'unit', 'centimeters', 'position', [7 6 10 8]);
plot(ref_cmd(:, 1), ref_cmd(:, 2), 'red', 'linewidth', 2); hold on;
plot(uav_state(:, 1), uav_state(:, 2), 'blue', 'linewidth', 2);
set(gca, 'Fontname', 'Times New Roman','FontSize',12);
grid on;

%% y
figure(4);
set(gcf, 'unit', 'centimeters', 'position', [7 6 10 8]);
plot(ref_cmd(:, 1), ref_cmd(:, 3), 'red', 'linewidth', 2); hold on;
plot(uav_state(:, 1), uav_state(:, 3), 'blue', 'linewidth', 2);
set(gca, 'Fontname', 'Times New Roman','FontSize',12);
grid on;

%% z
figure(5);
set(gcf, 'unit', 'centimeters', 'position', [7 6 10 8]);
plot(ref_cmd(:, 1), ref_cmd(:, 4), 'red', 'linewidth', 2); hold on;
plot(uav_state(:, 1), uav_state(:, 4), 'blue', 'linewidth', 2);
set(gca, 'Fontname', 'Times New Roman','FontSize',12);
grid on;

%% vx
figure(6);
set(gcf, 'unit', 'centimeters', 'position', [7 6 10 8]);
plot(ref_cmd(:, 1), ref_cmd(:, 5), 'red', 'linewidth', 2); hold on;
plot(uav_state(:, 1), uav_state(:, 5), 'blue', 'linewidth', 2);
set(gca, 'Fontname', 'Times New Roman','FontSize',12);
grid on;

%% vy
figure(7);
set(gcf, 'unit', 'centimeters', 'position', [7 6 10 8]);
plot(ref_cmd(:, 1), ref_cmd(:, 6), 'red', 'linewidth', 2); hold on;
plot(uav_state(:, 1), uav_state(:, 6), 'blue', 'linewidth', 2);
set(gca, 'Fontname', 'Times New Roman','FontSize',12);
grid on;

%% vz
figure(8);
set(gcf, 'unit', 'centimeters', 'position', [7 6 10 8]);
plot(ref_cmd(:, 1), ref_cmd(:, 7), 'red', 'linewidth', 2); hold on;
plot(uav_state(:, 1), uav_state(:, 7), 'blue', 'linewidth', 2);
set(gca, 'Fontname', 'Times New Roman','FontSize',12);
grid on;
