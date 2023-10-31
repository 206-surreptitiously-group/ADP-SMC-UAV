clc;clear;

%% attitude
att_ref = csvread('./surface_mesh/uniform_mc_ref_traj.csv', 1, 0);  % ref
att_smc = csvread('./surface_mesh/uniform_mc_test_smc.csv', 1, 0);% smc
att_rl = csvread('./surface_mesh/uniform_mc_test_rl.csv', 1, 0); % rl

fx = att_ref(:, 1) * 180 / pi;
fy = att_ref(:, 4);

figure
set(gca, 'LooseInset', [0.01, 0.01, 0.01, 0.01]);
[x, y] = meshgrid(linspace(min(fx), max(fx), 50), linspace(min(fy), max(fy), 50));
% plot smc
fz = att_smc(:, 1);
z=griddata(fx, fy, fz, x, y);
mesh(x, y, z, 'facecolor', 'b', 'EdgeColor', 'none'); hold on;
% plot rl
fz = att_rl(:, 1);
z = griddata(fx, fy, fz, x, y);
mesh(x, y, z, 'facecolor', 'r', 'EdgeColor', 'none'); hold on;
% legend('smc', 'rl');
% title('attitude');
grid on;
