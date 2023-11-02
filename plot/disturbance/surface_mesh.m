clc;clear;

%% load file
xy = csvread('./systemic_mesh.csv', 1, 0);
smc_no = csvread('./systemic_mesh_result_fntsmc_noobs.csv', 1, 0);
smc_o = csvread('./systemic_mesh_result_fntsmc_obs.csv', 1, 0);
rl_no = csvread('./systemic_mesh_result_rl_noobs.csv', 1, 0);
rl_o = csvread('./systemic_mesh_result_rl_obs.csv', 1, 0);

fx = xy(:, 1);
fy = xy(:, 5);

%% plot

set(gca, 'LooseInset', [0.01, 0.01, 0.01, 0.01]);
[x, y] = meshgrid(linspace(min(fx), max(fx), 50),linspace(min(fy), max(fy), 50));

% smc no obs
figure(1);
fz1 = smc_no(:, 1);
z = griddata(fx, fy, fz1, x, y);
mesh(x, y, z, 'facecolor', 'b', 'EdgeColor', 'none'); hold on;

% smc obs
fz2 = smc_o(:, 1);
z = griddata(fx, fy, fz2, x, y);
mesh(x, y, z,'facecolor','r', 'EdgeColor', 'none'); hold on;

% rl no obs
fz3 = rl_no(:, 1);
z = griddata(fx, fy, fz3, x, y);
mesh(x, y, z, 'facecolor', 'g', 'EdgeColor', 'none'); hold on;

% rl obs
fz4 = rl_o(:, 1);
z = griddata(fx, fy, fz4, x, y);
mesh(x, y, z, 'facecolor', 'yellow', 'EdgeColor', 'none'); hold on;

grid on;
% legend('smc-no', 'smc-obs', 'rl-no', 'rl-obs');
