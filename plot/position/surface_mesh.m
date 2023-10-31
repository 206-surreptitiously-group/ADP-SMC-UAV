clc;clear;

%% attitude
% att_ref = csvread('uniform_mc_ref_traj.csv', 1, 0);  % ref
% att_smc = csvread('uniform_mc_test_smc.csv', 1, 0);% smc
% att_rl = csvread('uniform_mc_test_rl.csv', 1, 0); % rl
% 
% fx = att_ref(:, 1);
% fy = att_ref(:, 4);
% 
% figure
% set(gca, 'LooseInset', [0.01,0.01,0.01,0.01]);
% [x,y]=meshgrid(linspace(min(fx),max(fx),50),linspace(min(fy),max(fy),50));
% % plot smc
% fz = att_smc(:, 1);
% z=griddata(fx,fy,fz,x,y);
% mesh(x,y,z,'facecolor','b', 'EdgeColor', 'none');hold on;
% % plot rl
% fz = att_rl(:, 1);
% z=griddata(fx,fy,fz,x,y);
% mesh(x,y,z,'facecolor','r', 'EdgeColor', 'none');hold on;
% legend('smc', 'rl');
% title('attitude');
% grid on;

%% position
pos_ref = csvread('./surface_mesh/uniform_mc_test_traj.csv', 1, 0);  % ref
pos_smc = csvread('./surface_mesh/uniform_mc_test_smc.csv', 1, 0);% smc
pos_new1_260 = csvread('./surface_mesh/uniform_mc_test_rl_pos_new1-260.csv', 1, 0); % rl
% pos_joint2_120 = csvread('./surface_mesh/uniform_mc_test_rl_joint2-120.csv', 1, 0); 
% pos_debug4_4070 = csvread('./surface_mesh/uniform_mc_test_rl_pos_debug4-4070.csv', 1, 0);

fx = pos_ref(:, 1);
fy = pos_ref(:, 5);

figure;
set(gca, 'LooseInset', [0.01, 0.01, 0.01, 0.01]);
[x, y] = meshgrid(linspace(min(fx), max(fx), 50),linspace(min(fy), max(fy), 50));

% plot smc
fz = pos_smc(:, 1);
z = griddata(fx,fy,fz,x,y);
mesh(x, y, z, 'facecolor', 'b', 'EdgeColor', 'none'); hold on;

% plot rl
fz = pos_new1_260(:, 1);
z=griddata(fx,fy,fz,x,y);
mesh(x,y,z,'facecolor','r', 'EdgeColor', 'none'); hold on;
