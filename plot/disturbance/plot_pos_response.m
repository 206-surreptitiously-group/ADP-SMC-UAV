clear; clc;

%% load file
smc_param_pos = csvread('./response/opt_param_pos.csv', 1, 0);
smc_param_att = csvread('./response/opt_param_att.csv', 1, 0);
ref_cmd = csvread('./response/ref_cmd.csv', 1, 0);
uav_state = csvread('./response/uav_state.csv', 1, 0);
observer = csvread('./response/observe.csv', 1, 0);

%% SMC parameter
% draw_param(smc_param_pos, smc_param_att);

%% state response
draw_state(ref_cmd, uav_state);

%% observer
% draw_observer(observer);

function draw_param(p_pos, p_att)
    p_pos(1, :) = [];
    p_att(1, :) = [];
    shape = size(p_pos);
    figure(1);
    set(gcf, 'unit', 'centimeters', 'position', [7 6 10 8]);
    for i = 1 : shape(2)
        plot((1: length(p_pos(:, i)))/1e2*2, p_pos(:, i), 'linewidth', 2); hold on;
    end
    set(gca, 'Fontname', 'Times New Roman','FontSize',12);
    % legend('k11', 'k12', 'k13', 'k21', 'k22', 'k23', 'gamma', 'lambda');
    grid on;

    shape = size(p_att);
    figure(2);
    set(gcf, 'unit', 'centimeters', 'position', [7 6 10 8]);
    for i = 1 : shape(2)
        plot((1: length(p_att(:, i)))/1e2*2, p_att(:, i), 'linewidth', 2); hold on;
    end
    set(gca, 'Fontname', 'Times New Roman','FontSize',12);
    % legend('k11', 'k12', 'k13', 'k21', 'k22', 'k23', 'gamma', 'lambda');
    grid on;
end

function draw_state(ref_cmd, uav_state)
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
end

function draw_observer(observe)
    t = observe(:, 1);
    figure(9);
    set(gcf, 'unit', 'centimeters', 'position', [7 6 10 8]);
    plot(t, observe(:, 2), 'r', 'linewidth', 2);hold on;
    plot(t, observe(:, 5), 'b', 'linewidth', 2);
    grid on;
    
    figure(10);
    set(gcf, 'unit', 'centimeters', 'position', [7 6 10 8]);
    plot(t, observe(:, 3), 'r', 'linewidth', 2);hold on;
    plot(t, observe(:, 6), 'b', 'linewidth', 2);
    grid on;
    
    figure(11);
    set(gcf, 'unit', 'centimeters', 'position', [7 6 10 8]);
    plot(t, observe(:, 4), 'r', 'linewidth', 2);hold on;
    plot(t, observe(:, 7), 'b', 'linewidth', 2);
    grid on;
end