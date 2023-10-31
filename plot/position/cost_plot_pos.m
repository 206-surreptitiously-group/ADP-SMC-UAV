clear; clc;

%% load evaluation files: phase1 to phase3
p1_eval = csvread('./phase1_train/test_record.csv', 1, 0);
p2_eval = csvread('./phase2_train/test_record.csv', 1, 0);
p3_eval = csvread('./phase3_train/test_record.csv', 1, 0);
p1_eval = p1_eval(:, 2); p1_eval = p1_eval(p1_eval < 0);
p2_eval = p2_eval(:, 2); p2_eval = p2_eval(p2_eval < 0);
p3_eval = p3_eval(:, 2); p3_eval = p3_eval(p3_eval < 0);
len_e1 = length(p1_eval);
len_e2 = length(p2_eval);
len_e3 = length(p3_eval);
l_e1 = 1 : len_e1;
l_e2 = (1 : len_e2) + len_e1;
l_e3 = (1 : len_e3) + len_e1 + len_e2;

%% load training files: phase1 to phase3
p1_train = csvread('./phase1_train/sumr_list.csv', 1, 0);
p2_train = csvread('./phase2_train/sumr_list.csv', 1, 0);
p3_train = csvread('./phase3_train/sumr_list.csv', 1, 0);
p1_train = p1_train(:, 2); p1_train = p1_train(p1_train < 0);
p2_train = p2_train(:, 2); p2_train = p2_train(p2_train < 0);
p3_train = p3_train(:, 2); p3_train = p3_train(p3_train < 0);
len_t1 = length(p1_train);
len_t2 = length(p2_train);
len_t3 = length(p3_train);
l_t1 = 1 : len_t1;
l_t2 = (1 : len_t2) + len_t1;
l_t3 = (1 : len_t3) + len_t1 + len_t2;

%% plot evaluation
figure(1);
set(gca, 'LooseInset', [0, 0, 0, 0]);
set(gcf, 'unit', 'centimeters', 'position', [7 6 16 8]);
set(gca, 'Fontname', 'Times New Roman', 'FontSize', 12);
plot(l_e1 / 1e3, p1_eval / 1e3, 'red', 'linewidth', 2); hold on;
plot(l_e2 / 1e3, p2_eval / 1e3, 'green', 'linewidth', 2); hold on;
plot(l_e3 / 1e3, p3_eval / 1e3, 'blue', 'linewidth', 2); hold on;
grid on;

figure(2);
set(gca, 'LooseInset', [0, 0, 0, 0]);
set(gcf, 'unit', 'centimeters', 'position', [7 6 8 4]);
set(gca, 'Fontname', 'Times New Roman', 'FontSize', 12);
plot(l_e3 / 1e2, p3_eval / 1e2, 'blue', 'linewidth', 1);
set(gca, 'ylim', [-100, 0] / 1e2, 'ytick', [-100, -75, -50, -25, 0] / 1e2);
set(gca, 'xtick', []);
grid on;

%% plot training
figure(3);
set(gca, 'LooseInset', [0, 0, 0, 0]);
set(gcf, 'unit', 'centimeters', 'position', [7 6 16 8]);
set(gca, 'Fontname', 'Times New Roman', 'FontSize', 12);
plot(l_t1 / 1e3, p1_train / 1e2, 'red', 'linewidth', 1); hold on;
plot(l_t2 / 1e3, p2_train / 1e2, 'green', 'linewidth', 1); hold on;
plot(l_t3 / 1e3, p3_train / 1e2, 'blue', 'linewidth', 1); hold on;
grid on;

figure(4);
set(gca, 'LooseInset', [0, 0, 0, 0]);
set(gcf, 'unit', 'centimeters', 'position', [7 6 8 4]);
set(gca, 'Fontname', 'Times New Roman', 'FontSize', 12);
plot(l_t3 / 1e2, p3_train / 1e2, 'blue', 'linewidth', 1);
% set(gca, 'ylim', [-100, 0] / 1e2, 'ytick', [-100, -75, -50, -25, 0] / 1e2);
set(gca, 'xtick', []);
grid on;