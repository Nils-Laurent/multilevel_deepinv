function [] = func_plot_mult(fn_plot, custom_plt, x_vec, y_vec, exp_name_a, exp_name_b, level)
%%
win = "sinc";
suffix = "costs";
%% set figure PnP
f = figure;
f.Position = [100, 300, 1800, 350];
tcl = tiledlayout(1,4);
% h(1) = subplot(1,4,1); % leftmost plot
nexttile(tcl);
grid on;
exp_name = exp_name_a + "0" + exp_name_b;
hold on;
fn_plot(custom_plt, exp_name, level, win, suffix);
hold off;
xlabel('time (s)');
ylabel('PSNR');
if length(y_vec) > 0
    ylim(y_vec);
end
if length(x_vec) > 0
    xlim(x_vec)
end

for k=2:4
    % h(k) = subplot(1,4,k); % leftmost plot
    nexttile(tcl);
    exp_name = exp_name_a + string(k-1) + exp_name_b;
    grid on;
    hold on;
    fn_plot(custom_plt, exp_name, level, win, suffix);
    hold off;
    if length(y_vec) > 0
        ylim(y_vec);
    end
    if length(x_vec) > 0
        xlim(x_vec)
    end
end
legend('Location','eastoutside');
end

