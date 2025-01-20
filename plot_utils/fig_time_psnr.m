set(0,'defaulttextInterpreter','latex');

%% results inpainting, demosaicing
win = "sinc";

level = "4";
exp_name = "ct_pl_cset_3_n0.1_denoising";

% level = "4";
% exp_name = "ct_cset_3_n0.1_inpainting0.5";

% level = "4";
% exp_name = "ct_cset_1_n0.1_demosaicing";

% level = "3";
% exp_name = "ct_cset_3_n0.1_mri";

% level = "4";
% exp_name = "ct_cset_3_n0.1_blur";

%%
suffix = "costs";
%% set figure PnP
figure;
grid on;
hold on;
func_plot_PnP_init(@custom_plt, exp_name, level, win, suffix);
hold off;
xlabel('time (s)');
ylabel('PSNR');
legend('Location','southeast');
% file_name = exp_name+'_PnP.eps';
% saveas(gcf, file_name, 'epsc');
% movefile(file_name, "../../Presentations/last_simu/"+file_name);

%% set figure PnP PGD
% figure;
% grid on;
% hold on;
% func_plot_PnP_PGD_init(@custom_plt, exp_name, level, win, suffix);
% ylabel('PSNR');
% xlabel('time (s)');
% legend('Location','southeast');
% saveas(gcf,exp_name+'_PnP_PGD.eps', 'epsc');
% movefile(exp_name+'_PnP_PGD.eps', "../../Presentations/last_simu/"+exp_name+'_PnP_PGD.eps')

%% set figure PnP NE
% figure;
% grid on;
% hold on;
% func_plot_PnP_NE_init(@custom_plt, exp_name, level, win, suffix);
% ylabel('PSNR');
% xlabel('time (s)');
% legend('Location','southeast');
% saveas(gcf,exp_name+'_PnP_NE.eps', 'epsc');
% movefile(exp_name+'_PnP_NE.eps', "../../Presentations/last_simu/"+exp_name+'_PnP_NE.eps')

%% set figure RED
% figure;
% grid on;
% hold on;
% % func_plot_RED(@custom_plt, exp_name, level, win, suffix);
% func_plot_RED_init(@custom_plt, exp_name, level, win, suffix);
% % yscale log;
% % xscale log;
% ylabel('PSNR');
% xlabel('time (s)');
% legend('Location','southeast');

%% custom plot
function custom_plt(def_name, def_form, lgd_name)
    try
       z = load("out\" + def_name + ".mat");
    catch ME
       return;
    end
    
    x_vec = cumsum(z.time)/1000; % ms to seconds
    y_vec = z.psnr;
    imp_plot(x_vec, y_vec, def_form, lgd_name);
end
