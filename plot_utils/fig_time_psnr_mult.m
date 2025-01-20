set(0,'defaulttextInterpreter','latex');

x_vec = [];
y_vec = [];
exp_name_a = 'ct_cset_';

%% results inpainting, demosaicing
%% Poisson noise
% y_vec = [10, 25];
% level = "4";
% exp_name_a = 'ct_pl_cset_';
% exp_name_b = '_n0.1_denoising';

%% inpainting
level = "4";
exp_name_b = '_n0.1_inpainting0.5';
y_vec = [5, 35];

%% demosaicing 
% level = "4";
% exp_name_b = "_n0.1_demosaicing";
% y_vec = [2, 31];

%% blur
% level = "4";
% exp_name_b = "_n0.1_blur";
% y_vec = [1, 26];
% x_vec = [0, 2];

%% main curves
% func_plot_mult(@func_plot_PnP_init, @custom_plt, x_vec, y_vec, exp_name_a, exp_name_b, level);
% file_name = (exp_name_a + "mult_many" + exp_name_b) +'_PnP.eps';
% saveas(gcf, file_name, 'epsc');

%% cmp denoisers
func_plot_mult(@func_plot_PnP_init_CMP_denoisers, @custom_plt, x_vec, y_vec, exp_name_a, exp_name_b, level);
% file_name = (exp_name_a + "mult_Denoisers" + exp_name_b) +'_PnP.eps';
% saveas(gcf, file_name, 'epsc');

%% cmp ours
func_plot_mult(@func_plot_PnP_init_CMP_ours, @custom_plt, x_vec, y_vec, exp_name_a, exp_name_b, level);
% file_name = (exp_name_a + "mult_Ours" + exp_name_b) +'_PnP.eps';
% saveas(gcf, file_name, 'epsc');

%% State of the art
func_plot_mult(@func_plot_PnP_init_SA, @custom_plt, x_vec, y_vec, exp_name_a, exp_name_b, level);
% file_name = (exp_name_a + "mult_SA" + exp_name_b) +'_PnP.eps';
% saveas(gcf, file_name, 'epsc');

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


% movefile(file_name, "../../Presentations/last_simu/"+file_name);