set(0,'defaulttextInterpreter','latex');

%% results inpainting, demosaicing
win = "sinc";

level = "4";
exp_name = "ct_pl_cset_3_n0.1_denoising";

level = "4";
exp_name = "ct_cset_3_n0.1_inpainting0.5";

% level = "4";
% exp_name = "ct_cset_3_n0.1_demosaicing";

% level = "3";
% exp_name = "ct_cset_3_n0.1_mri";

% level = "4";
% exp_name = "ct_cset_3_n0.1_blur";

%%
suffix = "costs";
% sz_vec = ["0.1", "0.2", "0.3", "1.1", "1.2"];
% sz_vec = 0.1:0.1:0.9;
sz_vec = [0.1, 0.3, 0.6, 0.9];
gp_vec = [0.02, 0.05, 0.08, 0.11];

p_str = ["noname"];
id = 1;
for gp = gp_vec
    for sz = sz_vec
        str_id = ("gp" + gp + "sz" + sz);
        p_str(id) = str_id;
        id = id + 1;
    end
end

pref = "FPAR_";
ml_part = string(level)+"L"+win + '_';

%% DRUNet and GS (sigma_denoiser and stepsize params)
% % set figure PnP-DRUNet
% suffix_ = exp_name + '_PnP_' + suffix;
% fun_plot_PnP_FIXED_PAR(pref, suffix_, p_str, "\gamma = ");
% title("PnP-DRUNet");
% saveas(gcf, "FPAR_PnP-DRUNet", 'epsc');
% 
% % set figure PnP-ML-DRUNet init
% suffix_ = exp_name + '_PnP_ML_INIT_' + ml_part + suffix;
% fun_plot_PnP_FIXED_PAR(pref, suffix_, p_str, "\gamma = ");
% title("ML-PnP-DRUNet init");
% saveas(gcf, "FPAR_ML-PnP-DRUNet_init", 'epsc');
% 
% % set figure PnP-GS
% suffix_ = exp_name + '_PnP_prox_' + suffix;
% fun_plot_PnP_FIXED_PAR(pref, suffix_, p_str, "\gamma = ");
% title("PnP-GS");
% saveas(gcf, "FPAR_PnP-GS", 'epsc');
% 
% % set figure PnP-ML-DRUNet init
% suffix_ = exp_name + '_PnP_prox_ML_INIT_' + ml_part + suffix;
% fun_plot_PnP_FIXED_PAR(pref, suffix_, p_str, "\gamma = ");
% title("ML-PnP-GS init");
% saveas(gcf, "FPAR_ML-PnP-GS_init", 'epsc');

%% DnCNN and SCUNet (stepsize param)

% sz (x4) : PnP_DnCNN, PnP_SCUNet

p_str = ["noname"];

id = 1;
for sz = sz_vec
    str_id = ("sz" + sz);
    p_str(id) = str_id;
    id = id + 1;
end

% set figure PnP-SCUNet
suffix_ = exp_name + '_PnP_SCUNet_' + suffix;
fun_plot_PnP_FIXED_PAR(pref, suffix_, p_str, "\gamma = ");
title("PnP-SCUNet");
saveas(gcf, "FPAR_PnP-SCUNet", 'epsc');

% set figure PnP-ML-SCUNet init
suffix_ = exp_name + '_PnP_ML_SCUNet_init_' + ml_part + suffix;
fun_plot_PnP_FIXED_PAR(pref, suffix_, p_str, "\gamma = ");
title("ML-PnP-SCUNet init");
saveas(gcf, "FPAR_ML-PnP-SCUNet_init", 'epsc');

% set figure PnP-DnCNN
suffix_ = exp_name + '_PnP_DnCNN_' + suffix;
fun_plot_PnP_FIXED_PAR(pref, suffix_, p_str, "\gamma = ");
title("PnP-DnCNN");
saveas(gcf, "FPAR_PnP-DnCNN", 'epsc');

% set figure PnP-ML-DnCNN init
suffix_ = exp_name + '_PnP_ML_DnCNN_init_' + ml_part + suffix;
fun_plot_PnP_FIXED_PAR(pref, suffix_, p_str, "\gamma = ");
title("ML-PnP-DnCNN init");
saveas(gcf, "FPAR_ML-PnP-DnCNN_init", 'epsc');
