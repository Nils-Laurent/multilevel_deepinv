function func_plot_PnP_init_CMP_denoisers(custom_plt, exp_name, level, win, suffix)
mlevel_suffix = string(level)+"L"+win+"_"+suffix;

%% PnP-GS

name = exp_name + "_PnP_prox_"+suffix;
custom_plt(name, '#FF0000--', 'PnP-GS');

name = exp_name + "_PnP_prox_ML_INIT_"+mlevel_suffix;
custom_plt(name, '#FF0000-', 'ML-PnP-GS init');

%% PnP DRUNet

name = exp_name + "_PnP_"+suffix;
custom_plt(name, '#000000--', 'PnP-DRUNet');

name = exp_name + "_PnP_ML_INIT_"+mlevel_suffix;
custom_plt(name, '#000000-', 'ML-PnP-DRUNet init');

%% PnP SCUNet

name = exp_name + "_PnP_SCUNet_"+suffix;
custom_plt(name, '#4DBEEE--', 'PnP-SCUNet');

name = exp_name + "_PnP_ML_SCUNet_INIT_"+mlevel_suffix;
custom_plt(name, '#4DBEEE-', 'ML-PnP-SCUNet init');

%% PnP DnCNN

name = exp_name + "_PnP_DnCNN_"+suffix;
custom_plt(name, '#EDB120--', 'PnP-DnCNN');

name = exp_name + "_PnP_ML_DnCNN_INIT_"+mlevel_suffix;
custom_plt(name, '#EDB120-', 'ML-PnP-DnCNN init');

end