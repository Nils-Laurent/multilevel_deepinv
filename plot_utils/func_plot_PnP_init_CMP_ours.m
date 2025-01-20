function func_plot_PnP_init_CMP_ours(custom_plt, exp_name, level, win, suffix)
mlevel_suffix = string(level)+"L"+win+"_"+suffix;

name = exp_name + "_PnP_"+suffix;
custom_plt(name, '#000000--', 'PnP');

name = exp_name + "_PnP_ML_"+mlevel_suffix;
custom_plt(name, '#FF8822--', 'ML-PnP');

name = exp_name + "_PnP_ML_Moreau_"+mlevel_suffix;
custom_plt(name, '#FF8822-', 'ML-PnP-TV');

% === INIT ===
name = exp_name + "_PnP_INIT_"+suffix;
custom_plt(name, '#000000-.', 'PnP init');

name = exp_name + "_PnP_ML_INIT_"+mlevel_suffix;
custom_plt(name, '#EE2222--', 'ML-PnP init');

name = exp_name + "_PnP_ML_Moreau_INIT_"+mlevel_suffix;
custom_plt(name, '#EE2222-', 'ML-PnP-TV init');

end