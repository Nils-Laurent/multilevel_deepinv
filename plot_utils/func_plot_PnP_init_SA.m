function func_plot_PnP_init_SA(custom_plt, exp_name, level, win, suffix)
mlevel_suffix = string(level)+"L"+win+"_"+suffix;

%% DPIR plots

name = exp_name + "_DPIR_"+suffix;
custom_plt(name, '#22AA55-', 'DPIR 8it');

name = exp_name + "_DPIR_Long_"+suffix;
custom_plt(name, '#22DD99--', 'DPIR 200it');

%% TV plots
name = exp_name + "_FB_TV_"+suffix;
custom_plt(name, 'b--', 'FB TV');

name = exp_name + "_FB_TV_ML_"+mlevel_suffix;
custom_plt(name, 'b-', 'ML-FB TV');

%% PnP-GS

name = exp_name + "_PnP_prox_"+suffix;
custom_plt(name, '#EE22AA--', 'PnP-GS');

%% PnP plots

name = exp_name + "_PnP_ML_"+mlevel_suffix;
custom_plt(name, '#FF8822--', 'ML-PnP');

% === INIT ===
name = exp_name + "_PnP_ML_INIT_"+mlevel_suffix;
custom_plt(name, '#EE2222--', 'ML-PnP init');

end