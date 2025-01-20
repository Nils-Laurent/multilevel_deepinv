function fun_plot_PnP_FIXED_PAR(prefix, suffix, vec, lgd_prefix)

% col_vec = ["#FF0000--", "#FF0000", "#FF7F00", "#FFFF00", "#00FF00", "#0000FF", "#4B0082", "#9400D3", "#9400D3--"];
col_vec = ["#FF0000", "#00FF00", "#0000FF", "#9400D3"];
for k = 5:8
    col_vec(k) = (col_vec(k - 4) + "-.");
end
for k = 9:12
    col_vec(k) = (col_vec(k - 8) + "--");
end
for k = 13:16
    col_vec(k) = (col_vec(k - 12) + ":");
end


figure;
grid on;
hold on;

k = 0;
for v = vec
    k = k + 1;
    def_name = prefix + v + '_' + suffix;
    try
       z = load("out\" + def_name + ".mat");
    catch ME
       continue;
    end
    
    x_vec = cumsum(z.time)/1000; % ms to seconds
    y_vec = z.psnr;

    def_form = char(col_vec(k));
    imp_plot(x_vec, y_vec, def_form, lgd_prefix + v);
end
hold off;
xlabel('time (s)');
ylabel('PSNR');
legend('Location','southeast');

end

