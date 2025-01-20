function imp_plot(x_vec, y_vec, def_form, lgd_name)
    if def_form(1) == '#'
        plot(x_vec, y_vec, def_form(8:end), 'color', def_form(1:7), 'DisplayName', lgd_name);
    else
        plot(x_vec, y_vec, def_form, 'DisplayName', lgd_name);
    end
end

