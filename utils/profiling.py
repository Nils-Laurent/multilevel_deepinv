
# out_dir = get_out_dir()
# out_f = os.path.join(out_dir, 'cprofile_data')
# prof = torch.profiler.profile(
#    schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=1),
#    on_trace_ready=torch.profiler.tensorboard_trace_handler(out_f),
#    with_modules=True)
# prof.start()
# for step in range(1 + 1 + 3):
#    prof.step()
#    ra.RED_GD(p_red.copy())
# prof.stop()
# cProfile.runctx(
#    statement='ra.RED_GD(param)',
#    globals={'param' : p_red.copy(), 'ra': ra},
#    locals={},
#    filename=out_f
# )