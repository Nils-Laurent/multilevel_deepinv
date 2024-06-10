from matplotlib import pyplot
import numpy

if __name__ == "__main__":
    np_res = numpy.load('out/data_drunet_time.npy')
    [nb_pixels, vec_size, y_time] = np_res
    fig = pyplot.figure()
    pyplot.plot(list(vec_size), y_time, linestyle='--', marker='o')
    pyplot.xlabel("image size (pixels)")
    pyplot.ylabel("denoiser time (seconds)")
    pyplot.show()