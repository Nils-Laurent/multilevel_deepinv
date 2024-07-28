import numpy as np
import scipy.signal
import scipy.signal as signal
import scipy.interpolate as interpolate
import matplotlib.pyplot as plt

# Normalized frequencies at which interpolate
num_points = 20
f_vec = np.linspace(0, 1, num_points)

# Desired magnitude response
cut_off = 0.5
H = np.where(f_vec <= cut_off, 1, 0)  # impossible to get in practice

# Perform cubic spline interpolation
spline = interpolate.CubicSpline(f_vec, H)

# Interpolated frequency response
f_illustr = np.linspace(0, 1, 1000)
H_interp = spline(f_illustr)

scipy.signal.spline_filter()

# Plot the desired and interpolated frequency response
plt.figure(figsize=(10, 6))
plt.plot(f_vec, H, 'o', label='Desired Points')
plt.plot(f_illustr, H_interp, '-', label='Cubic Spline Interpolation')
plt.title('Low-Pass Filter Design using Cubic Spline Interpolation')
plt.xlabel('Normalized Frequency')
plt.ylabel('Magnitude Response')
plt.legend()
plt.grid(True)
plt.show()

# Inverse Fourier Transform to get the impulse response
h = np.fft.ifft(H_interp)
h = np.real(h)  # Take the real part as impulse response should be real

# Plot the impulse response
plt.figure(figsize=(10, 6))
plt.plot(h, label='Impulse Response')
plt.title('Impulse Response of the Low-Pass Filter')
plt.xlabel('Samples')
plt.ylabel('Amplitude')
plt.legend()
plt.grid(True)
plt.show()

# Frequency response of the designed filter
w, h_response = signal.freqz(h)

plt.figure(figsize=(10, 6))
plt.plot(w/np.pi, np.abs(h_response), label='Frequency Response')
plt.title('Frequency Response of the Designed Low-Pass Filter')
plt.xlabel('Normalized Frequency')
plt.ylabel('Magnitude Response')
plt.legend()
plt.grid(True)
plt.show()