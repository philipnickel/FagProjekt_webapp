import numpy as np


def compute_wave_propagation_opgave1(L, w, k0, v, N, dt, T):
    dx = L / N
    steps = int(T / dt)
    x = np.linspace(-L / 2, L / 2, N, endpoint=False)
    k = 2 * np.pi * np.fft.fftfreq(N, d=dx)
    u0 = np.exp(-(x**2) / w**2) * np.cos(k0 * x)
    u_numerical = np.zeros((steps, N), dtype=np.complex64)
    u_numerical[0, :] = np.fft.fft(u0)
    n_values = np.arange(1, steps)[:, np.newaxis]
    u_numerical[1:, :] = u_numerical[0, :] * np.cos(k * v * dt * n_values)
    u_numerical = np.fft.ifft(u_numerical, axis=1)
    u_moving = np.zeros((steps, N))
    for i, ti in enumerate(np.linspace(0, T, steps)):
        u_moving[i, :] = 0.5 * np.exp(-((x - v * ti) ** 2) / w**2) * np.cos(
            k0 * (x - v * ti)
        ) + 0.5 * np.exp(-((x + v * ti) ** 2) / w**2) * np.cos(k0 * (x + v * ti))
    return x, u_numerical, u_moving, steps


def compute_wave_propagation_opgave2(L, w, k0, v, N, dt, T):
    dx = L / N
    steps = int(T / dt)
    x = np.linspace(-L / 2, L / 2, N, endpoint=False)
    k = 2 * np.pi * np.fft.fftfreq(N, d=dx)
    u0 = np.exp(-(x**2) / w**2) * np.cos(k0 * x)
    u_numerical = np.zeros((steps, N), dtype=np.complex64)
    u_numerical[0, :] = np.fft.fft(u0)
    n_values = np.arange(1, steps)[:, np.newaxis]
    u_numerical[1:, :] = u_numerical[0, :] * np.cos(k * v * dt * n_values)
    u_numerical = np.fft.ifft(u_numerical, axis=1)
    u_moving = np.zeros((steps, N))
    for i, ti in enumerate(np.linspace(0, T, steps)):
        u_moving[i, :] = 0.5 * np.exp(-((x - v * ti) ** 2) / w**2) * np.cos(
            k0 * (x - v * ti)
        ) + 0.5 * np.exp(-((x + v * ti) ** 2) / w**2) * np.cos(k0 * (x + v * ti))
    return x, u_numerical, u_moving, steps


def compute_wave_propagation_opgave3(L, w, k0, v, N, dt, T):
    dx = L / N
    steps = int(T / dt)
    x = np.linspace(-L / 2, L / 2, N, endpoint=False)
    k = 2 * np.pi * np.fft.fftfreq(N, d=dx)
    # Initial condition
    u0 = np.cos(k0 * x) * np.exp(-((x / w) ** 2))
    # np.exp(-(x**2/ w**2)) * (1/2*np.exp(-1j*k0*x) + 1/2*np.exp(1j*k0*x))

    # Numerical solution using FFT
    u_numerical = np.zeros((steps, N), dtype=np.complex64)
    A_numerical = np.zeros((steps, N), dtype=np.complex64)
    A_numerical[0, :] = np.fft.fft(u0)

    n_values = np.arange(1, steps)[:, np.newaxis]  # Add new axis to align shapes
    u_numerical[1:, :] = A_numerical[0, :] * np.cos(
        v * np.sqrt(k**2 + 20) * dt * n_values
    )

    u_numerical = np.fft.ifft(u_numerical, axis=1)

    return x, u_numerical, steps


def compute_wave_propagation_opgave4(L, w, k0, v, N, dt, T):
    dx = L / N
    steps = int(T / dt)
    x = np.linspace(-L / 2, L / 2, N, endpoint=False)
    k = 2 * np.pi * np.fft.fftfreq(N, d=dx)
    # Initial condition
    A = 1
    u0 = A * np.exp(-((x / w) ** 2))
    # np.exp(-(x**2/ w**2)) * (1/2*np.exp(-1j*k0*x) + 1/2*np.exp(1j*k0*x))

    # Numerical solution using FFT
    u_numerical = np.zeros((steps, N), dtype=np.complex64)
    A_numerical = np.zeros((steps, N), dtype=np.complex64)
    A_numerical[0, :] = np.fft.fft(u0)

    n_values = np.arange(1, steps)[:, np.newaxis]  # Add new axis to align shapes
    u_numerical[1:, :] = A_numerical[0, :] * np.cos(
        k**2 * v * dt * n_values
    ) - 1j * A_numerical[0, :] * np.sin(k**2 * dt * v * n_values)

    u_numerical = np.fft.ifft(u_numerical, axis=1)

    u_moving = np.zeros((steps, N))
    for i, ti in enumerate(np.linspace(0, T, steps)):
        u_moving[i, :] = np.real(
            (np.exp(-(x**2) / (4 * 1j * ti * v + w**2)) * A * w * np.sqrt(np.pi))
            / (np.sqrt(np.pi) * np.sqrt(4 * 1j * ti * v + w**2))
        )

    return x, u_numerical, u_moving, steps


def compute_wave_propagation_projektplan_opgave1(L, w, k0, v, N, dt, T):
    dx = L / N
    steps = int(T / dt)
    x = np.linspace(-L / 2, L / 2, N, endpoint=False)
    k = 2 * np.pi * np.fft.fftfreq(N, d=dx)
    # Initial condition
    A = 0.5
    u0 = A * np.exp(-((x / w) ** 2))
    # np.exp(-(x**2/ w**2)) * (1/2*np.exp(-1j*k0*x) + 1/2*np.exp(1j*k0*x))

    # Numerical solution using FFT
    u_numerical = np.zeros((steps, N), dtype=np.complex64)
    A_numerical = np.zeros((steps, N), dtype=np.complex64)
    A_numerical[0, :] = np.fft.fft(u0)

    n_values = np.arange(1, steps)[:, np.newaxis]  # Add new axis to align shapes
    u_numerical[1:, :] = (
        2 * A_numerical[0, :] * np.cos(np.sqrt(k**2 + 1) * dt * v * n_values)
    )

    u_numerical = np.fft.ifft(u_numerical, axis=1)

    return x, u_numerical, steps


def compute_wave_propagation_projektplan_opgave2(L, w, k0, v, N, dt, T):
    dx = L / N
    steps = int(T / dt)
    x = np.linspace(-L / 2, L / 2, N, endpoint=False)
    k = 2 * np.pi * np.fft.fftfreq(N, d=dx)
    # Initial condition
    A = 0.5
    u0 = A * np.exp(-((x / w) ** 2)) * np.cos(5 * x)
    # np.exp(-(x**2/ w**2)) * (1/2*np.exp(-1j*k0*x) + 1/2*np.exp(1j*k0*x))

    # Numerical solution using FFT
    u_numerical = np.zeros((steps, N), dtype=np.complex64)
    A_numerical = np.zeros((steps, N), dtype=np.complex64)
    A_numerical[0, :] = np.fft.fft(u0)

    n_values = np.arange(1, steps)[:, np.newaxis]  # Add new axis to align shapes
    u_numerical[1:, :] = (
        2 * A_numerical[0, :] * np.cos(np.sqrt(k**2 + 1) * dt * n_values)
    )

    u_numerical = np.fft.ifft(u_numerical, axis=1)

    return x, u_numerical, steps
