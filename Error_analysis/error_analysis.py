import numpy as np


# Full step split step fourier method
def split_step_fourier_full_step(C1, C2, u, k, dt):
    # Nonlinear part
    u *= np.exp((C2 * 1j * (np.abs(u) ** 2) * dt))
    # Linear part
    u_fft = np.fft.fft(u)
    u_fft *= np.exp(C1 * -1j * (k**2) * dt)
    u = np.fft.ifft(u_fft)

    return u


# Half step split step fourier method
def split_step_fourier(C1, C2, u, k, dt):
    # Nonlinear part
    u *= np.exp((C2 * 1j * (np.abs(u) ** 2) * dt / 2))

    # Linear part
    u_fft = np.fft.fft(u)
    u_fft *= np.exp(C1 * -1j * (k**2) * dt)
    u = np.fft.ifft(u_fft)

    # Nonlinear part, again
    u *= np.exp((C2 * 1j * (np.abs(u) ** 2) * dt / 2))

    return u


def compute_wave_propagation(
    C1, C2, L, sigma, v, theta, N, dt, T, initial_condition, method
):
    steps = int(T / dt)
    # Spatial domain setup
    dx = L / N
    x = np.linspace(-L / 2, L / 2, N, endpoint=False)
    k = 2 * np.pi * np.fft.fftfreq(N, d=dx)

    A = 1  # Amplitude

    if initial_condition == 1:
        # Initial condition (Gaussian wave packet)
        # u0 = np.exp(-(x/w)**2 + 1j * k0 * x)
        u0 = 0
    elif initial_condition == 2:
        u0 = (
            A
            * 1
            / np.cosh(A * np.sqrt(C2 / (2 * C1)) * (x - theta))
            * np.exp(1j * v / (2 * C1) * x + 1j * sigma)
        )

    # Prepare the storage for the solution at each time step
    u_storage = np.zeros((steps, N), dtype=np.complex64)
    u = u0

    u_analytical = np.zeros((steps, N), dtype=np.complex64)

    if method == "half":
        for i in range(steps):
            # Perform SSFM step
            u_storage[i, :] = u
            u = split_step_fourier(C1, C2, u, k, dt)
    elif method == "full":
        for i in range(steps):
            # Perform SSFM step
            u_storage[i, :] = u
            u = split_step_fourier_full_step(C1, C2, u, k, dt)

    x = np.linspace(-L / 2, L / 2, N, endpoint=False)
    n_values = np.arange(1, steps)[:, np.newaxis]  # Add new axis to align shapes

    u_analytical[0, :] = (
        A
        * 1
        / np.cosh(A * np.sqrt(C2 / (2 * C1)) * (x - theta))
        * np.exp(1j * v / (2 * C1) * x + 1j * sigma)
    )
    u_analytical[1:, :] = (
        A
        * 1
        / np.cosh(A * np.sqrt(C2 / (2 * C1)) * (x - v * dt * n_values - theta))
        * np.exp(
            1j * v / (2 * C1) * x
            + 1j * ((C2 * A**2) / 2 - (v**2 / (4 * C1))) * dt * n_values
            + 1j * sigma
        )
    )

    return x, u_storage, u_analytical, steps


# Set initial conditions
initial_L = 30
initial_T = 10
initial_sigma = 0
initial_N = 1000
initial_dt = 0.005
initial_v = 10
initial_condition = 2
initial_C1 = 1
initial_C2 = 2
initial_theta = 0
initial_interval = 10
initial_N2 = 500
initial_T2 = 10

# Perform initial computation
x, u_numerical, u_analytical, steps = compute_wave_propagation(
    initial_C1,
    initial_C2,
    initial_L,
    initial_sigma,
    initial_v,
    initial_theta,
    initial_N,
    initial_dt,
    initial_T,
    initial_condition,
    "half",
)
