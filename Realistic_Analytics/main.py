import torch
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

device = "cpu"

# -----------------------------
# Parameters
# -----------------------------
N = 5
T = 2000
dt = 0.005

A = 1.0
m = 1.0
d = 0.4
k = 4.0

kp = 20.0
kd = 8.0

K_sync = 1.5

# nominal frequencies
omega = torch.tensor([2.0 * torch.pi, 2.0 * torch.pi, 2.0 * torch.pi, 2.0 * torch.pi, 2.0 * torch.pi], device=device)

# -----------------------------
# Initial conditions
# -----------------------------
x = torch.zeros(N, device=device)
v = torch.zeros(N, device=device)
phi = torch.tensor([0.0, torch.pi / 2, torch.pi, 3 * torch.pi / 2, 2 * torch.pi], device=device)

# storage
x_hist = []
v_hist = []
phi_hist = []
xref_hist = []

# -----------------------------
# Simulation loop
# -----------------------------
for t in range(T):
    # pairwise phase differences: phi_i - phi_j
    phi_i = phi.unsqueeze(1)          # (N, 1)
    phi_j = phi.unsqueeze(0)          # (1, N)
    phase_diff = phi_i - phi_j        # (N, N)

    # synchronization term
    sync_term = -K_sync * torch.sum(torch.sin(phase_diff), dim=1)

    # phase dynamics
    phi_dot = omega * (1 + sync_term)

    # reference trajectory
    x_ref = A * torch.sin(phi)
    v_ref = A * torch.cos(phi) * phi_dot

    # PD controller
    u = kp * (x_ref - x) + kd * (v_ref - v)

    # physical dynamics
    x_dot = v
    v_dot = (u - d * v - k * x) / m

    # Euler integration
    x = x + dt * x_dot
    v = v + dt * v_dot
    phi = phi + dt * phi_dot

    # wrap phase to [0, 2pi)
    phi = torch.remainder(phi, 2.0 * torch.pi)

    # save
    x_hist.append(x.clone())
    v_hist.append(v.clone())
    phi_hist.append(phi.clone())
    xref_hist.append(x_ref.clone())

# convert to tensors
x_hist = torch.stack(x_hist)         # (T, N)
v_hist = torch.stack(v_hist)
phi_hist = torch.stack(phi_hist)
xref_hist = torch.stack(xref_hist)

# -----------------------------
# Plot
# -----------------------------
time = torch.arange(T) * dt

plt.figure(figsize=(10, 4))
for i in range(N):
    plt.plot(time, x_hist[:, i], label=f"x_{i}")
    plt.plot(time, xref_hist[:, i], "--", label=f"xref_{i}")
plt.xlabel("Time [s]")
plt.ylabel("Position")
plt.title("Oscillator tracking")
plt.legend()
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 4))
for i in range(N):
    plt.plot(time, phi_hist[:, i], label=f"phi_{i}")
plt.xlabel("Time [s]")
plt.ylabel("Phase")
plt.title("Phase evolution")
plt.legend()
plt.tight_layout()
plt.show()

phase_error = torch.atan2(
    torch.sin(phi_hist[:, 0] - phi_hist[:, 1]),
    torch.cos(phi_hist[:, 0] - phi_hist[:, 1])
)

plt.figure(figsize=(10, 4))
plt.plot(time, phase_error)
plt.xlabel("Time [s]")
plt.ylabel("Wrapped phase error")
plt.title("Phase synchronization error")
plt.tight_layout()
plt.show()

# -----------------------------
# Polar plot of phase evolution (time as radius)
# -----------------------------
plt.figure(figsize=(6, 6))
ax = plt.subplot(111, projection='polar')

# create time vector
T = phi_hist.shape[0]
time = torch.arange(T) * dt   # make sure dt exists

for i in range(N):
    theta = phi_hist[:, i].cpu()
    r = time.cpu()
    
    ax.plot(theta, r, label=f"osc {i}")

ax.set_title("Phase evolution (polar)")
ax.set_xlabel("Phase φ [rad]")
ax.set_ylabel("Time [s]")
ax.legend(loc="upper right")

plt.show()


plt.figure(figsize=(6, 6))

for i in range(N):
    x_phase = torch.cos(phi_hist[:, i])
    y_phase = torch.sin(phi_hist[:, i])
    plt.plot(x_phase, y_phase, label=f"osc {i}")

# draw unit circle
theta = torch.linspace(0, 2*torch.pi, 200)
plt.plot(torch.cos(theta), torch.sin(theta), 'k--', alpha=0.3)

plt.xlabel("cos(phi)")
plt.ylabel("sin(phi)")
plt.title("Phase evolution on unit circle")
plt.axis("equal")
plt.legend()
plt.show()


phase_diff_x = torch.cos(phi_hist[:, 0]) - torch.cos(phi_hist[:, 1])
phase_diff_y = torch.sin(phi_hist[:, 0]) - torch.sin(phi_hist[:, 1])

distance = torch.sqrt(phase_diff_x**2 + phase_diff_y**2)

plt.figure(figsize=(10, 4))
plt.plot(time, distance)
plt.xlabel("Time [s]")
plt.ylabel("Distance on unit circle")
plt.title("Phase synchronization (should go to 0)")
plt.show()