import torch


class Oscillator:
    def __init__(self, m, d, k, A, kp, kd, omega, device="cpu"):
        self.m = m
        self.d = d
        self.k = k
        self.A = A
        self.kp = kp
        self.kd = kd
        self.omega = omega
        self.device = device

    # -----------------------------
    # 1. Phase dynamics (synchronization)
    # -----------------------------
    def phase_dynamics(self, phi_i, phi_all, k_sync):
        # pairwise phase differences
        phase_diff = phi_i - phi_all

        # synchronization term
        sync_term = -k_sync * torch.sum(torch.sin(phase_diff))

        # frequency modulation (Eq. 6 style)
        phi_dot = self.omega * (1 + sync_term)

        return phi_dot

    # -----------------------------
    # 2. Reference trajectory
    # -----------------------------
    def reference(self, phi, phi_dot):
        x_ref = self.A * torch.sin(phi)
        v_ref = self.A * torch.cos(phi) * phi_dot
        return x_ref, v_ref

    # -----------------------------
    # 3. PD + feedforward controller
    # -----------------------------
    def controller(self, x, v, x_ref, v_ref):
        u = (
            self.kp * (x_ref - x)
            + self.kd * (v_ref - v)
            + self.d * v_ref
            + self.k * x_ref
        )
        return u

    # -----------------------------
    # 4. Physical dynamics
    # -----------------------------
    def dynamics(self, x, v, u):
        x_dot = v
        v_dot = (u - self.d * v - self.k * x) / self.m
        return x_dot, v_dot

    # -----------------------------
    # 5. Full step
    # -----------------------------
    def step(self, x, v, phi, phi_all, dt, k_sync):
        # (1) phase update
        phi_dot = self.phase_dynamics(phi, phi_all, k_sync)

        # (2) reference from phase
        x_ref, v_ref = self.reference(phi, phi_dot)

        # (3) control
        u = self.controller(x, v, x_ref, v_ref)

        # (4) physics
        x_dot, v_dot = self.dynamics(x, v, u)

        # integrate
        x_next = x + dt * x_dot
        v_next = v + dt * v_dot
        phi_next = phi + dt * phi_dot
        phi_next = torch.remainder(phi_next, 2.0 * torch.pi)

        return x_next, v_next, phi_next, x_ref, v_ref, u
    



class Oscillator:
    def dynamics(self, x, v, u):
        x_dot = v
        v_dot = (u - d * v - k * x) / m
        return x_dot, v_dot

    def step(self, x, v, u, dt):
        x_dot, v_dot = self.dynamics(x, v, u)
        x_next = x + dt * x_dot
        v_next = v + dt * v_dot
        return x_next, v_next
    
    