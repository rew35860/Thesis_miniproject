import torch


class SynchronizationController:
    def __init__(self, k_ps, delta_phi_star=None):
        # k_ps: synchronization gain
        # delta_phi_star: desired phase offsets between oscillators
        self.k_ps = k_ps
        self.delta_phi_star = delta_phi_star

    def corrected_frequency(self, i, phi, omega_i):
        """
        Compute corrected frequency for oscillator i using phase error.
        """
        phi_i = phi[i]

        if self.delta_phi_star is None:
            delta_row = torch.zeros_like(phi)
        else:
            delta_row = self.delta_phi_star[i]

        coupling = torch.sum(torch.sin(delta_row + phi_i - phi))
        omega_tilde = omega_i * (1 - self.k_ps * coupling)

        return omega_tilde