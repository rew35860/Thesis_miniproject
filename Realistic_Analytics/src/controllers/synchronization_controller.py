import torch


class SynchronizationController:
    def __init__(self, k_ps, delta_phi_star=None):
        # k_ps: synchronization gain
        # delta_phi_star: desired phase offsets between oscillators
        self.k_ps = k_ps
        self.delta_phi_star = delta_phi_star

    def corrected_frequency(self, i, phi, omega_i):
        """
        Compute corrected frequency for oscillator i using raw phase values.
        """
        phi_i = phi[i]

        if self.delta_phi_star is None:
            delta_row = torch.zeros_like(phi)
        else:
            delta_row = self.delta_phi_star[i]

        coupling = torch.sum(torch.sin(delta_row + phi_i - phi))
        omega_tilde = omega_i * (1 - self.k_ps * coupling)

        return omega_tilde

    def corrected_frequency_from_sin_cos(self, i, sin_phi, cos_phi, omega_i):
        """
        Compute corrected frequency using (sin φ, cos φ) estimates instead of
        raw phase values.  Avoids atan2 and is numerically robust.

        Uses the identity:
            sin(φ_i - φ_j) = sin_i * cos_j - cos_i * sin_j

        sin_phi, cos_phi: tensors of shape [N] — estimated for all oscillators
        omega_i:          scalar — natural frequency of oscillator i
        """
        sin_i = sin_phi[i]
        cos_i = cos_phi[i]

        if self.delta_phi_star is None:
            # sin(phi_i - phi_j) expanded without atan2
            coupling = torch.sum(sin_i * cos_phi - cos_i * sin_phi)
        else:
            delta_row = self.delta_phi_star[i]   # desired offsets [N]
            # sin(delta + phi_i - phi_j) = sin(delta)*cos(phi_i-phi_j) + cos(delta)*sin(phi_i-phi_j)
            sin_diff = sin_i * cos_phi - cos_i * sin_phi
            cos_diff = cos_i * cos_phi + sin_i * sin_phi
            coupling = torch.sum(
                torch.sin(delta_row) * cos_diff + torch.cos(delta_row) * sin_diff
            )

        omega_tilde = omega_i * (1 - self.k_ps * coupling)
        return omega_tilde