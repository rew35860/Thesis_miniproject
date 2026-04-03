import torch


class SinusoidalReference:
    def __init__(self, A):
        # Amplitude of the sinusoidal reference trajectory
        self.A = A

    def get_reference(self, phi, phi_dot):
        # Convert phase into desired position and velocity
        x_ref = self.A * torch.sin(phi)
        v_ref = self.A * torch.cos(phi) * phi_dot
        return x_ref, v_ref