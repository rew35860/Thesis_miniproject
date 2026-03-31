import numpy as np

class Oscillator:
    def __init__(self, theta, omega, amplitude=1.0):
        """
        theta: initial phase
        omega: natural frequency / target (nominal) frequency
        amplitude: signal amplitude 
        position: observable position
        """
        self.theta = theta
        self.omega = omega
        self.amplitude = amplitude
        self.position = self.get_position()

    def step(self, thetas_all, K, dt):
        """
        Update phase using coupling (synchronization rule)
        eq: (dtheta/dt)_i = omega + K * sum(sin(theta_j - theta_i))
        
        thetas_all: array of all oscillator phases
        K: coupling strength
        dt: timestep
        """
        coupling = np.sum(np.sin(self.theta - thetas_all))
        dtheta = self.omega * (1 - K * coupling)
        self.theta += dtheta * dt

    def get_position(self):
        """
        Convert phase to observable motion
        """
        self.position = self.amplitude * np.sin(self.theta)
        return self.position
