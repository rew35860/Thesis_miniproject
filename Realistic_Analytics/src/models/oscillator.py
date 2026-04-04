

class Oscillator:
    def __init__(self, m, d, k):
        # Physical parameters of the mass-spring-damper system
        self.m = m
        self.d = d
        self.k = k

    def dynamics(self, x, v, u):
        """
        Continuous-time dynamics:
            m * x_ddot + d * x_dot + k * x = u
        """
        x_dot = v
        v_dot = (u - self.d * v - self.k * x) / self.m
        return x_dot, v_dot

    def step(self, x, v, u, dt):
        """
        One Euler integration step of the physical system.
        """
        x_dot, v_dot = self.dynamics(x, v, u)
        x_next = x + dt * x_dot
        v_next = v + dt * v_dot
        return x_next, v_next