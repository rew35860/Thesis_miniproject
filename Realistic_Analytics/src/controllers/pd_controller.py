

class PDController:
    def __init__(self, kp, kd, d, k):
        # kp: position gain, kd: velocity gain (feedback terms)
        # d, k: system parameters used for feedforward compensation
        self.kp = kp
        self.kd = kd
        self.d = d
        self.k = k

    def compute(self, x, v, x_ref, v_ref):
        """
        Compute control input to track the reference trajectory.

        - PD term corrects tracking error (x_ref - x, v_ref - v)
        - Feedforward term (d*v_ref + k*x_ref) compensates system dynamics
        """
        u = (
            self.kp * (x_ref - x)
            + self.kd * (v_ref - v)
            + self.d * v_ref
            + self.k * x_ref
        )
        return u