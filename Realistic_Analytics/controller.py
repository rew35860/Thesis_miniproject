class PDController:
    def __init__(self, kp, kd, d, k):
        self.kp = kp
        self.kd = kd
        self.d = d
        self.k = k

    def compute(self, x, v, x_ref, v_ref):
        u = (
            self.kp * (x_ref - x)
            + self.kd * (v_ref - v)
            + self.d * v_ref
            + self.k * x_ref
        )
        return u