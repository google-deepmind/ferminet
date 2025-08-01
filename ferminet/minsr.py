

class MinSR:

    def __init__(self, lr=0.01, damping=1e-4, adaptive_step=False):
        self.lr = lr
        self.damping = damping
        self.adaptive_step = adaptive_step