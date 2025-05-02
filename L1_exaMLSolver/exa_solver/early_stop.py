class EarlyStopping:
    def __init__(self, rtol, atol, dtol, patience):
        self.rtol = rtol
        self.atol = atol
        self.dtol = dtol
        self.patience = patience
        self.best_loss = float("inf")
        self.wait = 0

    def check(self, residual_norm, b_norm):
        # Convergence criteria
        if residual_norm < max(self.rtol * b_norm, self.atol):
            return True, "Converged"
        # Divergence criteria
        if residual_norm > self.dtol * b_norm:
            return True, "Diverged"
        return False, None

    def update(self, current_loss):
        if current_loss < self.best_loss:
            self.best_loss = current_loss
            self.wait = 0
        else:
            self.wait += 1
        return self.wait > self.patience
