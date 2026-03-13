import torch


class TrainingMonitor:

    def __init__(self):
        self.loss_history = []
        self.gradient_norms = []

    def record_loss(self, loss):
        self.loss_history.append(loss)

    def record_gradient_norm(self, model):

        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2

        total_norm = total_norm ** 0.5

        self.gradient_norms.append(total_norm)

    def get_features(self):

        loss_slope = self.loss_hisotry[-1] - self.loss_history[0]

        avg_grad = sum(self.gradient_norms) / len(self.gradient_norms)

        grad_var = torch.tensor(self.gradient_norms).var().items()

        return {
            "loss_slope": loss_slope,
            "avg_grad_norm": avg_grad,
            "grad_variance": grad_var,
        }
