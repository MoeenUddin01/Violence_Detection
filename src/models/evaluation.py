import torch
from torch.utils.data import DataLoader


class Evaluator:
    """
    This class is responsible for evaluating a trained model
    on validation or test data.
    """

    def __init__(self, model, data: DataLoader, device: str):
        self.model = model
        self.data = data
        self.device = device
        self.loss_fn = torch.nn.CrossEntropyLoss()

    def evaluate(self):
        """
        Evaluates the model on the given dataset.
        Returns:
            avg_loss (float)
            accuracy (float)
        """

        # switch model to evaluation mode
        self.model.eval()

        total_loss = 0.0
        correct = 0
        total = 0

        # no gradients needed during evaluation
        with torch.no_grad():
            for x, y in self.data:
                # move data to device
                x = x.to(self.device)
                y = y.to(self.device)

                # forward pass
                outputs = self.model(x)

                # compute loss
                loss = self.loss_fn(outputs, y)
                total_loss += loss.item()

                # get predicted class
                _, predicted = torch.max(outputs, dim=1)

                total += y.size(0)
                correct += (predicted == y).sum().item()

        # calculate final metrics
        avg_loss = total_loss / len(self.data)
        accuracy = 100.0 * correct / total if total > 0 else 0.0

        return avg_loss, accuracy
