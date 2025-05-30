import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset


class UWBFollowModel:
    def __init__(self, csv_filename=None, hidden_size=32, learning_rate=0.01):
        self.input_size = 4  # 2 ranges + 2 angles
        self.output_size = 2  # v_linear, v_angular
        self.hidden_size = hidden_size
        self.learning_rate = learning_rate
        self.batch_size = 256
        self.num_epochs = 100

        self.data = pd.read_csv(csv_filename) if csv_filename else None
        if self.data is not None:
            self.preprocess_data()

        self.model = self.MLP_Model(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            output_size=self.output_size,
        )

        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

    def preprocess_data(self):
        """Convert angles to radians and split data."""
        self.data[['d1', 'd2']] = np.radians(self.data[['d1', 'd2']].values)
        X = self.data[['r1', 'r2', 'd1', 'd2']].values
        y = self.data[['v_linear', 'v_angular']].values

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)
        self.X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
        self.y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
        self.X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
        self.y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

        train_dataset = TensorDataset(self.X_train_tensor, self.y_train_tensor)
        self.train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)

    class MLP_Model(nn.Module):
        def __init__(self, input_size, hidden_size, output_size):
            super().__init__()
            self.fc1 = nn.Linear(input_size, hidden_size)
            self.fc2 = nn.Linear(hidden_size, hidden_size)
            self.fc3 = nn.Linear(hidden_size, output_size)
            self.relu = nn.ReLU()

        def forward(self, x):
            x = self.relu(self.fc1(x))
            x = self.relu(self.fc2(x))
            return self.fc3(x)

    def train(self):
        loss_list = []

        for epoch in range(self.num_epochs):
            self.model.train()
            for X_batch, y_batch in self.train_loader:
                y_pred = self.model(X_batch)
                loss = self.criterion(y_pred, y_batch)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                loss_list.append(loss.item())

            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(f"Epoch [{epoch + 1}/{self.num_epochs}], Loss: {loss.item():.4f}")

        plt.plot(loss_list)
        plt.title("Training Loss Over Epochs")
        plt.xlabel("Iteration")
        plt.ylabel("Loss")
        plt.grid(True)
        plt.show()

    def _evaluate_common(self, X_tensor, y_tensor, title_prefix="Test"):
        self.model.eval()
        with torch.no_grad():
            y_pred = self.model(X_tensor)
            loss = self.criterion(y_pred, y_tensor).item()
            y_pred_np = y_pred.numpy()
            y_true_np = y_tensor.numpy()

            mae = np.mean(np.abs(y_pred_np - y_true_np))
            rmse = np.sqrt(np.mean((y_pred_np - y_true_np) ** 2))

            print(f"{title_prefix} Loss: {loss:.4f}")
            print(f"{title_prefix} MAE: {mae:.4f}")
            print(f"{title_prefix} RMSE: {rmse:.4f}")

            plt.figure(figsize=(12, 6))
            for i, name in enumerate(['Linear', 'Angular']):
                plt.subplot(1, 2, i + 1)
                plt.plot(y_true_np[:, i], label='Ground Truth', linestyle='--')
                plt.plot(y_pred_np[:, i], label='Predicted', linestyle='-')
                plt.title(f'{title_prefix} {name} Velocity')
                plt.xlabel('Sample Index')
                plt.ylabel(f'{name} Velocity')
                plt.legend()

            plt.tight_layout()
            plt.show()
            return y_pred_np, y_true_np

    def evaluate(self):
        return self._evaluate_common(self.X_test_tensor, self.y_test_tensor, "Test")

    def evaluate_train(self):
        return self._evaluate_common(self.X_train_tensor, self.y_train_tensor, "Train")

    def predict(self, new_data):
        """
        Predict velocity commands for new sensor input.
        `new_data` should be a NumPy array of shape (n_samples, 4)
        """
        if new_data.shape[1] != self.input_size:
            raise ValueError(f"Expected input with shape (n_samples, {self.input_size})")

        new_data = new_data.copy()
        new_data[:, 2:4] = np.radians(new_data[:, 2:4])
        self.model.eval()
        with torch.no_grad():
            input_tensor = torch.tensor(new_data, dtype=torch.float32)
            prediction = self.model(input_tensor)
        return prediction.numpy()

    def export_to_onnx(self, onnx_filename="models/uwb_mlp.onnx"):
        self.model.eval()
        dummy_input = torch.ones(1, self.input_size)
        torch.onnx.export(
            self.model,
            dummy_input,
            onnx_filename,
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
        )
        print(f"Model exported to {onnx_filename}")


def main():
    model = UWBFollowModel(csv_filename='train/data.csv')
    model.train()
    model.evaluate()
    model.evaluate_train()
    model.export_to_onnx(onnx_filename='models/uwb_mlp.onnx')


if __name__ == "__main__":
    main()
