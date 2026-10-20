import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
import time
from sklearn.multioutput import MultiOutputRegressor
import numpy as np
import torch
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
import onnx
import time

from sklearn.metrics import mean_squared_error
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType


class UWBFollowModel:
    def __init__(self, csv_filename=None, seq_length=1, learning_rate=0.01):
        if csv_filename:
            self.data = pd.read_csv(csv_filename)
        else:
            self.data = None

        self.seq_length = seq_length
        self.input_size = 4  # r1, r2, d1, d2
        self.output_size = 2  # v_linear, v_angular
        self.learning_rate = learning_rate

        if self.data is not None:
            self.preprocess_data()

        # Initialize SVM regressor
        self.model = SVR(kernel='rbf')  # Radial basis function kernel

    def preprocess_data(self):
        self.data[['d1', 'd2']] = np.radians(self.data[['d1', 'd2']].values)

        X = self.data[['r1', 'r2', 'd1', 'd2']].values
        y = self.data[['v_linear', 'v_angular']].values

        # points = self.data.iloc[:, 6:].values  # Assuming point data starts from column 5 onward

        # X = np.concatenate((X, points), axis=1)

        # Split the data
        # split_index = 1000
        # self.X_train, self.X_test = self.X_seq[split_index:], self.X_seq[:split_index]
        # self.y_train, self.y_test = self.y_seq[split_index:], self.y_seq[:split_index]

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                    X, y, test_size=0.1, random_state=None)

        print(f"X_train shape: {self.X_train.shape}")
        print(f"y_train shape: {self.y_train.shape}")
 



    def train(self):

        # Fit the SVM regressor
        multioutput_model = MultiOutputRegressor(self.model)
        multioutput_model.fit(self.X_train, self.y_train)
        self.model = multioutput_model
        print("Model trained.")

    def evaluate(self):
        # Evaluate the model on the test set
        print((self.X_test[0][0]))
        y_pred_test = self.model.predict(self.X_test)
        
        # Inverse transform the scaled predictions and actual values
        y_pred_test_original = y_pred_test
        y_test_original = self.y_test

        # Calculate Mean Absolute Error (MAE) and Root Mean Squared Error (RMSE)
        mae = np.mean(np.abs(y_pred_test_original - y_test_original))
        rmse = np.sqrt(np.mean((y_pred_test_original - y_test_original) ** 2))

        print(f"Test Mean Absolute Error (MAE): {mae:.4f}")
        print(f"Test Root Mean Squared Error (RMSE): {rmse:.4f}")

        test_loss = mean_squared_error(y_pred_test, y_test_original)
        print(f'Test Loss: {test_loss.item():.4f}')
        

        # Plot predictions vs actual values
        plt.figure(figsize=(12, 6))

        # Plot for v_linear
        plt.subplot(1, 2, 1)
        plt.plot(y_test_original[:, 0], label='Ground Truth (v_linear)', color='blue', linestyle='--')
        plt.plot(y_pred_test_original[:, 0], label='Predicted (v_linear)', color='red', linestyle='-')
        plt.title('Linear Velocity (v_linear)')
        plt.xlabel('Sample Index')
        plt.ylabel('Linear Velocity')
        plt.legend()

        # Plot for v_angular
        plt.subplot(1, 2, 2)
        plt.plot(y_test_original[:, 1], label='Ground Truth (v_angular)', color='blue', linestyle='--')
        plt.plot(y_pred_test_original[:, 1], label='Predicted (v_angular)', color='red', linestyle='-')
        plt.title('Angular Velocity (v_angular)')
        plt.xlabel('Sample Index')
        plt.ylabel('Angular Velocity')
        plt.legend()

        plt.tight_layout()
        # plt.show()

    def evaluate_train(self):
        # Evaluate the model on the test set
        y_pred_test = self.model.predict(self.X_train)
        
        # Inverse transform the scaled predictions and actual values
        y_pred_test_original = y_pred_test
        y_test_original = self.y_train

        # Calculate Mean Absolute Error (MAE) and Root Mean Squared Error (RMSE)
        mae = np.mean(np.abs(y_pred_test_original - y_test_original))
        rmse = np.sqrt(np.mean((y_pred_test_original - y_test_original) ** 2))

        print(f"Train Mean Absolute Error (MAE): {mae:.4f}")
        print(f"Train Root Mean Squared Error (RMSE): {rmse:.4f}")

        train_loss = mean_squared_error(y_pred_test_original, y_test_original)
        print(f'Train Loss: {train_loss.item():.4f}')

        # Plot predictions vs actual values
        plt.figure(figsize=(12, 6))

        # Plot for v_linear
        plt.subplot(1, 2, 1)
        plt.plot(y_test_original[:, 0], label='Ground Truth (v_linear)', color='blue', linestyle='--')
        plt.plot(y_pred_test_original[:, 0], label='Predicted (v_linear)', color='red', linestyle='-')
        plt.title('Linear Velocity (v_linear)')
        plt.xlabel('Sample Index')
        plt.ylabel('Linear Velocity')
        plt.legend()

        # Plot for v_angular
        plt.subplot(1, 2, 2)
        plt.plot(y_test_original[:, 1], label='Ground Truth (v_angular)', color='blue', linestyle='--')
        plt.plot(y_pred_test_original[:, 1], label='Predicted (v_angular)', color='red', linestyle='-')
        plt.title('Angular Velocity (v_angular)')
        plt.xlabel('Sample Index')
        plt.ylabel('Angular Velocity')
        plt.legend()

        plt.tight_layout()
        # plt.show()

    def predict(self, new_data):
        # Example: Predicting new cmd_vel commands
        if self.data is None:
            print("No data available to predict with.")
            return None

        new_data[:, 2:4] = np.radians(new_data[:, 2:4])  # Convert angles to radians
        new_data_reshaped = new_data.reshape(1, self.seq_length * self.input_size)  # Flatten the new data

        # Get the predicted velocity commands
        prediction = self.model.predict(new_data_reshaped)

        # Return the predicted velocity (inverse scaled)
        return self.scaler_y.inverse_transform(prediction)

    def save_model(self, model_path):
        # Save the trained model to disk
        import joblib
        joblib.dump(self.model, model_path)
        print(f"Model saved to {model_path}.")

    def load_model(self, model_path):
        # Load a saved model
        import joblib
        self.model = joblib.load(model_path)
        print(f"Model loaded from {model_path}.")


    def export_to_onnx(self, onnx_filename="models/onnx/uwb_svm.onnx", opset_version=9):
        """
        Export the trained model to ONNX format.
        """
        initial_type = [('input', FloatTensorType([None, self.X_train.shape[1]]))]
        onnx_model = convert_sklearn(self.model, initial_types=initial_type, target_opset=opset_version)
        with open(onnx_filename, "wb") as f:
            f.write(onnx_model.SerializeToString())

        print(f"Model has been exported to {onnx_filename}.")

def main():

    model = UWBFollowModel(csv_filename='data/csv/free/coop.csv')

    # Train the model (only if csv_filename is provided)
    if model.data is not None:
        model.train()
        # model.save_model(model_path='models/torch/svm_scan.pth')


    # model.export_to_onnx(onnx_filename='models/onnx/uwb_svm.onnx')

    if model.data is not None:
        model.evaluate()

    if model.data is not None:
        model.evaluate_train()


if __name__ == "__main__":
    main()