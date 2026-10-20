import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
import torch.onnx
import time
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset




class UWBFollowModel:
    def __init__(self, csv_filename=None, seq_length=100, hidden_size=32, num_layers=1, learning_rate=0.01):
        if csv_filename:
            self.data = pd.read_csv(csv_filename)
        else:
            self.data = None

        self.seq_length = seq_length
        self.input_size = 4
        self.output_size = 2  # v_linear, v_angular
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.learning_rate = learning_rate
        self.batch_size = 1024
        self.num_epochs = 100

        if self.data is not None:
            self.preprocess_data()
        
        self.model = self.LSTM_Model(
            input_size=4,
            hidden_size=self.hidden_size,
            output_size=self.output_size,
            num_layers=self.num_layers,
        )

        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

    def preprocess_data(self):

        self.data[['d1', 'd2']] = np.radians(self.data[['d1', 'd2']].values)

        X = self.data[['r1', 'r2', 'd1', 'd2']].values
        y = self.data[['v_linear', 'v_angular']].values
        # points = self.data.iloc[:, 6:].values  # Assuming point data starts from column 5 onward

        # X = np.concatenate((X, points), axis=1)

        # self.scaler_X = StandardScaler()
        # X_scaled = self.scaler_X.fit_transform(X)
        # self.scaler_y = StandardScaler()
        # y_scaled = self.scaler_y.fit_transform(y)

        X_seq, y_seq = [], []
        for i in range(len(X) - self.seq_length):
            X_seq.append(X[i:i + self.seq_length])
            y_seq.append(y[i + self.seq_length])

        self.X_seq = np.array(X_seq)
        self.y_seq = np.array(y_seq)


        # split_index = 1000

        # # Manually split the data
        # self.X_train, self.X_test = self.X_seq[split_index:], self.X_seq[:split_index]
        # self.y_train, self.y_test = self.y_seq[split_index:], self.y_seq[:split_index]


        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X_seq, self.y_seq, test_size=0.20, random_state=None)

        self.X_train_tensor = torch.tensor(self.X_train, dtype=torch.float32)
        self.y_train_tensor = torch.tensor(self.y_train, dtype=torch.float32)
        self.X_test_tensor = torch.tensor(self.X_test, dtype=torch.float32)
        self.y_test_tensor = torch.tensor(self.y_test, dtype=torch.float32)

        train_dataset = TensorDataset(self.X_train_tensor, self.y_train_tensor)

        # Step 3: Create DataLoaders
        self.train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        # self.test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)



        # # Plot predictions vs actual values
        # plt.figure(figsize=(12, 6))

        # # Plot for v_linear
        # plt.subplot(1, 2, 1)
        # plt.plot(self.y_seq[:, 0], label='Ground Truth (v_linear)', color='blue', linestyle='--')
        # # plt.plot(y_pred_test_numpy[:, 0], label='Predicted (v_linear)', color='red', linestyle='-')
        # plt.title('Linear Velocity (v_linear)')
        # plt.xlabel('Sample Index')
        # plt.ylabel('Linear Velocity')
        # plt.legend()

        # # Plot for v_angular
        # plt.subplot(1, 2, 2)
        # plt.plot(self.y_seq[:, 1], label='Ground Truth (v_angular)', color='blue', linestyle='--')
        # # plt.plot(y_pred_test_numpy[:, 1], label='Predicted (v_angular)', color='red', linestyle='-')
        # plt.title('Angular Velocity (v_angular)')
        # plt.xlabel('Sample Index')
        # plt.ylabel('Angular Velocity')
        # plt.legend()

        # plt.tight_layout()
        # plt.show()
    class LSTM_Model(nn.Module):
        def __init__(self, input_size, hidden_size, output_size, num_layers=1):
            super(UWBFollowModel.LSTM_Model, self).__init__()

            self.input_size = input_size
            self.hidden_size = hidden_size
            self.num_layers = num_layers
            self.output_size = output_size

            self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
            self.fc = nn.Linear(hidden_size, output_size)

        def forward(self, x):
            # print(np.shape(x))
            lstm_out, _ = self.lstm(x)
            lstm_out = lstm_out[:, -1, :]  # Use the output of the last time step
    
            output = self.fc(lstm_out)
            return output

    def train(self):
        loss_list= []
        if self.data is None:
            print("No data available to train the model.")
            return
        
        for epoch in range(self.num_epochs):
            self.model.train()  # Set model to training mode
            for batch_idx, (X_batch, y_batch) in enumerate(self.train_loader):

                y_pred = self.model(X_batch)
        
                # Compute the loss
                loss = self.criterion(y_pred, y_batch)
                
                # Backward pass: Compute gradients
                self.optimizer.zero_grad()
                loss.backward()
                loss_list.append(loss.item())
                
                # Update the weights
                self.optimizer.step()

            if (epoch + 1) % 100 == 0:  # Print loss every 100 epochs
                print(f'Epoch [{epoch+1}/{self.num_epochs}], Loss: {loss.item():.4f}')
        plt.plot(loss_list)
        plt.show()
    
    def evaluate(self):
        self.model.eval()  # Set the actual model (LSTM_Model) to evaluation mode
        with torch.no_grad():  # Disable gradient calculation during evaluation
            start_time = time.time()  # Record start time
            y_pred_test = self.model(self.X_test_tensor)
            end_time = time.time()  # Record end time
            
            inference_time = end_time - start_time  # Calculate inference time
            print(f"Inference time: {inference_time:.6f} seconds")
            
            # Calculate loss (for both training and testing)
            test_loss = self.criterion(y_pred_test, self.y_test_tensor)
            print(f'Test Loss: {test_loss.item():.4f}')
            
            # Convert the predictions and actual values to numpy for further calculations
            y_pred_test_numpy = y_pred_test.numpy()
            y_test_numpy = self.y_test_tensor.numpy()

            # y_pred_test_original = self.scaler_y.inverse_transform(y_pred_test.numpy())
            # y_test_original = self.scaler_y.inverse_transform(self.y_test_tensor.numpy())

            # Calculate Mean Absolute Error (MAE) and Root Mean Squared Error (RMSE)
            mae = np.mean(np.abs(y_pred_test_numpy - y_test_numpy))
            rmse = np.sqrt(np.mean((y_pred_test_numpy - y_test_numpy) ** 2))

            print(f"Test Mean Absolute Error (MAE): {mae:.4f}")
            print(f"Test Root Mean Squared Error (RMSE): {rmse:.4f}")

            # Plot predictions vs actual values
            plt.figure(figsize=(12, 6))
            print(np.shape(y_test_numpy))

            # Plot for v_linear
            plt.subplot(1, 2, 1)
            plt.plot(y_test_numpy[:, 0], label='Ground Truth (v_linear)', color='blue', linestyle='--')
            plt.plot(y_pred_test_numpy[:, 0], label='Predicted (v_linear)', color='red', linestyle='-')
            plt.title('Linear Velocity (v_linear)')
            plt.xlabel('Sample Index')
            plt.ylabel('Linear Velocity')
            plt.legend()

            # Plot for v_angular
            plt.subplot(1, 2, 2)
            plt.plot(y_test_numpy[:, 1], label='Ground Truth (v_angular)', color='blue', linestyle='--')
            plt.plot(y_pred_test_numpy[:, 1], label='Predicted (v_angular)', color='red', linestyle='-')
            plt.title('Angular Velocity (v_angular)')
            plt.xlabel('Sample Index')
            plt.ylabel('Angular Velocity')
            plt.legend()

            plt.tight_layout()
            # plt.show()
            
            # Return predictions and ground truth for further analysis
            return y_pred_test_numpy, y_test_numpy
    def evaluate_train(self):
        self.model.eval()  # Set the actual model (LSTM_Model) to evaluation mode
        with torch.no_grad():  # Disable gradient calculation during evaluation
            y_pred_train = self.model(self.X_train_tensor)
                        
            # Calculate loss (for both training and testing)
            train_loss = self.criterion(y_pred_train, self.y_train_tensor)
            print(f'Train Loss: {train_loss.item():.4f}')
            
            # Convert the predictions and actual values to numpy for further calculations
            y_pred_train_numpy = y_pred_train.numpy()
            y_train_numpy = self.y_train_tensor.numpy()

            # y_pred_test_original = self.scaler_y.inverse_transform(y_pred_test.numpy())
            # y_test_original = self.scaler_y.inverse_transform(self.y_test_tensor.numpy())

            # Calculate Mean Absolute Error (MAE) and Root Mean Squared Error (RMSE)
            mae = np.mean(np.abs(y_pred_train_numpy - y_train_numpy))
            rmse = np.sqrt(np.mean((y_pred_train_numpy - y_train_numpy) ** 2))

            print(f"Train Mean Absolute Error (MAE): {mae:.4f}")
            print(f"Train Root Mean Squared Error (RMSE): {rmse:.4f}")

            # Plot predictions vs actual values
            plt.figure(figsize=(12, 6))
            print(np.shape(y_train_numpy))

            # Plot for v_linear
            plt.subplot(1, 2, 1)
            plt.plot(y_train_numpy[:, 0], label='Ground Truth (v_linear)', color='blue', linestyle='--')
            plt.plot(y_pred_train_numpy[:, 0], label='Predicted (v_linear)', color='red', linestyle='-')
            plt.title('Linear Velocity (v_linear)')
            plt.xlabel('Sample Index')
            plt.ylabel('Linear Velocity')
            plt.legend()

            # Plot for v_angular
            plt.subplot(1, 2, 2)
            plt.plot(y_train_numpy[:, 1], label='Ground Truth (v_angular)', color='blue', linestyle='--')
            plt.plot(y_pred_train_numpy[:, 1], label='Predicted (v_angular)', color='red', linestyle='-')
            plt.title('Angular Velocity (v_angular)')
            plt.xlabel('Sample Index')
            plt.ylabel('Angular Velocity')
            plt.legend()

            plt.tight_layout()
            # plt.show()
            
            # Return predictions and ground truth for further analysis
            return y_pred_train_numpy, y_train_numpy
    def predict(self, new_data):
        # Example: Predicting new cmd_vel commands
        if self.data is None:
            print("No data available to predict with.")
            return None
        
        self.model.eval()
        new_data[:, 2:4] = np.radians(new_data[:, 2:4])
        # Ensure new data is in the shape (10, 4) for a single sequence
        new_data_reshaped = new_data.reshape(1, self.seq_length, self.input_size)  # Add batch dimension
        
        # Get the predicted velocity commands
        with torch.no_grad():
            prediction = self.model(torch.tensor(new_data_reshaped, dtype=torch.float32))
        
        # Return the predicted velocity (no need to inverse scale as we aren't scaling data)
        # predicted_velocity = self.scaler_y.inverse_transform(prediction.numpy())
        return prediction.numpy()

    def save_model(self, model_path):
        if self.data is None:
            print("No model to save.")
            return
        
        # Save the model parameters
        torch.save(self.model.state_dict(), model_path)
        print(f"Model saved to {model_path}.")

    def load_model(self, model_path):
        # Load the model parameters
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()  # Set model to evaluation mode
        print(f"Model loaded from {model_path}.")

    def eval(self):
        # Call eval on the underlying model
        self.model.eval() 

    def export_to_onnx(self, onnx_filename="models/onnx/scan_lstm_10.onnx"):
        """
        Export the trained model to ONNX format.
        """
        # Set the model to evaluation mode before exporting
        self.model.eval()
        
        # Create dummy input with the same shape as the input data used in training
        # Here the dummy input is a single batch (1 sample) with 4 features (input size = 4)
        dummy_input = torch.ones(1, self.seq_length, self.input_size)  # Batch size of 1, seq_length = 1, input_size = 4
        
        # Export the model to ONNX
        torch.onnx.export(
            self.model,               # Model to export
            dummy_input,              # Dummy input tensor
            onnx_filename,            # Path to save the ONNX model
            export_params=True,       # Export the parameters (weights)
            opset_version=11,         # ONNX version (ensure compatibility)
            do_constant_folding=True, # Fold constants for optimization
            input_names=['input'],    # Input tensor name
            output_names=['output'],  # Output tensor name
            dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}  # Support variable batch size
        )

        print(f"Model has been exported to {onnx_filename}.")

def main():
    # Initialize the model
    model = UWBFollowModel(csv_filename='data/csv/free/coop.csv')

    # Train the model (only if csv_filename is provided)
    if model.data is not None:
        model.train()
        # model.save_model(model_path='models/torch/scan_lstm_100seq_60_60scan.pth')


    # model.export_to_onnx(onnx_filename='models/onnx/uwb_lstm_100_2.onnx')

    # Evaluate the model (only if csv_filename is provided)

    if model.data is not None:
        predictions, ground_truth = model.evaluate_train()

    if model.data is not None:
        predictions, ground_truth = model.evaluate()


    # for i in range(500):  # Displaying first 5 predictions and actual values for comparison
    #     print(f"Prediction: {predictions[i]} | Actual: {ground_truth[i]}")

# Test the model with new data
    # new_data = np.array([
    #     [3.187000036239624,3.1459999084472656,-28.93000030517578,28.030000686645508],  # First time step
    #     [3.180000066757202,3.140000104904175,-28.25,28.149999618530273],  # Second time step
    #     [3.1679999828338623,3.1429998874664307,-28.350000381469727,28.780000686645508],  # Third time step
    #     [3.1589999198913574,3.1570000648498535,-27.8799991607666,28.739999771118164],  # Fourth time step
    #     [3.1449999809265137,3.1640000343322754,-27.90999984741211,28.299999237060547],  # Fifth time step
    #     [3.1410000324249268,3.1679999828338623,-27.280000686645508,28.540000915527344],  # Sixth time step
    #     [3.1470000743865967,3.1700000762939453,-26.90999984741211,28.6299991607666],  # Seventh time step
    #     [3.1480000019073486,3.1689999103546143,-27.1200008392334,28.84000015258789],  # Eighth time step
    #     [3.1449999809265137,3.1619999408721924,-27.219999313354492,28.25],  # Ninth time step
    #     [3.1480000019073486,3.1679999828338623,-27.549999237060547,28.739999771118164]   # Tenth time step
    # ])
    # # new_data = np.array([
    # #     [3.187000036239624,3.1459999084472656,-28.93000030517578,28.030000686645508]])
    # predicted_velocity = model.predict(new_data)
    # print("Predicted velocity:", predicted_velocity)

if __name__ == "__main__":
    main()