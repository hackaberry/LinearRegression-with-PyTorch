# Author: Anthony Madrigal Calderon
# Date: 07/19/2025
# Project Name: Learn2Fit (Linear Regression Neural Network)

import torch
from torch import nn
from torch import optim
from LinearRegressionModel import LinearRegressionModel
from matplotlib import pyplot as plt
from typing import Union
import tkinter as tk
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg,NavigationToolbar2Tk)

def plot_data(x_train,y_train,x_test,y_test,epoch_count,loss_count,predictions: Union[torch.Tensor, bool] = False):

	learning_graph.clear()
	loss_graph.clear()

	learning_graph.scatter(x_train,y_train, c="0.8", label="Training Data")
	learning_graph.scatter(x_test,y_test, c="0", label="Testing Data")

	if predictions is not False:
		learning_graph.scatter(x_test,predictions, c="g", label="Prediction Data")
		loss_graph.plot(epoch_count,loss_count, c="g", label="Loss Graph")

	learning_graph.legend()
	loss_graph.legend()
	canvas.draw()

def training_loop(model,next_n_epochs):
	global epoch

	# Train the model for n number of epochs
	for x in range(next_n_epochs):
		model.train()

		loss_function = nn.L1Loss()
		optimizer = optim.SGD(model.parameters(), lr=0.01)

		training_predictions = model(x_train)
		training_loss = loss_function(training_predictions,y_train)
		loss_count.append(training_loss)

		optimizer.zero_grad()
		training_loss.backward()
		optimizer.step()

		epoch +=1
		epoch_count.append(epoch)

	# Evaluate the performance using testing data
	model.eval()
	with torch.inference_mode():
		testing_predictions = model(x_test) # Make inferences
		plot_data(x_train,y_train,x_test,y_test,epoch_count,loss_count,testing_predictions.cpu())

	# Update learned parameters on GUI
	predicted_parameters.config(text=f"Predicted Weight: {model.state_dict()['layer.weight'][0][0].item():.2f}\n"
								     f"Predicted Bias: {model.state_dict()['layer.bias'][0].item():.2f}")


# For graphing the loss function
epoch = 0
epoch_count = []
loss_count = []

# Arbitrary weight & bias that the model will learn
weight = 0.7
bias = 0.3

# x-axis range
start = 0
finish = 1
step = 0.02

device = "cuda" if torch.cuda.is_available() else "cpu" # Device agnostic

X = torch.arange(start,finish,step).unsqueeze(1) # Linear layer requires 2D tensor (batch_size,data)
y = weight * X + bias

data_split = int(len(X) * 0.8) # 80/20 split of dataset

x_train, y_train = X[:data_split].to(device=device), y[:data_split].to(device=device)
x_test, y_test = X[data_split:].to(device=device), y[data_split:].to(device=device)


model = LinearRegressionModel()
model.to(device)

# Create the GUI window
root = tk.Tk()
root.title("Learn2Fit")
root.geometry("1000x1000")
root.configure(bg="lightgray")


# Initialize both graphs and plot all pre-training data
fig = plt.figure(figsize=(10, 4))
learning_graph = fig.add_subplot(1, 2, 1) # Left graph
loss_graph = fig.add_subplot(1,2,2) # Right graph
canvas = FigureCanvasTkAgg(fig, master=root) # Embed graphs in Tk window
learning_graph.scatter(x_train,y_train, c="0.8", label="Training Data")
learning_graph.scatter(x_test,y_test, c="0", label="Testing Data")
canvas.get_tk_widget().pack()
learning_graph.legend()
canvas.draw()
learning_graph.set_xlabel("X")
learning_graph.set_ylabel("y")
loss_graph.set_xlabel("Epoch")
loss_graph.set_ylabel("Loss")


# Create some tkinter elements for interactiveness
next_10_epochs = tk.Button(root,width=20,height=5,bg="lightblue", text="Next 10 epochs", command=lambda: training_loop(model,10))
next_10_epochs.place(relx=0.35,rely=0.6)

next_50_epochs = tk.Button(root,width=20,height=5,bg="lightblue", text="Next 50 epochs", command=lambda: training_loop(model,50))
next_50_epochs.place(relx=0.5,rely=0.6)

actual_parameters = tk.Label(root,text=f"Actual Weight: {weight}\nActual Bias: {bias}")
actual_parameters.place(relx=0.40,rely=0.54)

predicted_parameters = tk.Label(root, text="Predicted parameters")
predicted_parameters.place(relx=0.53,rely=0.54)

root.mainloop()