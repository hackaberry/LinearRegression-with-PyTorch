## Overview
This project demonstrates a simple linear regression neural network built using PyTorch. It includes an interactive GUI that visualizes how the model learns to fit a line through training data in real-time, updating weights and bias as it optimizes through epochs. This serves as an intuitive introduction to fundamental concepts in machine learning and neural network training.

---
## Features
- Linear regression model implemented with PyTorch’s `nn.Linear`
- Interactive GUI built with Tkinter for live visualization of training progress
- Real-time plots showing training data, test data, predictions, and loss curve
- Adjustable training epochs through GUI buttons to observe learning dynamics
- Device agnostic code supporting CPU and GPU (if available)

---
## Installation

1. Clone the repo:
```bash
git clone https://github.com/hackaberry/LinearRegression-with-PyTorch.git
```
2. Install dependencies:
```bash
pip install torch matplotlib tk
```
3. Run the program:
```bash
python main.py
```
