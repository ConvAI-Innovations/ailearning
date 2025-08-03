"""
LLM Training Learning Dashboard - Session-Based Learning System
Complete structured curriculum from Python fundamentals to LLM inference,
with manual coding practice and AI mentor guidance.
"""

import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext, filedialog, font
import threading
import sys
import io
import os
import re
import time
import urllib.request
import asyncio
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, Dict, List
import concurrent.futures
import json

# --- Dependency Checks ---
try:
    from llama_cpp import Llama
    LLAMA_CPP_AVAILABLE = True
except ImportError:
    LLAMA_CPP_AVAILABLE = False

try:
    from kokoro.pipeline import KPipeline
    import torch
    import sounddevice as sd
    KOKORO_TTS_AVAILABLE = True
except ImportError:
    KOKORO_TTS_AVAILABLE = False

# --- Data Classes ---
@dataclass
class Session:
    """Represents a learning session"""
    id: str
    title: str
    description: str
    reference_code: str
    learning_objectives: List[str]
    hints: List[str]
    completed: bool = False
    
@dataclass
class LearningProgress:
    """Tracks overall learning progress"""
    current_session_id: str = "python_fundamentals"
    completed_sessions: List[str] = field(default_factory=list)
    session_scores: Dict[str, float] = field(default_factory=dict)
    total_sessions: int = 0
    
    def get_completion_percentage(self) -> float:
        if self.total_sessions == 0:
            return 0.0
        return (len(self.completed_sessions) / self.total_sessions) * 100

# --- Core Systems ---

class SessionManager:
    """Manages learning sessions and progression"""
    
    def __init__(self):
        self.sessions = self._create_sessions()
        self.progress = LearningProgress(total_sessions=len(self.sessions))
        
    def _create_sessions(self) -> Dict[str, Session]:
        """Create all learning sessions"""
        sessions = {}
        
        # Session 1: Python Fundamentals
        sessions["python_fundamentals"] = Session(
            id="python_fundamentals",
            title="🐍 Python Fundamentals",
            description="""
# Python Fundamentals for Machine Learning

Learn essential Python concepts needed for ML/AI development:
- Variables and data types
- Functions and classes
- Lists and dictionaries
- Control flow (loops, conditionals)
- File handling and imports

These fundamentals are crucial for understanding ML code!
""",
            reference_code="""# Python Fundamentals for ML/AI
# Variables and data types essential for ML
learning_rate = 0.001  # float for hyperparameters
batch_size = 32        # int for training
model_name = "GPT"     # string for identifiers
is_training = True     # boolean for flags

# Lists for storing data (like training examples)
training_data = [1, 2, 3, 4, 5]
layer_sizes = [784, 256, 128, 10]

# Dictionaries for configuration
config = {
    "learning_rate": 0.001,
    "epochs": 100,
    "optimizer": "adam"
}

# Functions (building blocks of ML code)
def calculate_accuracy(predictions, targets):
    correct = sum(p == t for p, t in zip(predictions, targets))
    return correct / len(targets)

# Classes for organizing ML components
class ModelConfig:
    def __init__(self, lr=0.001, epochs=100):
        self.learning_rate = lr
        self.epochs = epochs
        self.optimizer = "adam"
    
    def __str__(self):
        return f"Config(lr={self.learning_rate}, epochs={self.epochs})"

# Test the concepts
predictions = [1, 0, 1, 1, 0]
targets = [1, 0, 1, 0, 0]
accuracy = calculate_accuracy(predictions, targets)
model_config = ModelConfig()

print(f"Training data: {training_data}")
print(f"Model accuracy: {accuracy:.2f}")
print(f"Configuration: {model_config}")
print(f"Layer sizes: {layer_sizes}")""",
            learning_objectives=[
                "Understand variable types used in ML",
                "Write functions for ML computations", 
                "Use classes to organize code",
                "Work with lists and dictionaries",
                "Apply Python basics to ML scenarios"
            ],
            hints=[
                "Variables store values - think of them as labeled boxes",
                "Functions help organize code - like recipes for computations",
                "Classes group related functions and data together",
                "Lists store sequences - perfect for datasets",
                "Dictionaries store key-value pairs - great for configs"
            ]
        )
        
        # Session 2: PyTorch and NumPy Operations
        sessions["pytorch_numpy"] = Session(
            id="pytorch_numpy",
            title="🔢 PyTorch & NumPy Operations",
            description="""
# PyTorch and NumPy Fundamentals

Master tensor operations and numerical computing:
- Creating and manipulating tensors
- Mathematical operations
- Reshaping and indexing
- Broadcasting and reduction operations
- GPU acceleration basics

Foundation for all neural network computations!
""",
            reference_code="""# PyTorch and NumPy Operations for Deep Learning
import torch
import numpy as np

print("🔢 Tensor Creation and Basic Operations")

# Creating tensors (the building blocks of neural networks)
tensor_1d = torch.tensor([1, 2, 3, 4, 5])
tensor_2d = torch.tensor([[1, 2], [3, 4], [5, 6]])
random_tensor = torch.randn(3, 4)  # Random normal distribution
zeros_tensor = torch.zeros(2, 3)
ones_tensor = torch.ones(4, 4)

print(f"1D tensor: {tensor_1d}")
print(f"2D tensor shape: {tensor_2d.shape}")
print(f"Random tensor:\\n{random_tensor}")

# Mathematical operations (essential for neural networks)
a = torch.tensor([1.0, 2.0, 3.0])
b = torch.tensor([4.0, 5.0, 6.0])

# Element-wise operations
addition = a + b
multiplication = a * b
power = torch.pow(a, 2)

print(f"\\nElement-wise addition: {addition}")
print(f"Element-wise multiplication: {multiplication}")
print(f"Square: {power}")

# Matrix operations (core of neural networks)
matrix_a = torch.randn(3, 4)
matrix_b = torch.randn(4, 2)
matrix_product = torch.matmul(matrix_a, matrix_b)  # Matrix multiplication

print(f"\\nMatrix multiplication result shape: {matrix_product.shape}")

# Reshaping (crucial for neural network layers)
original = torch.randn(2, 3, 4)
reshaped = original.view(2, 12)  # Flatten last two dimensions
flattened = original.flatten()

print(f"Original shape: {original.shape}")
print(f"Reshaped: {reshaped.shape}")
print(f"Flattened: {flattened.shape}")

# Reduction operations (used in loss functions)
data = torch.randn(3, 4)
mean_val = torch.mean(data)
sum_val = torch.sum(data)
max_val = torch.max(data)

print(f"\\nMean: {mean_val:.4f}")
print(f"Sum: {sum_val:.4f}")
print(f"Max: {max_val:.4f}")

# Gradients (automatic differentiation for learning)
x = torch.tensor([2.0], requires_grad=True)
y = x ** 2 + 3 * x + 1
y.backward()  # Compute dy/dx

print(f"\\nInput: {x.item()}")
print(f"Output: {y.item()}")
print(f"Gradient dy/dx: {x.grad.item()}")""",
            learning_objectives=[
                "Create and manipulate PyTorch tensors",
                "Perform mathematical operations on tensors",
                "Understand matrix multiplication for neural networks",
                "Master reshaping and indexing operations",
                "Learn automatic differentiation basics"
            ],
            hints=[
                "Tensors are like NumPy arrays but with GPU support and gradients",
                "Matrix multiplication is the core operation in neural networks",
                "View() and reshape() change tensor dimensions without copying data",
                "requires_grad=True enables automatic gradient computation",
                "Always check tensor shapes - mismatched shapes cause errors"
            ]
        )
        
        # Session 3: Neural Network Fundamentals
        sessions["neural_networks"] = Session(
            id="neural_networks",
            title="🧠 Neural Network Fundamentals",
            description="""
# Neural Network Building Blocks

Understand the core components of neural networks:
- Perceptrons and multi-layer networks
- Linear layers and activations
- Forward propagation
- nn.Module and PyTorch structure
- Simple network architectures

Building towards transformer understanding!
""",
            reference_code="""# Neural Network Fundamentals with PyTorch
import torch
import torch.nn as nn
import torch.nn.functional as F

print("🧠 Neural Network Building Blocks")

# Single neuron (perceptron) - the basic unit
class Perceptron(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.linear = nn.Linear(input_size, 1)  # W*x + b
    
    def forward(self, x):
        return torch.sigmoid(self.linear(x))  # Sigmoid activation

# Multi-layer neural network
class SimpleNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.layer2 = nn.Linear(hidden_size, hidden_size)
        self.layer3 = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(0.2)
    
    def forward(self, x):
        # Forward propagation through layers
        x = F.relu(self.layer1(x))      # Hidden layer 1 + ReLU
        x = self.dropout(x)             # Dropout for regularization
        x = F.relu(self.layer2(x))      # Hidden layer 2 + ReLU
        x = self.layer3(x)              # Output layer (no activation)
        return x

# Create sample data
batch_size = 4
input_size = 10
hidden_size = 20
output_size = 5

# Sample input (like features from an embedding)
sample_input = torch.randn(batch_size, input_size)

print(f"Input shape: {sample_input.shape}")

# Test perceptron
perceptron = Perceptron(input_size)
perceptron_output = perceptron(sample_input)
print(f"Perceptron output shape: {perceptron_output.shape}")

# Test multi-layer network
model = SimpleNet(input_size, hidden_size, output_size)
output = model(sample_input)
print(f"Multi-layer network output shape: {output.shape}")

# Count parameters (important for understanding model size)
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"\\nModel Architecture:")
print(f"Total parameters: {total_params:,}")
print(f"Trainable parameters: {trainable_params:,}")

# Demonstrate parameter access
print(f"\\nFirst layer weights shape: {model.layer1.weight.shape}")
print(f"First layer bias shape: {model.layer1.bias.shape}")

# Show how gradients work
loss = torch.mean(output ** 2)  # Dummy loss
loss.backward()

print(f"\\nAfter backward pass:")
print(f"First layer weight gradients shape: {model.layer1.weight.grad.shape}")
print("✅ Neural network fundamentals complete!")""",
            learning_objectives=[
                "Understand perceptrons and multi-layer networks",
                "Build networks using nn.Module",
                "Implement forward propagation",
                "Use activation functions effectively",
                "Count and understand model parameters"
            ],
            hints=[
                "nn.Linear performs matrix multiplication: y = Wx + b",
                "Activation functions add non-linearity between layers",
                "nn.Module is the base class for all neural network components",
                "Forward() defines how data flows through the network",
                "Dropout prevents overfitting by randomly zeroing neurons"
            ]
        )
        
        # Session 4: Backpropagation
        sessions["backpropagation"] = Session(
            id="backpropagation",
            title="⬅️ Backpropagation",
            description="""
# Backpropagation - How Neural Networks Learn

Understanding the learning mechanism:
- Chain rule and gradients
- Forward and backward passes
- Gradient computation
- Parameter updates
- Manual vs automatic differentiation

The foundation of all neural network training!
""",
            reference_code="""# Backpropagation - How Neural Networks Learn
import torch
import torch.nn as nn

print("⬅️ Understanding Backpropagation")

# Simple example to demonstrate backpropagation
class TinyNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.w1 = nn.Parameter(torch.tensor([[0.5, -0.2], [0.3, 0.1]]))
        self.b1 = nn.Parameter(torch.tensor([0.1, -0.1]))
        self.w2 = nn.Parameter(torch.tensor([[0.4], [0.6]]))
        self.b2 = nn.Parameter(torch.tensor([0.0]))
    
    def forward(self, x):
        # Forward pass step by step
        z1 = torch.matmul(x, self.w1) + self.b1  # Linear transformation
        a1 = torch.relu(z1)                       # Activation
        z2 = torch.matmul(a1, self.w2) + self.b2 # Output layer
        return z2

# Create model and data
model = TinyNetwork()
x = torch.tensor([[1.0, 0.5]])  # Input
target = torch.tensor([[1.0]])   # Target output

print("Initial parameters:")
print(f"W1: {model.w1.data}")
print(f"W2: {model.w2.data}")

# Forward pass
output = model(x)
print(f"\\nForward pass:")
print(f"Input: {x}")
print(f"Output: {output}")
print(f"Target: {target}")

# Compute loss
loss = 0.5 * (output - target) ** 2  # MSE loss
print(f"Loss: {loss.item():.4f}")

# Backward pass (automatic differentiation)
loss.backward()

print(f"\\nGradients after backward pass:")
print(f"dL/dW1: {model.w1.grad}")
print(f"dL/dW2: {model.w2.grad}")

# Manual parameter update (what optimizers do)
learning_rate = 0.1
with torch.no_grad():
    model.w1 -= learning_rate * model.w1.grad
    model.w2 -= learning_rate * model.w2.grad
    model.b1 -= learning_rate * model.b1.grad
    model.b2 -= learning_rate * model.b2.grad

print(f"\\nParameters after update:")
print(f"Updated W1: {model.w1.data}")
print(f"Updated W2: {model.w2.data}")

# Demonstrate the complete training step
def training_step(model, x, target, lr=0.1):
    # Zero gradients
    model.zero_grad()
    
    # Forward pass
    output = model(x)
    loss = 0.5 * (output - target) ** 2
    
    # Backward pass
    loss.backward()
    
    # Update parameters
    with torch.no_grad():
        for param in model.parameters():
            param -= lr * param.grad
    
    return loss.item()

# Train for a few steps
print(f"\\nTraining demonstration:")
for step in range(5):
    loss_val = training_step(model, x, target)
    output = model(x)
    print(f"Step {step+1}: Loss = {loss_val:.4f}, Output = {output.item():.4f}")

print("\\n✅ Backpropagation complete! This is how all neural networks learn.")""",
            learning_objectives=[
                "Understand gradient computation through chain rule",
                "See how forward and backward passes work together",
                "Learn parameter update mechanics",
                "Compare manual vs automatic differentiation",
                "Implement a complete training step"
            ],
            hints=[
                "Forward pass: compute output from input",
                "Backward pass: compute gradients from loss to parameters",
                "Chain rule: multiply gradients through connected operations",
                "Zero gradients before each backward pass",
                "Parameter update: param = param - lr * gradient"
            ]
        )
        
        # Session 5: Regularization
        sessions["regularization"] = Session(
            id="regularization",
            title="🛡️ Regularization Techniques",
            description="""
# Regularization - Preventing Overfitting

Techniques to make models generalize better:
- L1 and L2 regularization
- Dropout
- Batch normalization
- Early stopping
- Data augmentation concepts

Essential for robust model training!
""",
            reference_code="""# Regularization Techniques for Better Generalization
import torch
import torch.nn as nn
import torch.nn.functional as F

print("🛡️ Regularization Techniques")

# Network with various regularization techniques
class RegularizedNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, dropout_rate=0.3):
        super().__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.layer2 = nn.Linear(hidden_size, hidden_size)
        self.layer3 = nn.Linear(hidden_size, output_size)
        
        # Regularization layers
        self.dropout = nn.Dropout(dropout_rate)
        self.batch_norm1 = nn.BatchNorm1d(hidden_size)
        self.batch_norm2 = nn.BatchNorm1d(hidden_size)
        
    def forward(self, x):
        # Layer 1 with batch norm and dropout
        x = self.layer1(x)
        x = self.batch_norm1(x)
        x = F.relu(x)
        x = self.dropout(x)
        
        # Layer 2 with batch norm and dropout
        x = self.layer2(x)
        x = self.batch_norm2(x)
        x = F.relu(x)
        x = self.dropout(x)
        
        # Output layer (no regularization)
        x = self.layer3(x)
        return x

# L1 and L2 Regularization Functions
def l1_regularization(model, lambda_l1=0.01):
    \"\"\"Compute L1 regularization term\"\"\"
    l1_penalty = 0
    for param in model.parameters():
        l1_penalty += torch.sum(torch.abs(param))
    return lambda_l1 * l1_penalty

def l2_regularization(model, lambda_l2=0.01):
    \"\"\"Compute L2 regularization term\"\"\"
    l2_penalty = 0
    for param in model.parameters():
        l2_penalty += torch.sum(param ** 2)
    return lambda_l2 * l2_penalty

# Create model and sample data
model = RegularizedNet(input_size=20, hidden_size=50, output_size=10)
batch_size = 8
sample_input = torch.randn(batch_size, 20)
sample_target = torch.randint(0, 10, (batch_size,))

# Forward pass
output = model(sample_input)
base_loss = F.cross_entropy(output, sample_target)

print(f"Base loss: {base_loss.item():.4f}")

# Add regularization terms
l1_penalty = l1_regularization(model, lambda_l1=0.001)
l2_penalty = l2_regularization(model, lambda_l2=0.001)
total_loss = base_loss + l1_penalty + l2_penalty

print(f"L1 penalty: {l1_penalty.item():.4f}")
print(f"L2 penalty: {l2_penalty.item():.4f}")
print(f"Total loss: {total_loss.item():.4f}")

# Demonstrate dropout behavior
print(f"\\nDropout demonstration:")
model.train()  # Enable dropout
output_train1 = model(sample_input)
output_train2 = model(sample_input)

model.eval()   # Disable dropout
output_eval1 = model(sample_input)
output_eval2 = model(sample_input)

print(f"Training mode outputs differ: {not torch.equal(output_train1, output_train2)}")
print(f"Evaluation mode outputs same: {torch.equal(output_eval1, output_eval2)}")

# Weight decay (L2 regularization in optimizer)
optimizer_with_decay = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.01)
optimizer_without_decay = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0)

print(f"\\nOptimizers created:")
print(f"With weight decay: L2 regularization = {0.01}")
print(f"Without weight decay: L2 regularization = {0.0}")

# Early stopping simulation
class EarlyStopping:
    def __init__(self, patience=5, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float('inf')
        self.counter = 0
        
    def should_stop(self, current_loss):
        if current_loss < self.best_loss - self.min_delta:
            self.best_loss = current_loss
            self.counter = 0
            return False
        else:
            self.counter += 1
            return self.counter >= self.patience

# Test early stopping
early_stopper = EarlyStopping(patience=3)
losses = [1.0, 0.8, 0.7, 0.69, 0.68, 0.679, 0.678]  # Simulated training losses

print(f"\\nEarly stopping demonstration:")
for epoch, loss in enumerate(losses):
    should_stop = early_stopper.should_stop(loss)
    print(f"Epoch {epoch+1}: Loss = {loss:.3f}, Stop = {should_stop}")
    if should_stop:
        print("Early stopping triggered!")
        break

print("\\n✅ Regularization techniques complete!")""",
            learning_objectives=[
                "Implement L1 and L2 regularization",
                "Use dropout for preventing overfitting",
                "Apply batch normalization for stable training",
                "Understand early stopping mechanisms",
                "Combine multiple regularization techniques"
            ],
            hints=[
                "L1 regularization promotes sparsity (zeros in weights)",
                "L2 regularization prevents large weights (weight decay)",
                "Dropout randomly zeros neurons during training only",
                "Batch normalization normalizes inputs to each layer",
                "Early stopping prevents overfitting by monitoring validation loss"
            ]
        )
        
        # Session 6: Loss Functions and Optimizers
        sessions["loss_optimizers"] = Session(
            id="loss_optimizers",
            title="📉 Loss Functions & Optimizers",
            description="""
# Loss Functions and Optimization

Master the learning mechanisms:
- Cross-entropy loss for classification
- MSE loss for regression
- Custom loss functions
- SGD vs Adam vs AdamW
- Learning rate scheduling
- Gradient clipping

Critical for effective model training!
""",
            reference_code="""# Loss Functions and Optimizers for Deep Learning
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math

print("📉 Loss Functions and Optimizers")

# Different Loss Functions
def demonstrate_losses():
    print("Loss Functions Demonstration:")
    
    # Classification with Cross-Entropy
    batch_size = 4
    num_classes = 5
    logits = torch.randn(batch_size, num_classes)  # Raw model output
    targets = torch.tensor([1, 0, 3, 2])           # True class indices
    
    ce_loss = F.cross_entropy(logits, targets)
    print(f"Cross-Entropy Loss: {ce_loss.item():.4f}")
    
    # Regression with MSE
    predictions = torch.tensor([2.5, 1.8, 3.2, 0.9])
    true_values = torch.tensor([2.0, 2.0, 3.0, 1.0])
    mse_loss = F.mse_loss(predictions, true_values)
    print(f"MSE Loss: {mse_loss.item():.4f}")
    
    # Binary Classification with BCE
    binary_logits = torch.tensor([0.8, -0.3, 1.2, -0.9])
    binary_targets = torch.tensor([1.0, 0.0, 1.0, 0.0])
    bce_loss = F.binary_cross_entropy_with_logits(binary_logits, binary_targets)
    print(f"Binary Cross-Entropy Loss: {bce_loss.item():.4f}")

# Custom Loss Function (like Huber Loss)
class HuberLoss(nn.Module):
    def __init__(self, delta=1.0):
        super().__init__()
        self.delta = delta
    
    def forward(self, predictions, targets):
        diff = predictions - targets
        abs_diff = torch.abs(diff)
        
        # Quadratic for small errors, linear for large errors
        quadratic = torch.where(abs_diff <= self.delta, 
                               0.5 * diff ** 2,
                               self.delta * (abs_diff - 0.5 * self.delta))
        return torch.mean(quadratic)

# Simple model for optimizer comparison
class SimpleModel(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.linear1 = nn.Linear(input_size, 50)
        self.linear2 = nn.Linear(50, output_size)
    
    def forward(self, x):
        x = F.relu(self.linear1(x))
        return self.linear2(x)

# Optimizer Comparison
def compare_optimizers():
    print("\\nOptimizer Comparison:")
    
    # Create identical models
    model_sgd = SimpleModel(10, 1)
    model_adam = SimpleModel(10, 1)
    model_adamw = SimpleModel(10, 1)
    
    # Copy weights to make fair comparison
    model_adam.load_state_dict(model_sgd.state_dict())
    model_adamw.load_state_dict(model_sgd.state_dict())
    
    # Different optimizers
    opt_sgd = optim.SGD(model_sgd.parameters(), lr=0.01, momentum=0.9)
    opt_adam = optim.Adam(model_adam.parameters(), lr=0.001)
    opt_adamw = optim.AdamW(model_adamw.parameters(), lr=0.001, weight_decay=0.01)
    
    # Sample data
    x = torch.randn(32, 10)
    y = torch.randn(32, 1)
    
    # Training steps comparison
    models = [model_sgd, model_adam, model_adamw]
    optimizers = [opt_sgd, opt_adam, opt_adamw]
    names = ["SGD", "Adam", "AdamW"]
    
    for step in range(5):
        print(f"Step {step + 1}:")
        for model, optimizer, name in zip(models, optimizers, names):
            optimizer.zero_grad()
            output = model(x)
            loss = F.mse_loss(output, y)
            loss.backward()
            optimizer.step()
            print(f"  {name:6}: Loss = {loss.item():.4f}")

# Learning Rate Scheduling
def demonstrate_lr_scheduling():
    print("\\nLearning Rate Scheduling:")
    
    model = SimpleModel(10, 1)
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    
    # Different schedulers
    step_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)
    cosine_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
    
    print("Step LR Scheduler:")
    for epoch in range(10):
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch+1}: LR = {current_lr:.6f}")
        step_scheduler.step()
    
    # Reset for cosine scheduler
    for param_group in optimizer.param_groups:
        param_group['lr'] = 0.01
    
    print("\\nCosine Annealing Scheduler:")
    for epoch in range(10):
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch+1}: LR = {current_lr:.6f}")
        cosine_scheduler.step()

# Gradient Clipping
def demonstrate_gradient_clipping():
    print("\\nGradient Clipping:")
    
    model = SimpleModel(10, 1)
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    
    # Create sample with potential for large gradients
    x = torch.randn(4, 10) * 10  # Large input
    y = torch.randn(4, 1) * 10   # Large target
    
    # Without gradient clipping
    optimizer.zero_grad()
    output = model(x)
    loss = F.mse_loss(output, y)
    loss.backward()
    
    grad_norm_before = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float('inf'))
    print(f"Gradient norm before clipping: {grad_norm_before:.4f}")
    
    # With gradient clipping
    optimizer.zero_grad()
    output = model(x)
    loss = F.mse_loss(output, y)
    loss.backward()
    
    grad_norm_after = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    print(f"Gradient norm after clipping: {grad_norm_after:.4f}")

# Run demonstrations
demonstrate_losses()
compare_optimizers()
demonstrate_lr_scheduling()
demonstrate_gradient_clipping()

# Test custom loss
huber_loss = HuberLoss(delta=1.0)
preds = torch.tensor([1.0, 2.0, 3.0])
targets = torch.tensor([1.1, 2.5, 2.8])
custom_loss = huber_loss(preds, targets)
print(f"\\nCustom Huber Loss: {custom_loss.item():.4f}")

print("\\n✅ Loss functions and optimizers complete!")""",
            learning_objectives=[
                "Understand different loss functions and their use cases",
                "Compare SGD, Adam, and AdamW optimizers",
                "Implement learning rate scheduling",
                "Apply gradient clipping for stable training",
                "Create custom loss functions"
            ],
            hints=[
                "Cross-entropy for classification, MSE for regression",
                "Adam adapts learning rates, SGD uses fixed rates",
                "AdamW fixes weight decay implementation in Adam",
                "Learning rate scheduling improves convergence",
                "Gradient clipping prevents exploding gradients"
            ]
        )
        
        # Session 7: LLM Architecture
        sessions["llm_architecture"] = Session(
            id="llm_architecture",
            title="🏗️ LLM Architecture",
            description="""
# Language Model Architecture

Understanding transformer-based LLMs:
- Embedding layers
- Positional encoding
- Multi-head attention
- Feed-forward networks
- Layer normalization
- Residual connections

The complete architecture behind GPT and similar models!
""",
            reference_code="""# Complete LLM Architecture Implementation
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

print("🏗️ Language Model Architecture")

class PositionalEncoding(nn.Module):
    \"\"\"Sinusoidal positional encoding for transformers\"\"\"
    
    def __init__(self, d_model, max_seq_len=512):
        super().__init__()
        
        # Create positional encoding matrix
        pe = torch.zeros(max_seq_len, d_model)
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        
        # Create div_term for sine and cosine
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)  # Even indices
        pe[:, 1::2] = torch.cos(position * div_term)  # Odd indices
        
        self.register_buffer('pe', pe.unsqueeze(0))
    
    def forward(self, x):
        # Add positional encoding to input embeddings
        return x + self.pe[:, :x.size(1)]

class MultiHeadAttention(nn.Module):
    \"\"\"Multi-head self-attention mechanism\"\"\"
    
    def __init__(self, d_model, n_heads):
        super().__init__()
        assert d_model % n_heads == 0
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        # Linear projections for Q, K, V
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, x, mask=None):
        batch_size, seq_len, d_model = x.shape
        
        # Create Q, K, V
        Q = self.w_q(x)  # (batch, seq_len, d_model)
        K = self.w_k(x)
        V = self.w_v(x)
        
        # Reshape for multi-head attention
        Q = Q.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        
        # Scaled dot-product attention
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        # Apply causal mask for language modeling
        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask == 0, -1e9)
        
        attention_weights = F.softmax(attention_scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention to values
        context = torch.matmul(attention_weights, V)
        
        # Concatenate heads and project
        context = context.transpose(1, 2).contiguous().view(
            batch_size, seq_len, d_model)
        
        return self.w_o(context)

class FeedForwardNetwork(nn.Module):
    \"\"\"Position-wise feed-forward network\"\"\"
    
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, x):
        return self.linear2(self.dropout(F.relu(self.linear1(x))))

class TransformerBlock(nn.Module):
    \"\"\"Single transformer block with attention and FFN\"\"\"
    
    def __init__(self, d_model, n_heads, d_ff):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, n_heads)
        self.ffn = FeedForwardNetwork(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, x, mask=None):
        # Self-attention with residual connection and layer norm
        attn_output = self.attention(self.norm1(x), mask)
        x = x + self.dropout(attn_output)
        
        # Feed-forward with residual connection and layer norm
        ffn_output = self.ffn(self.norm2(x))
        x = x + self.dropout(ffn_output)
        
        return x

class LanguageModel(nn.Module):
    \"\"\"Complete transformer-based language model\"\"\"
    
    def __init__(self, vocab_size, d_model=512, n_heads=8, n_layers=6, 
                 d_ff=2048, max_seq_len=512):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        
        # Embedding layers
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_seq_len)
        
        # Transformer blocks
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff)
            for _ in range(n_layers)
        ])
        
        # Output layer
        self.layer_norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(self, input_ids):
        batch_size, seq_len = input_ids.shape
        
        # Create causal mask
        mask = torch.tril(torch.ones(seq_len, seq_len)).to(input_ids.device)
        
        # Token embeddings + positional encoding
        x = self.token_embedding(input_ids) * math.sqrt(self.d_model)
        x = self.pos_encoding(x)
        
        # Pass through transformer blocks
        for transformer_block in self.transformer_blocks:
            x = transformer_block(x, mask)
        
        # Final layer norm and projection to vocabulary
        x = self.layer_norm(x)
        logits = self.lm_head(x)
        
        return logits

# Test the language model
vocab_size = 10000
batch_size = 2
seq_len = 16

# Create model
model = LanguageModel(
    vocab_size=vocab_size,
    d_model=512,
    n_heads=8,
    n_layers=6,
    d_ff=2048
)

# Sample input tokens
input_tokens = torch.randint(0, vocab_size, (batch_size, seq_len))

print(f"Language Model Architecture:")
print(f"Vocabulary size: {vocab_size:,}")
print(f"Model dimension: {model.d_model}")
print(f"Number of layers: {len(model.transformer_blocks)}")
print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")

# Forward pass
with torch.no_grad():
    logits = model(input_tokens)
    print(f"\\nInput shape: {input_tokens.shape}")
    print(f"Output logits shape: {logits.shape}")
    
    # Convert to probabilities
    probs = F.softmax(logits, dim=-1)
    print(f"Output probabilities shape: {probs.shape}")

# Show architecture components
print(f"\\n🏗️ Architecture Components:")
print(f"• Token Embedding: Maps tokens to {model.d_model}D vectors")
print(f"• Positional Encoding: Adds position information")
print(f"• Transformer Blocks: {len(model.transformer_blocks)} layers of attention + FFN")
print(f"• Layer Normalization: Stabilizes training")
print(f"• Language Modeling Head: Projects to vocabulary size")

print("\\n✅ Complete LLM architecture implemented!")""",
            learning_objectives=[
                "Understand transformer architecture components",
                "Implement multi-head self-attention",
                "Build feed-forward networks",
                "Use positional encoding for sequence information",
                "Combine components into complete language model"
            ],
            hints=[
                "Attention allows tokens to interact with each other",
                "Positional encoding tells the model about token positions",
                "Residual connections help with gradient flow",
                "Layer normalization stabilizes training",
                "The language modeling head predicts next tokens"
            ]
        )
        
        # Session 8: Tokenization and BPE
        sessions["tokenization_bpe"] = Session(
            id="tokenization_bpe",
            title="🔤 Tokenization & BPE",
            description="""
# Tokenization and Byte Pair Encoding

Converting text to tokens for LLMs:
- Character vs word vs subword tokenization
- Byte Pair Encoding (BPE) algorithm
- Building vocabularies from data
- Encoding and decoding text
- SentencePiece implementation

Foundation for text processing in LLMs!
""",
            reference_code="""# Tokenization and Byte Pair Encoding (BPE)
import re
from collections import defaultdict, Counter

print("🔤 Tokenization and Byte Pair Encoding")

class SimpleBPETokenizer:
    \"\"\"Simple implementation of Byte Pair Encoding\"\"\"
    
    def __init__(self):
        self.word_freqs = {}
        self.vocab = set()
        self.merges = []
    
    def train(self, texts, vocab_size=1000):
        \"\"\"Train BPE on a corpus of texts\"\"\"
        print(f"Training BPE tokenizer with vocab size {vocab_size}")
        
        # Step 1: Get word frequencies
        self.word_freqs = self._get_word_frequencies(texts)
        print(f"Found {len(self.word_freqs)} unique words")
        
        # Step 2: Initialize vocabulary with characters
        self.vocab = self._get_initial_vocab(self.word_freqs)
        print(f"Initial vocabulary size: {len(self.vocab)}")
        
        # Step 3: Iteratively merge most frequent pairs
        while len(self.vocab) < vocab_size:
            pairs = self._get_stats(self.word_freqs)
            if not pairs:
                break
            
            best_pair = max(pairs, key=pairs.get)
            self._merge_vocab(best_pair, self.word_freqs)
            self.merges.append(best_pair)
            self.vocab.add(''.join(best_pair))
            
            if len(self.vocab) % 100 == 0:
                print(f"Vocabulary size: {len(self.vocab)}")
        
        print(f"Final vocabulary size: {len(self.vocab)}")
    
    def _get_word_frequencies(self, texts):
        \"\"\"Count word frequencies in texts\"\"\"
        word_freqs = Counter()
        for text in texts:
            words = text.lower().split()
            for word in words:
                # Add end-of-word symbol
                word_freqs[' '.join(word) + ' </w>'] += 1
        return dict(word_freqs)
    
    def _get_initial_vocab(self, word_freqs):
        \"\"\"Create initial vocabulary from characters\"\"\"
        vocab = set()
        for word in word_freqs.keys():
            for char in word.split():
                vocab.add(char)
        return vocab
    
    def _get_stats(self, word_freqs):
        \"\"\"Count frequency of adjacent symbol pairs\"\"\"
        pairs = defaultdict(int)
        for word, freq in word_freqs.items():
            symbols = word.split()
            for i in range(len(symbols) - 1):
                pairs[(symbols[i], symbols[i + 1])] += freq
        return pairs
    
    def _merge_vocab(self, pair, word_freqs):
        \"\"\"Merge the most frequent pair in vocabulary\"\"\"
        bigram = re.escape(' '.join(pair))
        p = re.compile(r'(?<!\\S)' + bigram + r'(?!\\S)')
        
        new_word_freqs = {}
        for word in word_freqs:
            new_word = p.sub(''.join(pair), word)
            new_word_freqs[new_word] = word_freqs[word]
        
        word_freqs.clear()
        word_freqs.update(new_word_freqs)
    
    def encode(self, text):
        \"\"\"Encode text using learned BPE\"\"\"
        words = text.lower().split()
        encoded_words = []
        
        for word in words:
            # Add end-of-word symbol
            word_tokens = list(word) + ['</w>']
            
            # Apply merges
            for pair in self.merges:
                word_str = ' '.join(word_tokens)
                if ' '.join(pair) in word_str:
                    word_str = word_str.replace(' '.join(pair), ''.join(pair))
                    word_tokens = word_str.split()
            
            encoded_words.extend(word_tokens)
        
        return encoded_words
    
    def decode(self, tokens):
        \"\"\"Decode tokens back to text\"\"\"
        text = ''.join(tokens)
        text = text.replace('</w>', ' ')
        return text.strip()

# Character-level tokenizer for comparison
class CharacterTokenizer:
    def __init__(self):
        self.char_to_id = {}
        self.id_to_char = {}
    
    def train(self, texts):
        # Get all unique characters
        chars = set()
        for text in texts:
            chars.update(text)
        
        # Create mappings
        chars = sorted(list(chars))
        self.char_to_id = {char: i for i, char in enumerate(chars)}
        self.id_to_char = {i: char for char, i in self.char_to_id.items()}
        
        print(f"Character vocabulary size: {len(self.char_to_id)}")
    
    def encode(self, text):
        return [self.char_to_id.get(char, 0) for char in text]
    
    def decode(self, ids):
        return ''.join([self.id_to_char.get(id, '') for id in ids])

# Word-level tokenizer for comparison
class WordTokenizer:
    def __init__(self):
        self.word_to_id = {}
        self.id_to_word = {}
    
    def train(self, texts):
        # Get all unique words
        words = set()
        for text in texts:
            words.update(text.lower().split())
        
        # Create mappings
        words = sorted(list(words))
        self.word_to_id = {word: i for i, word in enumerate(words)}
        self.id_to_word = {i: word for word, i in self.word_to_id.items()}
        
        print(f"Word vocabulary size: {len(self.word_to_id)}")
    
    def encode(self, text):
        words = text.lower().split()
        return [self.word_to_id.get(word, 0) for word in words]
    
    def decode(self, ids):
        return ' '.join([self.id_to_word.get(id, '<UNK>') for id in ids])

# Sample training data (Shakespeare-like)
training_texts = [
    "to be or not to be that is the question",
    "whether tis nobler in the mind to suffer",
    "the slings and arrows of outrageous fortune",
    "or to take arms against a sea of troubles",
    "and by opposing end them to die to sleep",
    "no more and by a sleep to say we end",
    "the heart ache and the thousand natural shocks",
    "that flesh is heir to tis a consummation"
]

print("Sample training texts:")
for i, text in enumerate(training_texts[:3]):
    print(f"{i+1}. {text}")

# Test different tokenization approaches
print(f"\\n📊 Tokenization Comparison:")

# Character-level
char_tokenizer = CharacterTokenizer()
char_tokenizer.train(training_texts)

# Word-level
word_tokenizer = WordTokenizer()
word_tokenizer.train(training_texts)

# BPE
bpe_tokenizer = SimpleBPETokenizer()
bpe_tokenizer.train(training_texts, vocab_size=100)

# Test encoding
test_text = "to be or not to be"
print(f"\\nTest text: '{test_text}'")

# Character encoding
char_encoded = char_tokenizer.encode(test_text)
char_decoded = char_tokenizer.decode(char_encoded)
print(f"Character: {char_encoded[:10]}... -> '{char_decoded}'")

# Word encoding
word_encoded = word_tokenizer.encode(test_text)
word_decoded = word_tokenizer.decode(word_encoded)
print(f"Word: {word_encoded} -> '{word_decoded}'")

# BPE encoding
bpe_encoded = bpe_tokenizer.encode(test_text)
bpe_decoded = bpe_tokenizer.decode(bpe_encoded)
print(f"BPE: {bpe_encoded} -> '{bpe_decoded}'")

# Show vocabulary samples
print(f"\\n📚 Vocabulary Samples:")
print(f"Character vocab: {sorted(list(char_tokenizer.char_to_id.keys()))[:20]}")
print(f"BPE vocab sample: {sorted(list(bpe_tokenizer.vocab))[:20]}")

# Demonstrate subword capabilities
rare_word = "shakespeare"
print(f"\\nRare word handling: '{rare_word}'")
print(f"BPE tokens: {bpe_tokenizer.encode(rare_word)}")

print("\\n✅ Tokenization and BPE complete!")
print("BPE creates subword units that balance vocabulary size and representation!")""",
            learning_objectives=[
                "Understand different tokenization approaches",
                "Implement Byte Pair Encoding algorithm",
                "Build vocabularies from training data",
                "Compare character, word, and subword tokenization",
                "Handle rare words with subword tokenization"
            ],
            hints=[
                "Character tokenization: small vocab, long sequences",
                "Word tokenization: large vocab, short sequences",
                "BPE: balanced approach with subword units",
                "BPE learns common subwords from data",
                "Subword tokenization handles out-of-vocabulary words"
            ]
        )
        
        # Continue with remaining sessions... (RoPE, RMS Norm, etc.)
        # Adding more sessions for completeness
        
        # Session 9: RoPE and Self-Attention
        sessions["rope_attention"] = Session(
            id="rope_attention",
            title="🎯 RoPE & Self-Attention",
            description="""
# Rotary Position Embedding and Self-Attention

Advanced attention mechanisms:
- Rotary Position Embedding (RoPE)
- Complex number rotations
- Position-aware attention
- Relative position encoding
- Implementation details

State-of-the-art position encoding for modern LLMs!
""",
            reference_code="""# Rotary Position Embedding (RoPE) and Self-Attention
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

print("🎯 Rotary Position Embedding and Self-Attention")

def precompute_freqs_cis(dim, seq_len, theta=10000.0):
    \"\"\"Precompute the frequency tensor for complex exponentials (cis)\"\"\"
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(seq_len, device=freqs.device)
    freqs = torch.outer(t, freqs).float()
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex exponentials
    return freqs_cis

def reshape_for_broadcast(freqs_cis, x):
    \"\"\"Reshape frequency tensor for broadcasting with attention weights\"\"\"
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)

def apply_rotary_emb(xq, xk, freqs_cis):
    \"\"\"Apply rotary embeddings to query and key tensors\"\"\"
    # Convert to complex numbers
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    
    # Reshape freqs_cis for broadcasting
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    
    # Apply rotation
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    
    return xq_out.type_as(xq), xk_out.type_as(xk)

class RoPEAttention(nn.Module):
    \"\"\"Multi-head attention with Rotary Position Embedding\"\"\"
    
    def __init__(self, d_model, n_heads, max_seq_len=512):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        
        self.wq = nn.Linear(d_model, d_model, bias=False)
        self.wk = nn.Linear(d_model, d_model, bias=False)
        self.wv = nn.Linear(d_model, d_model, bias=False)
        self.wo = nn.Linear(d_model, d_model, bias=False)
        
        # Precompute RoPE frequencies
        self.register_buffer(
            "freqs_cis",
            precompute_freqs_cis(self.head_dim, max_seq_len)
        )
    
    def forward(self, x, start_pos=0):
        batch_size, seq_len, _ = x.shape
        
        # Generate Q, K, V
        xq = self.wq(x)
        xk = self.wk(x)
        xv = self.wv(x)
        
        # Reshape for multi-head attention
        xq = xq.view(batch_size, seq_len, self.n_heads, self.head_dim)
        xk = xk.view(batch_size, seq_len, self.n_heads, self.head_dim)
        xv = xv.view(batch_size, seq_len, self.n_heads, self.head_dim)
        
        # Apply RoPE to queries and keys
        freqs_cis = self.freqs_cis[start_pos:start_pos + seq_len]
        xq, xk = apply_rotary_emb(xq, xk, freqs_cis)
        
        # Transpose for attention computation
        xq = xq.transpose(1, 2)  # (batch, n_heads, seq_len, head_dim)
        xk = xk.transpose(1, 2)
        xv = xv.transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(xq, xk.transpose(2, 3)) / math.sqrt(self.head_dim)
        
        # Causal mask
        mask = torch.tril(torch.ones(seq_len, seq_len)).to(x.device)
        scores = scores.masked_fill(mask == 0, float('-inf'))
        
        # Apply softmax
        attention_weights = F.softmax(scores, dim=-1)
        
        # Apply attention to values
        output = torch.matmul(attention_weights, xv)
        
        # Concatenate heads
        output = output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.d_model
        )
        
        return self.wo(output)

# Compare regular attention vs RoPE attention
class RegularAttention(nn.Module):
    \"\"\"Standard multi-head attention for comparison\"\"\"
    
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        
        self.wq = nn.Linear(d_model, d_model, bias=False)
        self.wk = nn.Linear(d_model, d_model, bias=False)
        self.wv = nn.Linear(d_model, d_model, bias=False)
        self.wo = nn.Linear(d_model, d_model, bias=False)
    
    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        
        # Generate Q, K, V
        q = self.wq(x).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.wk(x).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.wv(x).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        
        # Attention
        scores = torch.matmul(q, k.transpose(2, 3)) / math.sqrt(self.head_dim)
        
        # Causal mask
        mask = torch.tril(torch.ones(seq_len, seq_len)).to(x.device)
        scores = scores.masked_fill(mask == 0, float('-inf'))
        
        attention_weights = F.softmax(scores, dim=-1)
        output = torch.matmul(attention_weights, v)
        
        # Concatenate and project
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        return self.wo(output)

# Test both attention mechanisms
d_model = 256
n_heads = 8
seq_len = 16
batch_size = 2

# Create input
x = torch.randn(batch_size, seq_len, d_model)

# Create attention layers
regular_attn = RegularAttention(d_model, n_heads)
rope_attn = RoPEAttention(d_model, n_heads)

print(f"Input shape: {x.shape}")

# Forward pass through both
with torch.no_grad():
    regular_output = regular_attn(x)
    rope_output = rope_attn(x)

print(f"Regular attention output shape: {regular_output.shape}")
print(f"RoPE attention output shape: {rope_output.shape}")

# Demonstrate position sensitivity
print(f"\\n🔄 Position Sensitivity Test:")

# Create two sequences with swapped tokens
seq1 = torch.randn(1, 4, d_model)
seq2 = seq1.clone()
seq2[:, [0, 1]] = seq2[:, [1, 0]]  # Swap first two tokens

with torch.no_grad():
    # Regular attention (less position sensitive)
    out1_regular = regular_attn(seq1)
    out2_regular = regular_attn(seq2)
    regular_diff = torch.mean((out1_regular - out2_regular) ** 2)
    
    # RoPE attention (more position sensitive)
    out1_rope = rope_attn(seq1)
    out2_rope = rope_attn(seq2)
    rope_diff = torch.mean((out1_rope - out2_rope) ** 2)

print(f"Regular attention difference: {regular_diff:.6f}")
print(f"RoPE attention difference: {rope_diff:.6f}")
print(f"RoPE is more position-sensitive: {rope_diff > regular_diff}")

# Visualize rotation effect
print(f"\\n🌀 Rotation Demonstration:")
dummy_vec = torch.tensor([[1.0, 0.0]])  # Simple 2D vector
freqs = torch.tensor([math.pi / 4])     # 45 degree rotation
freqs_cis = torch.polar(torch.ones_like(freqs), freqs)

# Convert to complex and rotate
vec_complex = torch.view_as_complex(dummy_vec)
rotated_complex = vec_complex * freqs_cis
rotated_vec = torch.view_as_real(rotated_complex)

print(f"Original vector: {dummy_vec}")
print(f"After rotation: {rotated_vec}")
print(f"Rotation preserves magnitude: {torch.norm(dummy_vec):.3f} -> {torch.norm(rotated_vec):.3f}")

print("\\n✅ RoPE and Self-Attention complete!")
print("RoPE provides better position awareness for long sequences!")""",
            learning_objectives=[
                "Understand rotary position embedding",
                "Implement complex number rotations",
                "Compare RoPE vs standard position encoding",
                "Apply RoPE to multi-head attention",
                "Understand position sensitivity in attention"
            ],
            hints=[
                "RoPE rotates query and key vectors based on position",
                "Complex numbers enable efficient rotations",
                "RoPE preserves relative position information",
                "Better extrapolation to longer sequences",
                "Used in modern LLMs like LLaMA and PaLM"
            ]
        )
        
        # Session 10: RMS Normalization
        sessions["rms_norm"] = Session(
            id="rms_norm",
            title="⚖️ RMS Normalization",
            description="""
# Root Mean Square (RMS) Normalization

Modern normalization technique:
- RMS vs Layer Normalization
- Computational efficiency
- Gradient flow improvements
- Implementation details
- Integration with transformers

Used in state-of-the-art models like LLaMA!
""",
            reference_code="""# Root Mean Square (RMS) Normalization
import torch
import torch.nn as nn
import torch.nn.functional as F

print("⚖️ RMS Normalization")

class RMSNorm(nn.Module):
    \"\"\"Root Mean Square Layer Normalization\"\"\"
    
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    
    def _norm(self, x):
        # Compute RMS (root mean square)
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
    
    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight

class LayerNorm(nn.Module):
    \"\"\"Standard Layer Normalization for comparison\"\"\"
    
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim))
    
    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        var = x.var(-1, keepdim=True, unbiased=False)
        normalized = (x - mean) / torch.sqrt(var + self.eps)
        return normalized * self.weight + self.bias

# Comparison function
def compare_normalizations():
    print("🔍 Comparing RMS Norm vs Layer Norm")
    
    # Test data
    batch_size = 4
    seq_len = 8
    d_model = 256
    
    x = torch.randn(batch_size, seq_len, d_model) * 2  # Vary the scale
    
    # Create normalization layers
    rms_norm = RMSNorm(d_model)
    layer_norm = LayerNorm(d_model)
    
    # Apply normalizations
    x_rms = rms_norm(x)
    x_ln = layer_norm(x)
    
    print(f"Input statistics:")
    print(f"  Mean: {x.mean():.4f}, Std: {x.std():.4f}")
    print(f"  Min: {x.min():.4f}, Max: {x.max():.4f}")
    
    print(f"\\nRMS Norm output:")
    print(f"  Mean: {x_rms.mean():.4f}, Std: {x_rms.std():.4f}")
    print(f"  Min: {x_rms.min():.4f}, Max: {x_rms.max():.4f}")
    
    print(f"\\nLayer Norm output:")
    print(f"  Mean: {x_ln.mean():.4f}, Std: {x_ln.std():.4f}")
    print(f"  Min: {x_ln.min():.4f}, Max: {x_ln.max():.4f}")
    
    # Check variance along feature dimension
    rms_var = x_rms.var(-1).mean()
    ln_var = x_ln.var(-1).mean()
    
    print(f"\\nFeature dimension variance:")
    print(f"  RMS Norm: {rms_var:.4f}")
    print(f"  Layer Norm: {ln_var:.4f}")
    
    return x_rms, x_ln

# Efficiency comparison
def efficiency_comparison():
    print("\\n⚡ Efficiency Comparison")
    
    d_model = 512
    batch_size = 32
    seq_len = 128
    
    x = torch.randn(batch_size, seq_len, d_model)
    
    rms_norm = RMSNorm(d_model)
    layer_norm = LayerNorm(d_model)
    
    # Count operations
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters())
    
    rms_params = count_parameters(rms_norm)
    ln_params = count_parameters(layer_norm)
    
    print(f"Parameters:")
    print(f"  RMS Norm: {rms_params:,}")
    print(f"  Layer Norm: {ln_params:,}")
    print(f"  Reduction: {(ln_params - rms_params) / ln_params * 100:.1f}%")
    
    # Memory efficiency (no bias term in RMS)
    print(f"\\nMemory efficiency:")
    print(f"  RMS Norm: Only weight parameter")
    print(f"  Layer Norm: Weight + bias parameters")

# Gradient flow demonstration
def gradient_flow_test():
    print("\\n📈 Gradient Flow Test")
    
    d_model = 128
    x = torch.randn(2, 4, d_model, requires_grad=True)
    
    # Test with RMS Norm
    rms_norm = RMSNorm(d_model)
    output_rms = rms_norm(x)
    loss_rms = output_rms.sum()
    loss_rms.backward()
    grad_rms = x.grad.clone()
    
    # Reset gradients
    x.grad.zero_()
    
    # Test with Layer Norm
    layer_norm = LayerNorm(d_model)
    output_ln = layer_norm(x)
    loss_ln = output_ln.sum()
    loss_ln.backward()
    grad_ln = x.grad.clone()
    
    print(f"Gradient magnitudes:")
    print(f"  RMS Norm: {grad_rms.norm():.4f}")
    print(f"  Layer Norm: {grad_ln.norm():.4f}")
    
    # Check gradient distribution
    print(f"\\nGradient distribution:")
    print(f"  RMS Norm - Mean: {grad_rms.mean():.4f}, Std: {grad_rms.std():.4f}")
    print(f"  Layer Norm - Mean: {grad_ln.mean():.4f}, Std: {grad_ln.std():.4f}")

# Mathematical explanation
def mathematical_explanation():
    print("\\n📐 Mathematical Explanation")
    
    # Sample vector
    x = torch.tensor([1.0, 2.0, 3.0, 4.0])
    
    print(f"Input vector: {x}")
    
    # RMS calculation
    rms = torch.sqrt(torch.mean(x ** 2))
    x_rms_manual = x / rms
    
    print(f"\\nRMS Normalization (manual):")
    print(f"  RMS = sqrt(mean(x²)) = {rms:.4f}")
    print(f"  Normalized: {x_rms_manual}")
    
    # Layer norm calculation
    mean = torch.mean(x)
    var = torch.var(x, unbiased=False)
    x_ln_manual = (x - mean) / torch.sqrt(var)
    
    print(f"\\nLayer Normalization (manual):")
    print(f"  Mean = {mean:.4f}, Var = {var:.4f}")
    print(f"  Normalized: {x_ln_manual}")
    
    # Key difference
    print(f"\\n🎯 Key Differences:")
    print(f"  • RMS Norm: Only removes scale, preserves mean")
    print(f"  • Layer Norm: Removes both mean and scale")
    print(f"  • RMS Norm: Fewer parameters (no bias)")
    print(f"  • RMS Norm: Slightly more efficient computation")

# Integration with transformer block
class TransformerBlockWithRMS(nn.Module):
    \"\"\"Transformer block using RMS normalization\"\"\"
    
    def __init__(self, d_model, n_heads, d_ff):
        super().__init__()
        self.attention = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model)
        )
        
        # Use RMS Norm instead of Layer Norm
        self.norm1 = RMSNorm(d_model)
        self.norm2 = RMSNorm(d_model)
    
    def forward(self, x):
        # Pre-norm with RMS
        attn_out, _ = self.attention(self.norm1(x), self.norm1(x), self.norm1(x))
        x = x + attn_out
        
        ff_out = self.feed_forward(self.norm2(x))
        x = x + ff_out
        
        return x

# Run demonstrations
compare_normalizations()
efficiency_comparison()
gradient_flow_test()
mathematical_explanation()

# Test transformer block with RMS norm
print(f"\\n🏗️ Transformer Block with RMS Norm")
transformer_block = TransformerBlockWithRMS(d_model=256, n_heads=8, d_ff=1024)
test_input = torch.randn(2, 16, 256)
output = transformer_block(test_input)

print(f"Input shape: {test_input.shape}")
print(f"Output shape: {output.shape}")
print(f"Using RMS Norm in transformer blocks!")

print("\\n✅ RMS Normalization complete!")
print("RMS Norm: Simpler, more efficient normalization for modern LLMs!")""",
            learning_objectives=[
                "Understand RMS normalization vs Layer normalization",
                "Implement RMS normalization from scratch",
                "Compare computational efficiency",
                "Analyze gradient flow properties",
                "Integrate RMS norm into transformer blocks"
            ],
            hints=[
                "RMS norm only normalizes scale, not mean",
                "No bias parameter needed (more efficient)",
                "Used in modern models like LLaMA",
                "Simpler computation than layer norm",
                "Better numerical stability in some cases"
            ]
        )
        
        # Session 11: Feed-Forward Networks and Activations
        sessions["ffn_activations"] = Session(
            id="ffn_activations",
            title="🔄 FFN & Activation Functions",
            description="""
# Feed-Forward Networks and Activation Functions

Deep dive into neural network components:
- Feed-forward network architecture
- Activation functions (ReLU, GELU, SiLU/Swish)
- Why activations matter
- Position-wise feed-forward layers
- Gating mechanisms

Essential components of transformer blocks!
""",
            reference_code="""# Feed-Forward Networks and Activation Functions
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

print("🔄 Feed-Forward Networks and Activation Functions")

# Different activation functions
class ActivationFunctions:
    @staticmethod
    def relu(x):
        \"\"\"Rectified Linear Unit - simple and effective\"\"\"
        return torch.relu(x)
    
    @staticmethod
    def gelu(x):
        \"\"\"Gaussian Error Linear Unit - smoother than ReLU\"\"\"
        return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))
    
    @staticmethod
    def silu(x):
        \"\"\"SiLU/Swish - x * sigmoid(x)\"\"\"
        return x * torch.sigmoid(x)
    
    @staticmethod
    def gelu_approx(x):
        \"\"\"Approximate GELU for efficiency\"\"\"
        return 0.5 * x * (1 + torch.tanh(x * 0.7978845608 * (1 + 0.044715 * x * x)))

# Demonstrate different activations
def compare_activations():
    print("📊 Activation Function Comparison")
    
    # Test input range
    x = torch.linspace(-3, 3, 100)
    
    activations = {
        "ReLU": ActivationFunctions.relu(x),
        "GELU": ActivationFunctions.gelu(x), 
        "SiLU": ActivationFunctions.silu(x)
    }
    
    print(f"Input range: {x.min():.1f} to {x.max():.1f}")
    
    for name, output in activations.items():
        print(f"{name:6} - Range: {output.min():.3f} to {output.max():.3f}")
        print(f"       - Non-zero ratio: {(output != 0).float().mean():.3f}")

# Standard Feed-Forward Network (as used in transformers)
class FeedForwardNetwork(nn.Module):
    \"\"\"Position-wise feed-forward network\"\"\"
    
    def __init__(self, d_model, d_ff, activation='relu', dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        
        # Choose activation function
        if activation == 'relu':
            self.activation = F.relu
        elif activation == 'gelu':
            self.activation = F.gelu
        elif activation == 'silu':
            self.activation = F.silu
        else:
            self.activation = F.relu
        
        self.activation_name = activation
    
    def forward(self, x):
        # Linear -> Activation -> Dropout -> Linear
        x = self.linear1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x

# Gated Feed-Forward Network (used in some modern architectures)
class GatedFFN(nn.Module):
    \"\"\"Gated feed-forward network with SiLU activation\"\"\"
    
    def __init__(self, d_model, d_ff):
        super().__init__()
        # Two parallel linear layers for gating
        self.gate_proj = nn.Linear(d_model, d_ff, bias=False)
        self.up_proj = nn.Linear(d_model, d_ff, bias=False)
        self.down_proj = nn.Linear(d_ff, d_model, bias=False)
    
    def forward(self, x):
        # Gated mechanism: gate * up_projection
        gate = F.silu(self.gate_proj(x))  # Gating values
        up = self.up_proj(x)              # Up projection
        return self.down_proj(gate * up)  # Element-wise multiply and down project

# Compare different FFN architectures
def compare_ffn_architectures():
    print("\\n🏗️ FFN Architecture Comparison")
    
    d_model = 256
    d_ff = 1024
    batch_size = 4
    seq_len = 8
    
    # Input tensor
    x = torch.randn(batch_size, seq_len, d_model)
    
    # Different FFN types
    ffn_relu = FeedForwardNetwork(d_model, d_ff, activation='relu')
    ffn_gelu = FeedForwardNetwork(d_model, d_ff, activation='gelu')
    ffn_silu = FeedForwardNetwork(d_model, d_ff, activation='silu')
    ffn_gated = GatedFFN(d_model, d_ff)
    
    networks = [
        ("Standard FFN (ReLU)", ffn_relu),
        ("Standard FFN (GELU)", ffn_gelu), 
        ("Standard FFN (SiLU)", ffn_silu),
        ("Gated FFN", ffn_gated)
    ]
    
    print(f"Input shape: {x.shape}")
    
    for name, network in networks:
        with torch.no_grad():
            output = network(x)
            params = sum(p.numel() for p in network.parameters())
            print(f"{name:20} - Output: {output.shape}, Parameters: {params:,}")

# Demonstrate why activations matter
def why_activations_matter():
    print("\\n🤔 Why Activation Functions Matter")
    
    # Linear network without activation
    class LinearOnly(nn.Module):
        def __init__(self, input_size):
            super().__init__()
            self.layer1 = nn.Linear(input_size, 50)
            self.layer2 = nn.Linear(50, 25)
            self.layer3 = nn.Linear(25, 1)
        
        def forward(self, x):
            # No activation functions!
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            return x
    
    # Network with activations
    class WithActivations(nn.Module):
        def __init__(self, input_size):
            super().__init__()
            self.layer1 = nn.Linear(input_size, 50)
            self.layer2 = nn.Linear(50, 25)
            self.layer3 = nn.Linear(25, 1)
        
        def forward(self, x):
            # With ReLU activations
            x = F.relu(self.layer1(x))
            x = F.relu(self.layer2(x))
            x = self.layer3(x)
            return x
    
    input_size = 10
    x = torch.randn(5, input_size)
    
    linear_net = LinearOnly(input_size)
    activated_net = WithActivations(input_size)
    
    with torch.no_grad():
        linear_out = linear_net(x)
        activated_out = activated_net(x)
    
    print(f"Linear only output range: {linear_out.min():.3f} to {linear_out.max():.3f}")
    print(f"With activations range: {activated_out.min():.3f} to {activated_out.max():.3f}")
    print("Activations enable non-linear transformations!")

# Test activation properties
def test_activation_properties():
    print("\\n🔬 Activation Function Properties")
    
    # Test different properties
    test_input = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
    
    activations = {
        "ReLU": F.relu,
        "GELU": F.gelu,
        "SiLU": F.silu,
        "Tanh": torch.tanh,
        "Sigmoid": torch.sigmoid
    }
    
    print(f"Input: {test_input.tolist()}")
    print()
    
    for name, activation_fn in activations.items():
        output = activation_fn(test_input)
        
        # Check properties
        zero_centered = torch.mean(output).item()
        bounded = output.max().item() <= 10  # Roughly bounded
        monotonic = torch.all(output[1:] >= output[:-1]).item()
        
        print(f"{name:8} - Output: {[f'{x:.3f}' for x in output.tolist()]}")
        print(f"         - Zero-centered: {abs(zero_centered) < 0.1}")
        print(f"         - Roughly bounded: {bounded}")
        print()

# Run all demonstrations
compare_activations()
compare_ffn_architectures()
why_activations_matter()
test_activation_properties()

# Create a complete transformer-style FFN
print("🏆 Complete Transformer FFN")
transformer_ffn = FeedForwardNetwork(d_model=512, d_ff=2048, activation='gelu')
sample_input = torch.randn(2, 16, 512)

with torch.no_grad():
    ffn_output = transformer_ffn(sample_input)

print(f"Transformer FFN input: {sample_input.shape}")
print(f"Transformer FFN output: {ffn_output.shape}")
print(f"Parameters: {sum(p.numel() for p in transformer_ffn.parameters()):,}")

print("\\n✅ Feed-Forward Networks and Activations complete!")
print("FFNs provide the non-linear transformations that make deep learning powerful!")""",
            learning_objectives=[
                "Understand feed-forward network architecture",
                "Compare different activation functions",
                "Implement gated feed-forward networks", 
                "Learn why activations enable non-linearity",
                "Build transformer-style FFN blocks"
            ],
            hints=[
                "FFN = Linear → Activation → Linear layers",
                "ReLU is simple, GELU is smoother, SiLU is self-gated",
                "Activations enable non-linear transformations",
                "Gated FFNs use multiplicative interactions",
                "FFN is applied position-wise in transformers"
            ]
        )
        
        # Session 12: Training LLMs
        sessions["training_llm"] = Session(
            id="training_llm",
            title="🚂 Training Large Language Models",
            description="""
# Training Large Language Models

Complete training pipeline:
- Data preparation and tokenization
- Training loop implementation
- Loss computation and optimization
- Gradient accumulation and clipping
- Learning rate scheduling
- Monitoring and evaluation

Learn how to train your own LLM!
""",
            reference_code="""# Training Large Language Models - Complete Pipeline
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import math
import time

print("🚂 Training Large Language Models")

# Simple LLM for training demonstration
class SimpleLLM(nn.Module):
    def __init__(self, vocab_size, d_model=256, n_heads=8, n_layers=4, d_ff=1024):
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Embedding(512, d_model)  # Learnable positional embedding
        
        # Transformer blocks
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff)
            for _ in range(n_layers)
        ])
        
        self.layer_norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size)
    
    def forward(self, input_ids):
        batch_size, seq_len = input_ids.shape
        
        # Embeddings
        token_emb = self.embedding(input_ids)
        pos_ids = torch.arange(seq_len, device=input_ids.device).expand(batch_size, -1)
        pos_emb = self.pos_embedding(pos_ids)
        
        x = token_emb + pos_emb
        
        # Transformer layers
        for layer in self.layers:
            x = layer(x)
        
        # Output projection
        x = self.layer_norm(x)
        logits = self.lm_head(x)
        
        return logits

class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff):
        super().__init__()
        self.attention = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, x):
        # Self-attention with causal mask
        seq_len = x.size(1)
        causal_mask = torch.triu(torch.ones(seq_len, seq_len) * float('-inf'), diagonal=1)
        causal_mask = causal_mask.to(x.device)
        
        attn_out, _ = self.attention(x, x, x, attn_mask=causal_mask)
        x = self.norm1(x + self.dropout(attn_out))
        
        # Feed-forward
        ff_out = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_out))
        
        return x

# Dataset for language modeling
# Dataset for language modeling
class LanguageModelingDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length=128):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.examples = []
        
        # --- FIX START ---
        # Concatenate all texts into one long sequence for robust example creation.
        all_tokens = []
        for text in texts:
            # We add EOS token here to separate documents, but skip BOS
            all_tokens.extend(tokenizer.encode(text)[1:]) 

        # Create sliding window examples from the combined text
        # Use a stride of 1 to generate more examples.
        for i in range(0, len(all_tokens) - max_length, 1):
            chunk = all_tokens[i : i + max_length]
            # Ensure the chunk has the correct length before adding
            if len(chunk) == max_length:
                self.examples.append(chunk)

        if not self.examples:
            print(f"⚠️ WARNING: No training examples were created. "
                  f"The total number of tokens ({len(all_tokens)}) may be less than max_length ({max_length}).")
        # --- FIX END ---
    
    def __len__(self):
        # The number of samples should not be 0.
        return len(self.examples) if self.examples else 0
    
    def __getitem__(self, idx):
        tokens = self.examples[idx]
        # Input is tokens[:-1], target is tokens[1:]
        input_ids = torch.tensor(tokens[:-1], dtype=torch.long)
        labels = torch.tensor(tokens[1:], dtype=torch.long)
        return input_ids, labels

# Simple tokenizer for demonstration
class SimpleTokenizer:
    def __init__(self, vocab_size=5000):
        self.vocab_size = vocab_size
        self.word_to_id = {"<pad>": 0, "<unk>": 1, "<bos>": 2, "<eos>": 3}
        self.id_to_word = {v: k for k, v in self.word_to_id.items()}
        self.next_id = 4
    
    def build_vocab(self, texts):
        # Simple word-based vocabulary
        word_counts = {}
        for text in texts:
            words = text.lower().split()
            for word in words:
                word_counts[word] = word_counts.get(word, 0) + 1
        
        # Add most frequent words
        sorted_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
        for word, count in sorted_words[:self.vocab_size - 4]:
            if word not in self.word_to_id:
                self.word_to_id[word] = self.next_id
                self.id_to_word[self.next_id] = word
                self.next_id += 1
    
    def encode(self, text):
        words = text.lower().split()
        tokens = [2]  # <bos>
        for word in words:
            tokens.append(self.word_to_id.get(word, 1))  # <unk> if not found
        tokens.append(3)  # <eos>
        return tokens
    
    def decode(self, token_ids):
        words = [self.id_to_word.get(id, "<unk>") for id in token_ids]
        return " ".join(words).replace("<bos>", "").replace("<eos>", "").strip()

# Training function
def train_language_model():
    print("🏋️ Training Language Model")
    
    # Sample training data (Shakespeare-like)
    training_texts = [
        "to be or not to be that is the question",
        "whether tis nobler in the mind to suffer the slings and arrows",
        "of outrageous fortune or to take arms against a sea of troubles",
        "and by opposing end them to die to sleep no more",
        "and by a sleep to say we end the heartache and the thousand natural shocks",
        "that flesh is heir to tis a consummation devoutly to be wished",
        "to die to sleep to sleep perchance to dream ay there's the rub",
        "for in that sleep of death what dreams may come when we have shuffled"
    ] * 20  # Repeat for more training data
    
    # Initialize tokenizer and build vocabulary
    tokenizer = SimpleTokenizer(vocab_size=1000)
    tokenizer.build_vocab(training_texts)
    vocab_size = len(tokenizer.word_to_id)
    
    print(f"Vocabulary size: {vocab_size}")
    
    # Create dataset and dataloader
    dataset = LanguageModelingDataset(training_texts, tokenizer, max_length=32)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)
    
    print(f"Training examples: {len(dataset)}")
    
    # Initialize model
    model = SimpleLLM(vocab_size=vocab_size, d_model=128, n_heads=4, n_layers=3, d_ff=512)
    
    # Training setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)
    
    # Training loop
    model.train()
    epoch_losses = []
    
    print(f"\\nStarting training on {device}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    for epoch in range(5):  # Small number for demo
        epoch_loss = 0
        num_batches = 0
        
        for batch_idx, (input_ids, labels) in enumerate(dataloader):
            input_ids = input_ids.to(device)
            labels = labels.to(device)
            
            # Forward pass
            logits = model(input_ids)
            
            # Compute loss
            loss = F.cross_entropy(logits.view(-1, vocab_size), labels.view(-1))
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            # Update parameters
            optimizer.step()
            
            epoch_loss += loss.item()
            num_batches += 1
            
            # Log progress
            if batch_idx % 5 == 0:
                print(f"Epoch {epoch+1}, Batch {batch_idx+1}: Loss = {loss.item():.4f}")
        
        # Update learning rate
        scheduler.step()
        
        avg_loss = epoch_loss / num_batches
        epoch_losses.append(avg_loss)
        
        print(f"Epoch {epoch+1} completed. Average Loss: {avg_loss:.4f}, LR: {scheduler.get_last_lr()[0]:.6f}")
    
    return model, tokenizer, epoch_losses

# Text generation function
def generate_text(model, tokenizer, prompt="to be", max_length=20, temperature=1.0):
    \"\"\"Generate text using the trained model\"\"\"
    model.eval()
    device = next(model.parameters()).device
    
    # Encode prompt
    tokens = tokenizer.encode(prompt)
    input_ids = torch.tensor([tokens], dtype=torch.long, device=device)
    
    generated_tokens = tokens.copy()
    
    with torch.no_grad():
        for _ in range(max_length):
            # Get model predictions
            logits = model(input_ids)
            next_token_logits = logits[0, -1, :] / temperature
            
            # Sample next token
            probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1).item()
            
            # Add to sequence
            generated_tokens.append(next_token)
            
            # Update input (sliding window)
            new_token = torch.tensor([[next_token]], device=device)
            input_ids = torch.cat([input_ids, new_token], dim=1)
            
            # Stop if we hit end token
            if next_token == 3:  # <eos>
                break
    
    return tokenizer.decode(generated_tokens)

# Run training
print("Starting LLM training demonstration...")
trained_model, trained_tokenizer, losses = train_language_model()

# Test generation
print("\\n🎭 Text Generation Test")
prompts = ["to be", "whether tis", "and by"]

for prompt in prompts:
    generated = generate_text(trained_model, trained_tokenizer, prompt, max_length=15)
    print(f"Prompt: '{prompt}' → Generated: '{generated}'")

# Training analysis
print(f"\\n📊 Training Analysis")
print(f"Initial loss: {losses[0]:.4f}")
print(f"Final loss: {losses[-1]:.4f}")
print(f"Loss reduction: {((losses[0] - losses[-1]) / losses[0] * 100):.1f}%")

print("\\n✅ LLM Training Complete!")
print("This is the same process used to train GPT, BERT, and other language models!")""",
            learning_objectives=[
                "Build complete LLM training pipeline",
                "Implement language modeling dataset",
                "Set up training loop with proper optimization",
                "Apply gradient clipping and learning rate scheduling",
                "Generate text from trained models"
            ],
            hints=[
                "Language modeling predicts next token given previous tokens",
                "Use causal masking so model can't see future tokens",
                "Gradient clipping prevents exploding gradients",
                "Learning rate scheduling improves convergence",
                "Temperature controls randomness in text generation"
            ]
        )
        
        # Session 13: Inference and Text Generation
        sessions["inference_generation"] = Session(
            id="inference_generation", 
            title="🎯 Inference & Text Generation",
            description="""
# LLM Inference and Text Generation

Master text generation techniques:
- Greedy decoding vs sampling
- Temperature and top-k/top-p sampling
- Beam search for better quality
- Caching and optimization
- Batch inference
- Real-time generation

Turn your trained models into text generators!
""",
            reference_code="""# LLM Inference and Text Generation Techniques
import torch
import torch.nn.functional as F
import math
from typing import List, Optional

print("🎯 LLM Inference and Text Generation")

class TextGenerator:
    \"\"\"Advanced text generation with multiple decoding strategies\"\"\"
    
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.device = next(model.parameters()).device
        
        # Cache for efficient generation
        self.kv_cache = {}
    
    def greedy_generate(self, prompt: str, max_length: int = 50) -> str:
        \"\"\"Greedy decoding - always pick most likely token\"\"\"
        self.model.eval()
        
        tokens = self.tokenizer.encode(prompt)
        input_ids = torch.tensor([tokens], device=self.device)
        
        generated_tokens = tokens.copy()
        
        with torch.no_grad():
            for _ in range(max_length):
                logits = self.model(input_ids)
                next_token_logits = logits[0, -1, :]
                
                # Greedy: pick token with highest probability
                next_token = torch.argmax(next_token_logits).item()
                
                if next_token == self.tokenizer.word_to_id.get("<eos>", 3):
                    break
                
                generated_tokens.append(next_token)
                input_ids = torch.cat([input_ids, torch.tensor([[next_token]], device=self.device)], dim=1)
        
        return self.tokenizer.decode(generated_tokens)
    
    def temperature_sample(self, prompt: str, max_length: int = 50, temperature: float = 1.0) -> str:
        \"\"\"Temperature sampling - control randomness\"\"\"
        self.model.eval()
        
        tokens = self.tokenizer.encode(prompt)
        input_ids = torch.tensor([tokens], device=self.device)
        
        generated_tokens = tokens.copy()
        
        with torch.no_grad():
            for _ in range(max_length):
                logits = self.model(input_ids)
                next_token_logits = logits[0, -1, :] / temperature
                
                # Sample from probability distribution
                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1).item()
                
                if next_token == self.tokenizer.word_to_id.get("<eos>", 3):
                    break
                
                generated_tokens.append(next_token)
                input_ids = torch.cat([input_ids, torch.tensor([[next_token]], device=self.device)], dim=1)
        
        return self.tokenizer.decode(generated_tokens)
    
    def top_k_sample(self, prompt: str, max_length: int = 50, k: int = 10, temperature: float = 1.0) -> str:
        \"\"\"Top-k sampling - only consider k most likely tokens\"\"\"
        self.model.eval()
        
        tokens = self.tokenizer.encode(prompt)
        input_ids = torch.tensor([tokens], device=self.device)
        
        generated_tokens = tokens.copy()
        
        with torch.no_grad():
            for _ in range(max_length):
                logits = self.model(input_ids)
                next_token_logits = logits[0, -1, :] / temperature
                
                # Get top-k tokens
                top_k_logits, top_k_indices = torch.topk(next_token_logits, k)
                
                # Create probability distribution over top-k tokens
                top_k_probs = F.softmax(top_k_logits, dim=-1)
                
                # Sample from top-k
                sampled_index = torch.multinomial(top_k_probs, num_samples=1).item()
                next_token = top_k_indices[sampled_index].item()
                
                if next_token == self.tokenizer.word_to_id.get("<eos>", 3):
                    break
                
                generated_tokens.append(next_token)
                input_ids = torch.cat([input_ids, torch.tensor([[next_token]], device=self.device)], dim=1)
        
        return self.tokenizer.decode(generated_tokens)
    
    def top_p_sample(self, prompt: str, max_length: int = 50, p: float = 0.9, temperature: float = 1.0) -> str:
        \"\"\"Top-p (nucleus) sampling - dynamic vocabulary based on cumulative probability\"\"\"
        self.model.eval()
        
        tokens = self.tokenizer.encode(prompt)
        input_ids = torch.tensor([tokens], device=self.device)
        
        generated_tokens = tokens.copy()
        
        with torch.no_grad():
            for _ in range(max_length):
                logits = self.model(input_ids)
                next_token_logits = logits[0, -1, :] / temperature
                
                # Sort by probability
                sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                sorted_probs = F.softmax(sorted_logits, dim=-1)
                
                # Find cutoff point where cumulative probability exceeds p
                cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
                cutoff = torch.searchsorted(cumulative_probs, p).item() + 1
                
                # Keep only top-p tokens
                top_p_probs = sorted_probs[:cutoff]
                top_p_indices = sorted_indices[:cutoff]
                
                # Renormalize
                top_p_probs = top_p_probs / top_p_probs.sum()
                
                # Sample
                sampled_index = torch.multinomial(top_p_probs, num_samples=1).item()
                next_token = top_p_indices[sampled_index].item()
                
                if next_token == self.tokenizer.word_to_id.get("<eos>", 3):
                    break
                
                generated_tokens.append(next_token)
                input_ids = torch.cat([input_ids, torch.tensor([[next_token]], device=self.device)], dim=1)
        
        return self.tokenizer.decode(generated_tokens)
    
    def beam_search(self, prompt: str, max_length: int = 50, beam_size: int = 3) -> List[str]:
        self.model.eval()
        
        tokens = self.tokenizer.encode(prompt)
        
        # Initialize beams: (sequence, score)
        beams = [(tokens, 0.0)]
        
        with torch.no_grad():
            for _ in range(max_length):
                new_beams = []
                
                for sequence, score in beams:
                    if sequence[-1] == self.tokenizer.word_to_id.get("<eos>", 3):
                        new_beams.append((sequence, score))
                        continue
                    
                    # Get next token probabilities
                    input_ids = torch.tensor([sequence], device=self.device)
                    logits = self.model(input_ids)
                    next_token_logits = logits[0, -1, :]
                    log_probs = F.log_softmax(next_token_logits, dim=-1)
                    
                    # Get top beam_size candidates
                    top_log_probs, top_indices = torch.topk(log_probs, beam_size)
                    
                    for log_prob, token_id in zip(top_log_probs, top_indices):
                        new_sequence = sequence + [token_id.item()]
                        new_score = score + log_prob.item()
                        new_beams.append((new_sequence, new_score))
                
                # Keep top beam_size beams
                beams = sorted(new_beams, key=lambda x: x[1], reverse=True)[:beam_size]
                
      
                # Stop if the beam list becomes empty or if all beams have ended.
                # The 'if beams' check prevents 'all' on an empty list from being True.
                if not beams or all(seq[-1] == self.tokenizer.word_to_id.get("<eos>", 3) for seq, _ in beams):
                    break
        
        # Return decoded sequences, handling the case of an empty final beam
        return [self.tokenizer.decode(seq) for seq, _ in beams] if beams else []

# Demonstration of different generation methods
def demo_generation_methods():
    print("🎲 Generation Methods Demonstration")
    
    # Mock a simple model and tokenizer for demo
    class MockTokenizer:
        def __init__(self):
            self.word_to_id = {"<pad>": 0, "<unk>": 1, "<bos>": 2, "<eos>": 3, "the": 4, "quick": 5, "brown": 6, "fox": 7}
            self.id_to_word = {v: k for k, v in self.word_to_id.items()}
        
        def encode(self, text):
            return [2] + [self.word_to_id.get(word, 1) for word in text.split()] + [3]
        
        def decode(self, tokens):
            return " ".join([self.id_to_word.get(t, "<unk>") for t in tokens]).replace("<bos>", "").replace("<eos>", "").strip()
    
    class MockModel(torch.nn.Module):
        def __init__(self, vocab_size):
            super().__init__()
            self.vocab_size = vocab_size
        
        def forward(self, input_ids):
            batch_size, seq_len = input_ids.shape
            # Return random logits for demo
            return torch.randn(batch_size, seq_len, self.vocab_size)
    
    tokenizer = MockTokenizer()
    model = MockModel(len(tokenizer.word_to_id))
    generator = TextGenerator(model, tokenizer)
    
    prompt = "the quick"
    
    print(f"Prompt: '{prompt}'")
    print()
    
    # Demonstrate temperature effects
    print("🌡️ Temperature Effects:")
    for temp in [0.1, 0.7, 1.0, 1.5]:
        result = generator.temperature_sample(prompt, max_length=10, temperature=temp)
        print(f"Temperature {temp:3.1f}: {result}")
    
    print()
    
    # Demonstrate top-k effects
    print("🔝 Top-k Effects:")
    for k in [1, 3, 5, 10]:
        result = generator.top_k_sample(prompt, max_length=10, k=k)
        print(f"Top-k {k:2d}: {result}")
    
    print()
    
    # Demonstrate top-p effects  
    print("🎯 Top-p Effects:")
    for p in [0.5, 0.7, 0.9, 0.95]:
        result = generator.top_p_sample(prompt, max_length=10, p=p)
        print(f"Top-p {p:4.2f}: {result}")

# Generation quality metrics
def calculate_perplexity(model, tokenizer, text):
    \"\"\"Calculate perplexity of text under the model\"\"\"
    model.eval()
    tokens = tokenizer.encode(text)
    
    if len(tokens) < 2:
        return float('inf')
    
    input_ids = torch.tensor([tokens[:-1]], device=next(model.parameters()).device)
    targets = torch.tensor([tokens[1:]], device=next(model.parameters()).device)
    
    with torch.no_grad():
        logits = model(input_ids)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        perplexity = torch.exp(loss).item()
    
    return perplexity

def repetition_penalty(text):
    \"\"\"Calculate repetition in generated text\"\"\"
    words = text.split()
    if len(words) == 0:
        return 0
    
    unique_words = len(set(words))
    total_words = len(words)
    
    return 1 - (unique_words / total_words)

# Batch inference for efficiency
class BatchTextGenerator:
    \"\"\"Efficient batch text generation\"\"\"
    
    def __init__(self, model, tokenizer, batch_size=4):
        self.model = model
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.device = next(model.parameters()).device
    
    def generate_batch(self, prompts: List[str], max_length: int = 50) -> List[str]:
        \"\"\"Generate text for multiple prompts simultaneously\"\"\"
        self.model.eval()
        
        # Tokenize all prompts
        all_tokens = [self.tokenizer.encode(prompt) for prompt in prompts]
        
        # Pad to same length
        max_prompt_len = max(len(tokens) for tokens in all_tokens)
        padded_tokens = []
        
        for tokens in all_tokens:
            padded = tokens + [0] * (max_prompt_len - len(tokens))  # Pad with <pad>
            padded_tokens.append(padded)
        
        input_ids = torch.tensor(padded_tokens, device=self.device)
        generated_sequences = input_ids.clone()
        
        with torch.no_grad():
            for _ in range(max_length):
                logits = self.model(generated_sequences)
                next_token_logits = logits[:, -1, :]
                
                # Sample next tokens for all sequences
                probs = F.softmax(next_token_logits, dim=-1)
                next_tokens = torch.multinomial(probs, num_samples=1)
                
                # Append to sequences
                generated_sequences = torch.cat([generated_sequences, next_tokens], dim=1)
        
        # Decode all sequences
        results = []
        for sequence in generated_sequences:
            decoded = self.tokenizer.decode(sequence.tolist())
            results.append(decoded)
        
        return results

# Performance optimization tips
def optimization_tips():
    print("\\n⚡ Inference Optimization Tips")
    
    tips = [
        "1. Use model.eval() and torch.no_grad() for inference",
        "2. Implement KV-cache to avoid recomputing attention",
        "3. Use mixed precision (float16) for faster inference",
        "4. Batch multiple prompts together when possible",
        "5. Use beam search for quality, sampling for creativity",
        "6. Implement early stopping for efficiency",
        "7. Consider model quantization for deployment",
        "8. Use GPU when available, optimize for your hardware"
    ]
    
    for tip in tips:
        print(f"  {tip}")

# Run demonstrations
demo_generation_methods()
optimization_tips()

print("\\n🎯 Generation Strategies Summary:")
print("  • Greedy: Fast, deterministic, but can be repetitive")
print("  • Temperature: Control randomness (low=focused, high=creative)")  
print("  • Top-k: Limit vocabulary to k most likely tokens")
print("  • Top-p: Dynamic vocabulary based on probability mass")
print("  • Beam Search: Explore multiple paths for better quality")

print("\\n✅ Inference and Text Generation Complete!")
print("You now know how to generate text like ChatGPT and other LLMs!")""",
            learning_objectives=[
                "Implement different text generation strategies",
                "Understand trade-offs between quality and diversity", 
                "Apply temperature, top-k, and top-p sampling",
                "Use beam search for high-quality generation",
                "Optimize inference for speed and efficiency"
            ],
            hints=[
                "Greedy decoding is fast but can be repetitive",
                "Temperature controls randomness: low=focused, high=creative",
                "Top-k limits choices, top-p adapts vocabulary size",
                "Beam search explores multiple possibilities",
                "Batch processing improves efficiency"
            ]
        )
        
        return sessions
    
    def get_session(self, session_id: str) -> Optional[Session]:
        return self.sessions.get(session_id)
    
    def get_session_list(self) -> List[str]:
        return list(self.sessions.keys())
    
    def mark_session_complete(self, session_id: str):
        if session_id in self.sessions:
            self.sessions[session_id].completed = True
            if session_id not in self.progress.completed_sessions:
                self.progress.completed_sessions.append(session_id)
    
    def get_next_session(self) -> Optional[str]:
        session_order = self.get_session_list()
        try:
            current_index = session_order.index(self.progress.current_session_id)
            if current_index + 1 < len(session_order):
                return session_order[current_index + 1]
        except ValueError:
            pass
        return None

# Rest of the code remains the same as before but with session integration...
class ModelDownloader:
    def __init__(self, on_complete):
        self.on_complete = on_complete
        self.model_path = None
        self.model_url = "https://huggingface.co/Qwen/Qwen3-0.6B-GGUF/resolve/main/Qwen3-0.6B-Q8_0.gguf"
        self.filename = "Qwen3-0.6B-Q8_0.gguf"

    def run(self):
        self.setup_window = tk.Toplevel()
        self.setup_window.title("LLM Learning Dashboard Setup")
        self.setup_window.geometry("500x250")
        self.setup_window.resizable(False, False)
        self.setup_window.configure(bg='#1e1e1e')
        self.setup_window.transient()
        self.setup_window.grab_set()
        
        self.setup_window.update_idletasks()
        x = (self.setup_window.winfo_screenwidth() // 2) - (500 // 2)
        y = (self.setup_window.winfo_screenheight() // 2) - (250 // 2)
        self.setup_window.geometry(f'+{x}+{y}')

        tk.Label(self.setup_window, text="🧠 Initializing LLM Learning System", font=("Segoe UI", 16, "bold"), fg="#00aaff", bg='#1e1e1e').pack(pady=(20, 10))
        self.status_var = tk.StringVar(value="Checking for AI mentor model...")
        tk.Label(self.setup_window, textvariable=self.status_var, font=("Segoe UI", 11), fg="#cccccc", bg='#1e1e1e').pack(pady=5)
        
        s = ttk.Style()
        s.configure("Blue.Horizontal.TProgressbar", foreground='#007bff', background='#007bff')
        self.progress_bar = ttk.Progressbar(self.setup_window, style="Blue.Horizontal.TProgressbar", length=400, mode='determinate')
        self.progress_bar.pack(pady=20)
        
        self.setup_window.after(500, self.start_download)

    def start_download(self):
        threading.Thread(target=self._download_worker, daemon=True).start()

    def _download_worker(self):
        if not LLAMA_CPP_AVAILABLE:
            messagebox.showerror("Dependency Error", "Please install llama-cpp-python first:\npip install llama-cpp-python")
            self.setup_window.after(0, self._finalize, None)
            return

        model_dir = Path("models")
        model_dir.mkdir(exist_ok=True)
        self.model_path = model_dir / self.filename

        if self.model_path.exists() and self.model_path.stat().st_size > 100_000_000:
            self.status_var.set("AI Mentor found. Starting dashboard...")
            self.progress_bar['value'] = 100
            time.sleep(1.5)
            self.setup_window.after(0, self._finalize, str(self.model_path))
            return

        self.status_var.set(f"Downloading {self.filename}...")
        try:
            with urllib.request.urlopen(self.model_url) as response, open(self.model_path, 'wb') as out_file:
                total_size = int(response.info().get('Content-Length', 300_000_000))
                downloaded = 0
                while True:
                    buffer = response.read(8192)
                    if not buffer: break
                    out_file.write(buffer)
                    downloaded += len(buffer)
                    progress = (downloaded / total_size) * 100
                    self.progress_bar['value'] = progress
                    self.status_var.set(f"Downloading... {downloaded / 1_048_576:.1f} / {total_size / 1_048_576:.1f} MB")
            
            self.status_var.set("Download complete! Starting dashboard...")
            time.sleep(1.5)
            self.setup_window.after(0, self._finalize, str(self.model_path))
        except Exception as e:
            messagebox.showerror("Download Failed", f"Could not download the AI model.\nPlease check your internet connection.\nError: {e}")
            self.setup_window.after(0, self._finalize, None)

    def _finalize(self, model_path):
        self.setup_window.destroy()
        self.on_complete(model_path)

class LLMAIFeedbackSystem:
    def __init__(self, model_path: Optional[str]):
        self.model = None
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=2)
        
        if not model_path: 
            print("❌ No model path provided. AI features will be disabled.")
        elif not LLAMA_CPP_AVAILABLE: 
            print("❌ Llama CPP library not found. AI features will be disabled.")
        else:
            try:
                print(f"🤖 Loading LLM Training Mentor AI from: {model_path}")
                self.model = Llama(model_path=model_path, chat_format="chatml", n_ctx=4096, n_gpu_layers=-1, verbose=False)
                print("✅ LLM Training Mentor AI Initialized.")
            except Exception as e:
                print(f"❌ Failed to load local AI model: {e}")
                messagebox.showerror("AI Model Error", f"Could not load the AI model from the specified path.\nError: {e}")
    
    @property
    def is_available(self) -> bool: 
        return self.model is not None

    def generate_feedback(self, code: str, error: str, session_id: str) -> str:
        if not self.is_available: 
            return "AI mentor is currently unavailable."
        
        system_prompt = """You are Sandra, an expert LLM training mentor. You guide students through learning to build language models from scratch. 

Provide brief, encouraging feedback (1-2 sentences) focused on the current learning session. When code runs successfully, celebrate and suggest next steps. When code fails, give clear, specific hints to fix the error.

Don't tell about the next session or task, or next, just only focus on the current one, nothing more, nothing less./no_think"""

        if not error:
            user_prompt = f"Session: {session_id}. Student's code ran successfully! Give brief positive feedback and encourage them toward the next concept."
        else:
            user_prompt = f"Session: {session_id}. Student's code failed with: '{error}'. Give a concise hint to fix it, related to the session topic."
        
        try:
            response = self.model.create_chat_completion(
                messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
                temperature=0.1, max_tokens=1024
            )
            feedback = response['choices'][0]['message']['content'].strip()
            import re
            feedback = re.sub(r'<think>.*?</think>', '', feedback, flags=re.DOTALL).strip()
            
            return feedback if feedback else ("Excellent! Keep exploring!" if not error else "Check the error carefully and try again.")
        except Exception as e:
            print(f"❌ AI feedback generation error: {e}")
            return "Could not generate AI feedback at this time."

    def initial_session_message(self, session_id: str) -> str:
        """Generate initial message when starting a new session"""
        if not self.is_available:
            return "Welcome to the new session! Try typing the reference code to learn."
        
        system_prompt = """You are Sandra, an LLM training mentor. A student just started a new learning session. Give them a warm welcome and encourage them to manually type the reference code they see on the left side to practice. Be brief (1-2 sentences) and encouraging. Don't tell about next sessions"""
        
        user_prompt = f"Student just started session: {session_id}. Welcome them and encourage manual typing practice."
        
        try:
            response = self.model.create_chat_completion(
                messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
                temperature=0.7, max_tokens=1024
            )
            return response['choices'][0]['message']['content'].strip()
        except Exception as e:
            return "Welcome! Start by typing the reference code on the left to practice. Sandra is here to help!"

class EnhancedKokoroTTSSystem:
    def __init__(self):
        self.pipeline = None
        self.stop_event = threading.Event()
        self.is_speaking = False

        if KOKORO_TTS_AVAILABLE:
            try:
                self.pipeline = KPipeline(repo_id='hexgrad/Kokoro-82M', lang_code='a')
                print("✅ Enhanced Kokoro TTS System Initialized.")
            except Exception as e:
                print(f"❌ Failed to initialize Kokoro TTS: {e}")

    @property
    def is_available(self) -> bool: 
        return self.pipeline is not None

    def speak(self, text: str):
        if not self.is_available or not text or self.is_speaking: 
            return
        self.stop_event.clear()
        self.is_speaking = True
        threading.Thread(target=self._audio_worker, args=(text,), daemon=True).start()

    def stop_speech(self):
        if self.is_speaking: 
            self.stop_event.set()

    def _audio_worker(self, text: str):
        try:
            processed_text = self._preprocess_text_for_tts(text)
            audio_chunks = []
            
            for _, _, audio in self.pipeline(processed_text, voice='af_heart'):
                if self.stop_event.is_set():
                    print("🔊 Audio stopped by user.")
                    return
                audio_chunks.append(audio)
            
            if audio_chunks:
                full_audio = torch.cat(audio_chunks)
                sd.play(full_audio, samplerate=24000)
                while sd.get_stream().active:
                    if self.stop_event.is_set():
                        sd.stop()
                        print("🔊 Audio stream stopped.")
                        break
                    time.sleep(0.1)
        except Exception as e: 
            print(f"❌ Kokoro TTS error: {e}")
        finally:
            self.is_speaking = False
            self.stop_event.clear()

    def _preprocess_text_for_tts(self, text: str) -> str:
        """Preprocess text to improve TTS quality for technical content"""
        replacements = {
            "PyTorch": "pie torch",
            "LLM": "large language model",
            "RMSNorm": "R M S normalization",
            "RoPE": "rotary position encoding",
            "AdamW": "Adam W optimizer",
            "BPE": "byte pair encoding",
            "GPU": "graphics processing unit",
            "CPU": "central processing unit"
        }
        
        processed = text
        for term, replacement in replacements.items():
            processed = processed.replace(term, replacement)
        
        return processed

class ModernCodeEditor(tk.Frame):
    def __init__(self, parent, *args, **kwargs):
        super().__init__(parent, *args, **kwargs)
        self.configure(bg='#1e1e1e')
        
        # Line numbers
        self.line_numbers = tk.Text(self, width=4, padx=4, takefocus=0, bd=0, bg='#1e1e1e', fg='#6c757d',
                                    font=('Consolas', 12), state='disabled')
        self.line_numbers.pack(side='left', fill='y')
        
        # Main text editor
        self.text_widget = scrolledtext.ScrolledText(
            self, wrap=tk.WORD, font=('Consolas', 12), bg='#282c34', fg='#abb2bf',
            insertbackground='white', selectbackground='#3e4451', bd=0, undo=True, maxundo=20)
        self.text_widget.pack(side='right', fill='both', expand=True)
        
        # Bind events
        self.text_widget.bind('<KeyRelease>', self._on_change)
        self.text_widget.bind('<MouseWheel>', self._on_change)
        self.text_widget.bind('<Return>', self._on_return)
        self.text_widget.bind('<Tab>', self._on_tab)
        
        self._on_change()

    def _on_change(self, event=None): 
        self._update_line_numbers()
        
    def _on_tab(self, event=None):
        """Handle tab indentation"""
        self.text_widget.insert(tk.INSERT, '    ')  # 4 spaces
        return 'break'
        
    def _on_return(self, event=None):
        """Handle auto-indentation"""
        self.text_widget.insert(tk.INSERT, '\n')
        current_line_number_str = self.text_widget.index(tk.INSERT).split('.')[0]
        try:
            current_line_number = int(current_line_number_str)
            if current_line_number > 1:
                previous_line = self.text_widget.get(f'{current_line_number-1}.0', f'{current_line_number-1}.end')
                indent_match = re.match(r'^(\s*)', previous_line)
                indent = indent_match.group(0) if indent_match else ""
                if previous_line.strip().endswith(':'): 
                    indent += '    '
                self.text_widget.insert(tk.INSERT, indent)
        except (ValueError, tk.TclError): 
            pass 
        self._update_line_numbers()
        return 'break'
                
    def _update_line_numbers(self):
        self.line_numbers.config(state='normal')
        self.line_numbers.delete('1.0', 'end')
        line_count_str = self.text_widget.index('end-1c').split('.')[0]
        try:
            line_count = int(line_count_str)
            line_number_string = "\n".join(str(i) for i in range(1, line_count + 1))
            self.line_numbers.insert('1.0', line_number_string)
        except ValueError: 
            pass
        self.line_numbers.config(state='disabled')
        self.line_numbers.yview_moveto(self.text_widget.yview()[0])
        
    def get_text(self): 
        return self.text_widget.get('1.0', 'end-1c')
        
    def set_text(self, text):
        self.text_widget.delete('1.0', 'end')
        self.text_widget.insert('1.0', text)
        self._on_change()
        
    def clear(self): 
        self.set_text('')

class SessionBasedLLMLearningDashboard:
    def __init__(self, root: tk.Tk, model_path: Optional[str]):
        self.root = root
        self.session_manager = SessionManager()
        self.ai_system = LLMAIFeedbackSystem(model_path)
        self.tts_system = EnhancedKokoroTTSSystem()
        
        # Animation and loading states
        self.is_loading = False
        self.panel_animation_chars = ['|', '/', '-', '\\']
        self.panel_animation_index = 0
        self.button_animation_chars = [' .', ' ..', ' ...']
        self.button_animation_index = 0
        
        # Async executor for better performance
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=3)
        
        self._configure_styles()
        self._setup_window()
        self._create_widgets()
        self._load_current_session()
        
        # Welcome message
        self._show_initial_session_message()
        
        if not self.ai_system.is_available:
            self.status_label.config(text="🚨 AI Mentor Offline. Check console for errors.")

    def _setup_window(self):
        self.root.title("🧠 LLM Training Academy - Session-Based Learning")
        
        # Get screen dimensions
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        
        # Set window size as percentage of screen (not larger than screen)
        window_width = min(1400, int(screen_width * 0.9))
        window_height = min(900, int(screen_height * 0.9))
        
        self.root.geometry(f"{window_width}x{window_height}")
        self.root.configure(bg='#1e1e1e')
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        
        # Make window resizable
        self.root.resizable(True, True)
        self.root.minsize(1200, 700)  # Minimum size
        
        # Center the window
        self.root.update_idletasks()
        x = (screen_width // 2) - (window_width // 2)
        y = (screen_height // 2) - (window_height // 2)
        self.root.geometry(f'{window_width}x{window_height}+{x}+{y}')

    def _configure_styles(self):
        style = ttk.Style()
        style.theme_use('clam')
        
        # Button styles
        style.configure('TButton', font=('Segoe UI', 10, 'bold'), padding=8, relief='flat', background='#3a3d41', foreground='white')
        style.map('TButton', background=[('active', '#4a4d51')])
        style.configure('Run.TButton', background='#28a745', foreground='white')
        style.map('Run.TButton', background=[('active', '#218838')])
        style.configure('Clear.TButton', background='#dc3545', foreground='white')
        style.map('Clear.TButton', background=[('active', '#c82333')])
        style.configure('Session.TButton', background='#6f42c1', foreground='white')
        style.map('Session.TButton', background=[('active', '#5a32a3')])
        style.configure('Next.TButton', background='#007bff', foreground='white')
        style.map('Next.TButton', background=[('active', '#0056b3')])
        
        # Other styles
        style.configure('TLabel', background='#1e1e1e', foreground='#f8f9fa', font=('Segoe UI', 10))
        style.configure('Header.TLabel', font=('Segoe UI', 16, 'bold'), foreground='#00aaff')
        style.configure('Session.TLabel', font=('Segoe UI', 12, 'bold'), foreground='#ffc107')
        style.configure('Progress.TLabel', font=('Segoe UI', 11), foreground='#28a745')
        style.configure('TFrame', background='#1e1e1e')
        style.configure('Left.TFrame', background='#252526')
        style.configure('Right.TFrame', background='#1e1e1e')
        style.configure('TCheckbutton', background='#252526', foreground='white', font=('Segoe UI', 10))
        style.map('TCheckbutton', background=[('active', '#252526')], foreground=[('active', 'white')])
        style.configure('TPanedwindow', background='#1e1e1e')
        style.configure('TPanedwindow.Sash', sashthickness=6, relief='flat', background='#3a3d41')

    def _create_widgets(self):
        # Main paned window
        main_pane = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        main_pane.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Left panel (session content and reference code) - fixed width but responsive
        left_frame = ttk.Frame(main_pane, style='Left.TFrame')
        self._create_left_panel(left_frame)
        main_pane.add(left_frame, weight=1)
        
        # Right panel (IDE and output)
        right_pane = ttk.PanedWindow(main_pane, orient=tk.VERTICAL)
        self._create_right_panel(right_pane)
        main_pane.add(right_pane, weight=2)
        
        # Configure pane sizes after a brief delay to ensure proper sizing
        self.root.after(100, lambda: self._configure_pane_sizes(main_pane))
        
        # Status bar with progress
        status_frame = ttk.Frame(self.root, style='Left.TFrame', height=35)
        status_frame.pack(side='bottom', fill='x', padx=5, pady=(0, 5))
        status_frame.pack_propagate(False)
        
        self.status_label = ttk.Label(status_frame, text="🧠 LLM Training Mentor Ready.", background='#252526', anchor='w', style='TLabel')
        self.status_label.pack(side='left', padx=10, pady=5)
        
        # Progress indicator
        completion_pct = self.session_manager.progress.get_completion_percentage()
        self.progress_label = ttk.Label(status_frame, text=f"Progress: {completion_pct:.0f}% Complete", 
                                       background='#252526', anchor='e', style='Progress.TLabel')
        self.progress_label.pack(side='right', padx=10, pady=5)

    def _configure_pane_sizes(self, main_pane):
        """Configure pane sizes after window is displayed"""
        try:
            total_width = self.root.winfo_width()
            if total_width > 100:  # Make sure window is actually displayed
                left_width = min(500, int(total_width * 0.35))  # 35% or max 500px
                main_pane.sashpos(0, left_width)
        except tk.TclError:
            pass  # Ignore if window is not ready

    def _create_left_panel(self, parent):
        parent.configure(style='Left.TFrame')
        
        # Header with session info
        header_frame = ttk.Frame(parent, style='Left.TFrame')
        header_frame.pack(fill='x', padx=8, pady=8)
        
        ttk.Label(header_frame, text="🎯 Learning Sessions", style='Header.TLabel', background='#252526').pack(anchor='w')
        
        # Current session info
        current_session = self.session_manager.get_session(self.session_manager.progress.current_session_id)
        session_title = current_session.title if current_session else "No Session"
        self.current_session_label = ttk.Label(header_frame, text=f"Current: {session_title}", 
                                              style='Session.TLabel', background='#252526')
        self.current_session_label.pack(anchor='w', pady=(3, 0))
        
        # Session navigation
        nav_frame = ttk.Frame(parent, style='Left.TFrame')
        nav_frame.pack(fill='x', padx=8, pady=3)
        
        # Session selection dropdown
        session_names = [
            "🐍 Python Fundamentals",
            "🔢 PyTorch & NumPy", 
            "🧠 Neural Networks",
            "⬅️ Backpropagation",
            "🛡️ Regularization",
            "📉 Loss & Optimizers",
            "🏗️ LLM Architecture", 
            "🔤 Tokenization & BPE",
            "🎯 RoPE & Attention",
            "⚖️ RMS Normalization",
            "🔄 FFN & Activations",
            "🚂 Training LLMs",

        ]
        
        self.session_var = tk.StringVar(value=session_names[0])
        session_dropdown = ttk.OptionMenu(nav_frame, self.session_var, session_names[0], *session_names, command=self._on_session_change)
        session_dropdown.pack(side='left', fill='x', expand=True, padx=(0, 3))
        
        self.next_session_button = ttk.Button(nav_frame, text="Next →", command=self._next_session, style='Next.TButton')
        self.next_session_button.pack(side='right', padx=(3, 0))
        
        # Session description
        desc_frame = ttk.Frame(parent, style='Left.TFrame')
        desc_frame.pack(fill='both', expand=True, padx=8, pady=3)
        
        ttk.Label(desc_frame, text="📖 Reference Code", style='TLabel', background='#252526').pack(anchor='w')
        
        # Reference code display (read-only) - reduced height to prevent overflow
        self.reference_text = scrolledtext.ScrolledText(
            desc_frame, wrap=tk.WORD, font=('Consolas', 10), bg='#2d3748', fg='#e2e8f0',
            bd=1, relief='solid', padx=8, pady=8, state='disabled',
            height=20  # Reduced from 25 to 20
        )
        self.reference_text.pack(fill='both', expand=True, pady=3)
        
        # Instruction label
        instruction_frame = ttk.Frame(parent, style='Left.TFrame')
        instruction_frame.pack(fill='x', padx=8, pady=3)
        
        self.instruction_label = ttk.Label(instruction_frame, 
                                          text="👆 Type the code manually in the right editor to practice!",
                                          style='TLabel', background='#252526', foreground='#ffc107',
                                          font=('Segoe UI', 9))
        self.instruction_label.pack(anchor='w')
        
        # Action buttons
        action_frame = ttk.Frame(parent, style='Left.TFrame')
        action_frame.pack(fill='x', padx=8, pady=5)
        
        ttk.Button(action_frame, text="💡 Hint", command=self._get_hint, style='Session.TButton').pack(side='left', fill='x', expand=True, padx=(0, 3))
        ttk.Button(action_frame, text="📋 Copy", command=self._copy_reference, style='TButton').pack(side='left', fill='x', expand=True, padx=3)
        
        # Audio controls
        audio_frame = ttk.Frame(parent, style='Left.TFrame')
        audio_frame.pack(fill='x', padx=8, pady=3)
        
        self.audio_var = tk.BooleanVar(value=True)
        audio_check = ttk.Checkbutton(audio_frame, text="🔊 Sandra's Voice", variable=self.audio_var, style='TCheckbutton')
        audio_check.pack(side='left', padx=3)
        
        self.stop_audio_button = ttk.Button(audio_frame, text="⏹️ Stop", command=self.tts_system.stop_speech, state=tk.DISABLED)
        self.stop_audio_button.pack(side='right', padx=3)

    def _create_right_panel(self, parent):
        # IDE section
        ide_frame = ttk.Frame(parent, height=600)
        self._create_ide_section(ide_frame)
        parent.add(ide_frame, weight=3)
        
        # Output section
        output_frame = ttk.Frame(parent, height=400)
        self._create_output_section(output_frame)
        parent.add(output_frame, weight=2)

    def _create_ide_section(self, parent):
        parent.pack_propagate(False)
        
        # Header with typing instructions
        header_frame = ttk.Frame(parent)
        header_frame.pack(fill='x', pady=(10, 5), padx=10)
        
        ttk.Label(header_frame, text="💻 Code Practice Area", style='Header.TLabel').pack(side='left')
        ttk.Label(header_frame, text="Type the reference code manually - no copy/paste shortcuts!", 
                 style='TLabel', foreground='#ffc107').pack(side='right')
        
        # Code editor
        self.code_editor = ModernCodeEditor(parent)
        
        # Button toolbar
        button_bar = ttk.Frame(parent)
        button_bar.pack(fill='x', pady=5, padx=10)
        
        # Run button
        self.run_button = ttk.Button(button_bar, text="▶ Run Code", command=self._run_code, style='Run.TButton')
        self.run_button.pack(side='left', padx=2)
        
        # Clear button
        ttk.Button(button_bar, text="🧹 Clear", command=self.code_editor.clear, style='Clear.TButton').pack(side='left', padx=2)
        
        # Code help buttons
        ttk.Button(button_bar, text="↶ Undo", command=lambda: self.code_editor.text_widget.edit_undo()).pack(side='left', padx=2)
        ttk.Button(button_bar, text="↷ Redo", command=lambda: self.code_editor.text_widget.edit_redo()).pack(side='left', padx=2)
        
        # File operations
        ttk.Button(button_bar, text="💾 Save", command=self._save_code).pack(side='right', padx=2)
        ttk.Button(button_bar, text="📁 Load", command=self._load_code).pack(side='right', padx=2)
        
        self.code_editor.pack(fill='both', expand=True, pady=5, padx=10)

    def _create_output_section(self, parent):
        parent.pack_propagate(False)
        self.output_frame = parent
        
        ttk.Label(self.output_frame, text="📤 Output & AI Feedback from Sandra", style='Header.TLabel').pack(pady=10)
        
        # Loading animation label
        self.loading_label = ttk.Label(self.output_frame, text="", font=('Consolas', 14, 'bold'), 
                                      foreground="#00aaff", background='#1e1e1e')
        
        # Output text widget
        self.output_text = scrolledtext.ScrolledText(
            self.output_frame, wrap=tk.WORD, font=('Consolas', 11), bg='#282c34', fg='#f8f9fa', 
            bd=0, relief='flat', state='disabled', padx=10, pady=10
        )
        self.output_text.pack(fill='both', expand=True, pady=5, padx=10)
        
        # Configure text tags for different types of output
        self.output_text.tag_config('success', foreground='#28a745', font=('Consolas', 11, 'bold'))
        self.output_text.tag_config('error', foreground='#dc3545', font=('Consolas', 11, 'bold'))
        self.output_text.tag_config('ai_feedback', foreground='#00aaff', font=('Consolas', 12, 'italic'))
        self.output_text.tag_config('hint', foreground='#ffc107', font=('Consolas', 11, 'bold'))
        self.output_text.tag_config('info', foreground='#17a2b8', font=('Consolas', 11))
        self.output_text.tag_config('session_msg', foreground='#6f42c1', font=('Consolas', 12, 'bold'))

    def _load_current_session(self):
        """Load the current session content"""
        current_session = self.session_manager.get_session(self.session_manager.progress.current_session_id)
        if not current_session:
            return
        
        # Update session display
        self.current_session_label.config(text=f"Current: {current_session.title}")
        
        # Load reference code
        self.reference_text.config(state='normal')
        self.reference_text.delete('1.0', 'end')
        
        # Insert description and reference code
        full_content = current_session.description + "\n\n" + "="*60 + "\n" + "REFERENCE CODE TO TYPE:\n" + "="*60 + "\n\n" + current_session.reference_code
        self.reference_text.insert('1.0', full_content)
        self.reference_text.config(state='disabled')
        
        # Clear the editor - student must type manually
        self.code_editor.clear()
        
        # Update session dropdown
        session_mapping = {
            "python_fundamentals": "🐍 Python Fundamentals",
            "pytorch_numpy": "🔢 PyTorch & NumPy",
            "neural_networks": "🧠 Neural Networks", 
            "backpropagation": "⬅️ Backpropagation",
            "regularization": "🛡️ Regularization",
            "loss_optimizers": "📉 Loss & Optimizers",
            "llm_architecture": "🏗️ LLM Architecture",
            "tokenization_bpe": "🔤 Tokenization & BPE",
            "rope_attention": "🎯 RoPE & Attention",
            "rms_norm": "⚖️ RMS Normalization",
            "ffn_activations": "🔄 FFN & Activations",
            "training_llm": "🚂 Training LLMs",
            "inference_generation": "🎯 Inference & Generation"
        }
        
        display_name = session_mapping.get(current_session.id, current_session.title)
        self.session_var.set(display_name)
        
        # Update progress
        self._update_progress_display()

    def _update_progress_display(self):
        """Update the progress display"""
        completion_pct = self.session_manager.progress.get_completion_percentage()
        completed_count = len(self.session_manager.progress.completed_sessions)
        total_count = self.session_manager.progress.total_sessions
        
        self.progress_label.config(text=f"Progress: {completed_count}/{total_count} sessions ({completion_pct:.0f}%)")

    def _show_initial_session_message(self):
        """Show initial message when starting a session"""
        if self.ai_system.is_available:
            initial_msg = self.ai_system.initial_session_message(self.session_manager.progress.current_session_id)
            self._log_output(f"🤖 Sandra: {initial_msg}", 'session_msg')
        else:
            self._log_output("👋 Welcome! Start by typing the reference code from the left panel to practice.", 'session_msg')

    def _on_session_change(self, selected_session):
        """Handle session change from dropdown"""
        session_mapping = {
            "🐍 Python Fundamentals": "python_fundamentals",
            "🔢 PyTorch & NumPy": "pytorch_numpy",
            "🧠 Neural Networks": "neural_networks",
            "⬅️ Backpropagation": "backpropagation", 
            "🛡️ Regularization": "regularization",
            "📉 Loss & Optimizers": "loss_optimizers",
            "🏗️ LLM Architecture": "llm_architecture",
            "🔤 Tokenization & BPE": "tokenization_bpe",
            "🎯 RoPE & Attention": "rope_attention",
            "⚖️ RMS Normalization": "rms_norm",
            "🔄 FFN & Activations": "ffn_activations",
            "🚂 Training LLMs": "training_llm",
            "🎯 Inference & Generation": "inference_generation"
        }
        
        new_session_id = session_mapping.get(selected_session)
        if new_session_id and new_session_id != self.session_manager.progress.current_session_id:
            self.session_manager.progress.current_session_id = new_session_id
            self._load_current_session()
            self._clear_output()
            self._show_initial_session_message()
            self.status_label.config(text=f"📚 Switched to: {selected_session}")

    def _next_session(self):
        """Move to the next session"""
        next_session_id = self.session_manager.get_next_session()
        if next_session_id:
            # Mark current session as complete
            self.session_manager.mark_session_complete(self.session_manager.progress.current_session_id)
            
            # Move to next session
            self.session_manager.progress.current_session_id = next_session_id
            self._load_current_session()
            self._clear_output()
            self._show_initial_session_message()
            
            next_session = self.session_manager.get_session(next_session_id)
            self.status_label.config(text=f"🎉 Advanced to: {next_session.title}")
        else:
            messagebox.showinfo("Congratulations!", "🎉 You've completed all sessions! You're now ready to build your own LLMs!")

    def _copy_reference(self):
        """Copy reference code to clipboard"""
        current_session = self.session_manager.get_session(self.session_manager.progress.current_session_id)
        if current_session:
            self.root.clipboard_clear()
            self.root.clipboard_append(current_session.reference_code)
            self.status_label.config(text="📋 Reference code copied! But try typing it manually for better learning.")

    def _get_hint(self):
        """Get a hint for the current session"""
        current_session = self.session_manager.get_session(self.session_manager.progress.current_session_id)
        if current_session and current_session.hints:
            import random
            hint = random.choice(current_session.hints)
            self._log_output(f"\n💡 Sandra's Hint: {hint}", 'hint')

    def _clear_output(self):
        """Clear the output area"""
        self.output_text.config(state='normal')
        self.output_text.delete('1.0', 'end')
        self.output_text.config(state='disabled')

    def _start_panel_animation(self):
        self.is_loading = True
        self.loading_label.pack(pady=20)
        self.output_text.pack_forget()
        self._animate_panel()

    def _stop_panel_animation(self):
        self.is_loading = False
        self.loading_label.pack_forget()
        self.output_text.pack(fill='both', expand=True, pady=5, padx=10)

    def _animate_panel(self):
        if self.is_loading:
            char = self.panel_animation_chars[self.panel_animation_index]
            self.loading_label.config(text=f"Sandra is analyzing your code... {char}")
            self.panel_animation_index = (self.panel_animation_index + 1) % len(self.panel_animation_chars)
            self.root.after(150, self._animate_panel)
            
    def _animate_button(self, button, loading_text):
        """Animates the text of a specific button."""
        if self.is_loading:
            frame = self.button_animation_chars[self.button_animation_index]
            button.config(text=f"{loading_text}{frame}")
            self.button_animation_index = (self.button_animation_index + 1) % len(self.button_animation_chars)
            self.root.after(250, lambda: self._animate_button(button, loading_text))

    def _log_output(self, message, tag=None):
        self.output_text.config(state='normal')
        self.output_text.insert('end', message + '\n', tag)
        self.output_text.config(state='disabled')
        self.output_text.see('end')

    def _run_code(self):
        if self.is_loading: 
            return
            
        code = self.code_editor.get_text()
        if not code.strip():
            messagebox.showwarning("Input Error", "Code editor is empty. Try typing some code first!")
            return
        
        self._set_ui_loading(True)
        self._animate_button(self.run_button, "Running")
        self.tts_system.stop_speech()
        
        # Clear output
        self._clear_output()
        
        self._start_panel_animation()
        
        # Run code in separate thread
        threading.Thread(target=self._execute_code, args=(code,), daemon=True).start()

    def _set_ui_loading(self, is_loading):
        self.is_loading = is_loading
        state = tk.DISABLED if is_loading else tk.NORMAL
        
        self.run_button.config(state=state)
        
        # Restore button text when not loading
        if not is_loading:
            self.run_button.config(text="▶ Run Code")

    def _execute_code(self, code: str):
        output_buffer = io.StringIO()
        error_buffer = io.StringIO()
        
        # Redirect stdout and stderr
        sys.stdout, sys.stderr = output_buffer, error_buffer
        error_msg = ""
        
        try:
            # Execute the code
            exec(code, {'__builtins__': __builtins__})
        except Exception as e:
            error_msg = f"{type(e).__name__}: {e}"
        finally:
            # Restore stdout and stderr
            sys.stdout, sys.stderr = sys.__stdout__, sys.__stderr__
        
        # Process results on main thread
        self.root.after(0, self._process_execution_result, code, output_buffer.getvalue(), error_msg)

    def _process_execution_result(self, code, output, error):
        self._stop_panel_animation()
        
        # Generate AI feedback
        current_session_id = self.session_manager.progress.current_session_id
        feedback_text = self.ai_system.generate_feedback(code, error, current_session_id)
        
        # Display results
        if error:
            self._log_output(f"❌ ERROR:\n{error}", 'error')
            self._log_output(f"\n🔍 Debug tip: Check syntax, indentation, and variable names.", 'info')
        else:
            self._log_output("✅ SUCCESS! Code executed without errors.", 'success')
            if output:
                self._log_output(f"\n📋 Output:\n{output}", 'info')
        
        # Display AI feedback
        if feedback_text:
            self._log_output(f"\n🤖 Sandra says: {feedback_text}", 'ai_feedback')
            if self.audio_var.get():
                self._speak_with_stop_button(feedback_text)
        
        self.status_label.config(text="🧠 LLM Training Mentor Ready.")
        self._set_ui_loading(False)
                
    def _speak_with_stop_button(self, text):
        self.stop_audio_button.config(state=tk.NORMAL)
        self.tts_system.speak(text)
        
        def check_status():
            if not self.tts_system.is_speaking:
                self.stop_audio_button.config(state=tk.DISABLED)
            else:
                self.root.after(100, check_status)
        self.root.after(100, check_status)

    def _save_code(self):
        filepath = filedialog.asksaveasfilename(
            defaultextension=".py", 
            filetypes=[("Python Files", "*.py"), ("All Files", "*.*")]
        )
        if filepath:
            try:
                with open(filepath, 'w', encoding='utf-8') as f:
                    f.write(self.code_editor.get_text())
                messagebox.showinfo("Success", f"Code saved to {filepath}")
                self.status_label.config(text=f"💾 Code saved to {filepath}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save file: {e}")

    def _load_code(self):
        filepath = filedialog.askopenfilename(
            filetypes=[("Python Files", "*.py"), ("All Files", "*.*")]
        )
        if filepath:
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    self.code_editor.set_text(f.read())
                self.status_label.config(text=f"📁 Code loaded from {filepath}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load file: {e}")
                
    def on_closing(self):
        if messagebox.askokcancel("Quit", "Do you want to exit the LLM Learning Academy?"):
            self.tts_system.stop_speech()
            self.executor.shutdown(wait=False)
            self.root.destroy()


def main():
    # Check dependencies
    if not LLAMA_CPP_AVAILABLE:
        messagebox.showerror("Fatal Error", "Llama CPP library not found. Please run:\npip install llama-cpp-python")
        return
        
    if not KOKORO_TTS_AVAILABLE:
        messagebox.showwarning("Dependency Warning", 
                              "Kokoro TTS not found. Audio features disabled.\n"
                              "To enable audio: pip install kokoro-tts torch sounddevice")

    # Initialize main window
    root = tk.Tk()
    root.withdraw()  # Hide until setup complete

    def on_setup_complete(model_path):
        if model_path:
            root.deiconify()  # Show main window
            app = SessionBasedLLMLearningDashboard(root, model_path)
        else:
            messagebox.showerror("Setup Failed", "AI model setup failed. The application will now close.")
            root.destroy()

    # Start model download and setup
    downloader = ModelDownloader(on_complete=on_setup_complete)
    downloader.run()
    
    # Start main event loop
    root.mainloop()

if __name__ == "__main__":
    main()