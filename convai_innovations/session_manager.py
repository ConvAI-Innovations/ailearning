"""
Session management for ConvAI Innovations platform.
"""

from typing import Dict, List, Optional
from .models import Session, LearningProgress


class SessionManager:
    """Manages learning sessions and progression"""
    
    def __init__(self):
        self.sessions = self._create_sessions()
        self.progress = LearningProgress(total_sessions=len(self.sessions))
        
    def _create_sessions(self) -> Dict[str, Session]:
        """Create all learning sessions with visualization types"""
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
            ],
            visualization_type="python_basics"
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
            ],
            visualization_type="tensor_operations"
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
            ],
            visualization_type="neural_network"
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
            ],
            visualization_type="backpropagation"
        )
        
        # Continue with remaining sessions...
        
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