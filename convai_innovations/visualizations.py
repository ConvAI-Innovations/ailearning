"""
Interactive visualizations for ML concepts in ConvAI Innovations platform.
Enhanced with audio feedback, equations, and better animations.
"""

import tkinter as tk
from tkinter import ttk, Canvas
import math
import numpy as np
from typing import List, Tuple, Optional, Dict, Any
import threading
import time

from .models import VisualizationConfig


class BaseVisualization:
    """Base class for all visualizations with audio feedback support"""
    
    def __init__(self, parent, config: VisualizationConfig, tts_system=None):
        self.parent = parent
        self.config = config
        self.tts_system = tts_system
        self.canvas = None
        self.animation_running = False
        self.animation_thread = None
        self.current_step = 0
        self.max_steps = 10
        
    def create_canvas(self) -> Canvas:
        """Create and return canvas widget"""
        self.canvas = Canvas(
            self.parent,
            width=self.config.width,
            height=self.config.height,
            bg=self.config.background_color,
            highlightthickness=0
        )
        return self.canvas
    
    def speak_explanation(self, text: str):
        """Speak explanation if TTS is available"""
        if self.tts_system and hasattr(self.tts_system, 'speak'):
            self.tts_system.speak(text)
    
    def start_animation(self):
        """Start animation if enabled"""
        if self.config.animate and not self.animation_running:
            self.animation_running = True
            self.current_step = 0
            self.animation_thread = threading.Thread(target=self._animate, daemon=True)
            self.animation_thread.start()
    
    def stop_animation(self):
        """Stop animation"""
        self.animation_running = False
    
    def _animate(self):
        """Animation loop - override in subclasses"""
        pass
    
    def clear(self):
        """Clear the canvas"""
        if self.canvas:
            self.canvas.delete("all")
    
    def draw_equation(self, x: int, y: int, equation: str, size: int = 12):
        """Draw mathematical equation on canvas"""
        self.canvas.create_text(
            x, y, text=equation, fill="#ffffff", 
            font=("Times New Roman", size), anchor="center"
        )


class LossFunctionVisualization(BaseVisualization):
    """Comprehensive loss function visualization with 3D landscapes"""
    
    def __init__(self, parent, config: VisualizationConfig, tts_system=None):
        super().__init__(parent, config, tts_system)
        self.current_loss_type = "mse"
        self.data_points = []
        self.predictions = []
        self.targets = []
        
    def draw_loss_landscape(self, loss_type: str = "mse"):
        """Draw interactive loss function landscape"""
        self.clear()
        self.current_loss_type = loss_type
        
        # Title and equation
        title_text = f"{loss_type.upper()} Loss Function Visualization"
        self.canvas.create_text(
            self.config.width // 2, 30, text=title_text,
            fill="#ffffff", font=("Arial", 16, "bold")
        )
        
        # Draw loss equation
        equations = {
            "mse": "MSE = (1/n) Σ(yᵢ - ŷᵢ)²",
            "cross_entropy": "CE = -Σ yᵢ log(ŷᵢ)",
            "mae": "MAE = (1/n) Σ|yᵢ - ŷᵢ|",
            "huber": "Huber = { ½(y-ŷ)² if |y-ŷ| ≤ δ, δ|y-ŷ| - ½δ² otherwise }"
        }
        
        self.draw_equation(self.config.width // 2, 60, equations.get(loss_type, ""), 14)
        
        # Generate sample data
        self._generate_sample_data()
        
        # Draw loss landscape (2D representation of 3D surface)
        self._draw_loss_surface()
        
        # Draw data points and predictions
        self._draw_data_points()
        
        # Draw loss value evolution
        self._draw_loss_evolution()
        
        # Add interactive controls explanation
        self._add_controls_explanation()
        
        # Audio explanation
        explanations = {
            "mse": "Mean Squared Error penalizes large errors quadratically. The loss surface is smooth and convex.",
            "cross_entropy": "Cross-entropy loss is used for classification. It penalizes confident wrong predictions heavily.",
            "mae": "Mean Absolute Error is robust to outliers. The loss surface has sharp corners.",
            "huber": "Huber loss combines MSE and MAE benefits. It's quadratic for small errors, linear for large ones."
        }
        
        self.speak_explanation(explanations.get(loss_type, ""))
    
    def _generate_sample_data(self):
        """Generate sample data points for visualization"""
        np.random.seed(42)  # For reproducible results
        self.targets = np.random.randn(20) * 2 + 5
        self.predictions = self.targets + np.random.randn(20) * 1.5
        
    def _draw_loss_surface(self):
        """Draw 2D representation of loss surface"""
        # Create grid for loss surface
        surface_x = 150
        surface_y = 120
        surface_w = 300
        surface_h = 200
        
        # Draw grid background
        self.canvas.create_rectangle(
            surface_x, surface_y, surface_x + surface_w, surface_y + surface_h,
            fill="#1a1a2e", outline="#16213e", width=2
        )
        
        # Draw contour lines representing loss levels
        center_x = surface_x + surface_w // 2
        center_y = surface_y + surface_h // 2
        
        colors = ["#ff4757", "#ff6b7a", "#ff7675", "#fdcb6e", "#6c5ce7"]
        
        for i, color in enumerate(colors):
            radius = 20 + i * 25
            self.canvas.create_oval(
                center_x - radius, center_y - radius,
                center_x + radius, center_y + radius,
                outline=color, width=2, fill=""
            )
        
        # Add labels
        self.canvas.create_text(
            surface_x + surface_w // 2, surface_y - 15,
            text="Loss Landscape", fill="#ffffff", font=("Arial", 12, "bold")
        )
        
        # Add axis labels
        self.canvas.create_text(
            surface_x - 20, center_y, text="Weight 1", 
            fill="#cccccc", angle=90, font=("Arial", 10)
        )
        self.canvas.create_text(
            center_x, surface_y + surface_h + 20, text="Weight 2",
            fill="#cccccc", font=("Arial", 10)
        )
        
        # Draw minimum point
        self.canvas.create_oval(
            center_x - 5, center_y - 5, center_x + 5, center_y + 5,
            fill="#00ff00", outline="#ffffff", width=2
        )
        self.canvas.create_text(
            center_x + 20, center_y - 10, text="Global Minimum",
            fill="#00ff00", font=("Arial", 9)
        )
    
    def _draw_data_points(self):
        """Draw actual vs predicted values"""
        data_x = 500
        data_y = 120
        data_w = 250
        data_h = 200
        
        # Background
        self.canvas.create_rectangle(
            data_x, data_y, data_x + data_w, data_y + data_h,
            fill="#2c3e50", outline="#34495e", width=2
        )
        
        # Title
        self.canvas.create_text(
            data_x + data_w // 2, data_y - 15,
            text="Actual vs Predicted", fill="#ffffff", font=("Arial", 12, "bold")
        )
        
        # Draw diagonal line (perfect predictions)
        self.canvas.create_line(
            data_x + 20, data_y + data_h - 20,
            data_x + data_w - 20, data_y + 20,
            fill="#95a5a6", width=2, dash=(5, 5)
        )
        
        # Plot data points
        for i, (target, pred) in enumerate(zip(self.targets[:10], self.predictions[:10])):
            # Normalize to canvas coordinates
            norm_target = (target - np.min(self.targets)) / (np.max(self.targets) - np.min(self.targets))
            norm_pred = (pred - np.min(self.predictions)) / (np.max(self.predictions) - np.min(self.predictions))
            
            x = data_x + 20 + norm_target * (data_w - 40)
            y = data_y + data_h - 20 - norm_pred * (data_h - 40)
            
            # Draw point
            color = "#e74c3c" if abs(target - pred) > 1 else "#2ecc71"
            self.canvas.create_oval(
                x - 4, y - 4, x + 4, y + 4,
                fill=color, outline="#ffffff", width=1
            )
        
        # Axes labels
        self.canvas.create_text(
            data_x + data_w // 2, data_y + data_h + 15,
            text="Actual Values", fill="#cccccc", font=("Arial", 10)
        )
        self.canvas.create_text(
            data_x - 25, data_y + data_h // 2,
            text="Predicted", fill="#cccccc", angle=90, font=("Arial", 10)
        )
    
    def _draw_loss_evolution(self):
        """Draw loss evolution during training"""
        evo_x = 150
        evo_y = 350
        evo_w = 500
        evo_h = 150
        
        # Background
        self.canvas.create_rectangle(
            evo_x, evo_y, evo_x + evo_w, evo_y + evo_h,
            fill="#34495e", outline="#2c3e50", width=2
        )
        
        # Title
        self.canvas.create_text(
            evo_x + evo_w // 2, evo_y - 15,
            text="Loss Evolution During Training", fill="#ffffff", font=("Arial", 12, "bold")
        )
        
        # Generate sample loss curve
        epochs = np.arange(0, 100, 1)
        if self.current_loss_type == "mse":
            losses = 10 * np.exp(-epochs / 20) + np.random.normal(0, 0.1, len(epochs))
        elif self.current_loss_type == "cross_entropy":
            losses = 2.5 * np.exp(-epochs / 15) + 0.1 + np.random.normal(0, 0.05, len(epochs))
        else:
            losses = 5 * np.exp(-epochs / 25) + np.random.normal(0, 0.08, len(epochs))
        
        # Normalize and draw curve
        max_loss = np.max(losses)
        min_loss = np.min(losses)
        
        points = []
        for i, loss in enumerate(losses[::2]):  # Every 2nd point for smoother curve
            x = evo_x + 20 + (i * 2 / len(epochs)) * (evo_w - 40)
            y = evo_y + evo_h - 20 - ((loss - min_loss) / (max_loss - min_loss)) * (evo_h - 40)
            points.extend([x, y])
        
        if len(points) > 4:
            self.canvas.create_line(points, fill="#3498db", width=3, smooth=True)
        
        # Add axes
        self.canvas.create_line(
            evo_x + 20, evo_y + evo_h - 20,
            evo_x + evo_w - 20, evo_y + evo_h - 20,
            fill="#bdc3c7", width=2
        )
        self.canvas.create_line(
            evo_x + 20, evo_y + 20,
            evo_x + 20, evo_y + evo_h - 20,
            fill="#bdc3c7", width=2
        )
        
        # Labels
        self.canvas.create_text(
            evo_x + evo_w // 2, evo_y + evo_h + 15,
            text="Training Epochs", fill="#cccccc", font=("Arial", 10)
        )
        self.canvas.create_text(
            evo_x - 15, evo_y + evo_h // 2,
            text="Loss Value", fill="#cccccc", angle=90, font=("Arial", 10)
        )
    
    def _add_controls_explanation(self):
        """Add explanation of interactive controls"""
        controls_text = [
            "📊 Loss Function Properties:",
            f"• Convexity: {'Convex' if self.current_loss_type in ['mse', 'cross_entropy'] else 'Non-convex'}",
            f"• Differentiability: {'Smooth' if self.current_loss_type != 'mae' else 'Sharp at zero'}",
            f"• Robustness: {'High' if self.current_loss_type in ['mae', 'huber'] else 'Low'}"
        ]
        
        for i, text in enumerate(controls_text):
            self.canvas.create_text(
                50, 520 + i * 20, text=text, fill="#ecf0f1", 
                font=("Arial", 10, "bold" if i == 0 else "normal"), anchor="w"
            )
    
    def animate_gradient_descent(self):
        """Animate gradient descent on loss surface"""
        self.current_step = 0
        self.max_steps = 20
        self.start_animation()
    
    def _animate(self):
        """Animation loop for gradient descent"""
        surface_x = 150 + 150  # Center of loss surface
        surface_y = 120 + 100
        
        # Starting point (random)
        start_x = surface_x + np.random.uniform(-100, 100)
        start_y = surface_y + np.random.uniform(-80, 80)
        
        while self.animation_running and self.current_step < self.max_steps:
            # Calculate current position (moving towards center)
            progress = self.current_step / self.max_steps
            current_x = start_x + (surface_x - start_x) * progress
            current_y = start_y + (surface_y - start_y) * progress
            
            # Clear previous point
            self.canvas.delete("gradient_point")
            
            # Draw current position
            self.canvas.create_oval(
                current_x - 6, current_y - 6, current_x + 6, current_y + 6,
                fill="#ff6b6b", outline="#ffffff", width=2, tags="gradient_point"
            )
            
            # Draw trail
            if self.current_step > 0:
                prev_progress = (self.current_step - 1) / self.max_steps
                prev_x = start_x + (surface_x - start_x) * prev_progress
                prev_y = start_y + (surface_y - start_y) * prev_progress
                
                self.canvas.create_line(
                    prev_x, prev_y, current_x, current_y,
                    fill="#ff6b6b", width=2, tags="gradient_point"
                )
            
            self.current_step += 1
            time.sleep(0.2)
        
        self.animation_running = False


class NeuralNetworkVisualization(BaseVisualization):
    """Visualization for neural networks"""
    
    def __init__(self, parent, config: VisualizationConfig):
        super().__init__(parent, config)
        self.layers = [4, 6, 4, 2]  # Default network architecture
        self.weights = []
        self.activations = []
        self.neurons = []  # Store neuron positions
        
    def draw_network(self, layers: List[int] = None):
        """Draw the neural network"""
        if layers:
            self.layers = layers
            
        self.clear()
        self.neurons = []
        
        # Calculate positions
        layer_width = self.config.width // (len(self.layers) + 1)
        
        for layer_idx, layer_size in enumerate(self.layers):
            layer_neurons = []
            x = layer_width * (layer_idx + 1)
            
            # Calculate y positions for neurons in this layer
            if layer_size == 1:
                y_positions = [self.config.height // 2]
            else:
                y_start = self.config.height // 4
                y_end = 3 * self.config.height // 4
                y_positions = [y_start + i * (y_end - y_start) / (layer_size - 1) 
                             for i in range(layer_size)]
            
            for neuron_idx, y in enumerate(y_positions):
                # Draw connections to previous layer
                if layer_idx > 0:
                    for prev_y in self.neurons[layer_idx - 1]:
                        self.canvas.create_line(
                            x - layer_width, prev_y,
                            x, y,
                            fill="#444444",
                            width=1,
                            tags="connection"
                        )
                
                # Draw neuron
                neuron_id = self.canvas.create_oval(
                    x - 15, y - 15, x + 15, y + 15,
                    fill=self.config.primary_color,
                    outline="#ffffff",
                    width=2,
                    tags="neuron"
                )
                
                # Add layer labels
                if neuron_idx == 0:
                    layer_names = ["Input", "Hidden", "Hidden", "Output"]
                    if layer_idx < len(layer_names):
                        self.canvas.create_text(
                            x, y - 40,
                            text=layer_names[layer_idx],
                            fill="#ffffff",
                            font=("Arial", 10, "bold")
                        )
                
                layer_neurons.append(y)
            
            self.neurons.append(layer_neurons)
    
    def animate_forward_pass(self):
        """Animate forward propagation"""
        if not self.config.animate:
            return
            
        def highlight_layer(layer_idx):
            if layer_idx >= len(self.layers):
                return
                
            # Reset all neurons
            self.canvas.itemconfig("neuron", fill=self.config.primary_color)
            
            # Highlight current layer
            x = (self.config.width // (len(self.layers) + 1)) * (layer_idx + 1)
            for y in self.neurons[layer_idx]:
                items = self.canvas.find_overlapping(x-16, y-16, x+16, y+16)
                for item in items:
                    if "neuron" in self.canvas.gettags(item):
                        self.canvas.itemconfig(item, fill=self.config.success_color)
            
            self.parent.after(800, lambda: highlight_layer(layer_idx + 1))
        
        highlight_layer(0)


class BackpropagationVisualization(BaseVisualization):
    """Visualization for backpropagation"""
    
    def draw_backprop_flow(self):
        """Draw backpropagation gradient flow"""
        self.clear()
        
        # Draw simple 3-layer network
        layers = [3, 4, 2]
        self.draw_simple_network(layers)
        
        # Add gradient flow arrows
        self.add_gradient_arrows()
        
        # Add legend
        self.add_legend()
    
    def draw_simple_network(self, layers):
        """Draw a simple network for backprop visualization"""
        layer_width = self.config.width // (len(layers) + 1)
        self.neurons = []
        
        for layer_idx, layer_size in enumerate(layers):
            layer_neurons = []
            x = layer_width * (layer_idx + 1)
            
            y_positions = self._calculate_y_positions(layer_size)
            
            for y in y_positions:
                # Draw connections to previous layer
                if layer_idx > 0:
                    for prev_y in self.neurons[layer_idx - 1]:
                        self.canvas.create_line(
                            x - layer_width, prev_y, x, y,
                            fill="#666666", width=2
                        )
                
                # Draw neuron
                self.canvas.create_oval(
                    x - 12, y - 12, x + 12, y + 12,
                    fill=self.config.primary_color,
                    outline="#ffffff", 
                    width=2
                )
                
                layer_neurons.append(y)
            
            self.neurons.append(layer_neurons)
    
    def add_gradient_arrows(self):
        """Add gradient flow arrows"""
        # Draw arrows going backwards (right to left)
        for layer_idx in range(len(self.neurons) - 1, 0, -1):
            layer_width = self.config.width // (len(self.neurons) + 1)
            x_start = layer_width * (layer_idx + 1)
            x_end = layer_width * layer_idx
            
            for y_start in self.neurons[layer_idx]:
                for y_end in self.neurons[layer_idx - 1]:
                    # Draw gradient arrow
                    self.draw_arrow(
                        x_start - 15, y_start,
                        x_end + 15, y_end,
                        color=self.config.secondary_color,
                        arrow_size=8
                    )
    
    def draw_arrow(self, x1, y1, x2, y2, color, arrow_size=10):
        """Draw an arrow from (x1,y1) to (x2,y2)"""
        # Draw line
        self.canvas.create_line(x1, y1, x2, y2, fill=color, width=2)
        
        # Calculate arrow head
        angle = math.atan2(y2 - y1, x2 - x1)
        arrow_x1 = x2 - arrow_size * math.cos(angle - math.pi/6)
        arrow_y1 = y2 - arrow_size * math.sin(angle - math.pi/6)
        arrow_x2 = x2 - arrow_size * math.cos(angle + math.pi/6)
        arrow_y2 = y2 - arrow_size * math.sin(angle + math.pi/6)
        
        self.canvas.create_polygon(
            x2, y2, arrow_x1, arrow_y1, arrow_x2, arrow_y2,
            fill=color, outline=color
        )
    
    def add_legend(self):
        """Add legend explaining the visualization"""
        legend_x = 20
        legend_y = 20
        
        self.canvas.create_text(
            legend_x, legend_y,
            text="Backpropagation: Gradients Flow Backward",
            fill="#ffffff",
            font=("Arial", 12, "bold"),
            anchor="nw"
        )
        
        self.canvas.create_text(
            legend_x, legend_y + 25,
            text="🔵 Neurons    ➡️ Forward Pass    ⬅️ Gradient Flow",
            fill="#cccccc",
            font=("Arial", 10),
            anchor="nw"
        )
    
    def _calculate_y_positions(self, layer_size):
        """Calculate y positions for neurons in a layer"""
        if layer_size == 1:
            return [self.config.height // 2]
        else:
            y_start = self.config.height // 4
            y_end = 3 * self.config.height // 4
            return [y_start + i * (y_end - y_start) / (layer_size - 1) 
                   for i in range(layer_size)]


class ActivationFunctionVisualization(BaseVisualization):
    """Visualization for activation functions"""
    
    def __init__(self, parent, config: VisualizationConfig):
        super().__init__(parent, config)
        self.x_range = (-3, 3)
        self.y_range = (-1.5, 3)
        
    def draw_activation_function(self, func_name: str):
        """Draw specified activation function"""
        self.clear()
        
        # Draw axes
        self.draw_axes()
        
        # Generate points
        x_vals = np.linspace(self.x_range[0], self.x_range[1], 200)
        
        if func_name.lower() == "relu":
            y_vals = np.maximum(0, x_vals)
            color = "#ff6b6b"
        elif func_name.lower() == "sigmoid":
            y_vals = 1 / (1 + np.exp(-x_vals))
            color = "#4ecdc4"
        elif func_name.lower() == "tanh":
            y_vals = np.tanh(x_vals)
            color = "#45b7d1"
        elif func_name.lower() == "gelu":
            y_vals = x_vals * 0.5 * (1 + np.tanh(np.sqrt(2/np.pi) * (x_vals + 0.044715 * x_vals**3)))
            color = "#96ceb4"
        elif func_name.lower() == "silu":
            y_vals = x_vals / (1 + np.exp(-x_vals))
            color = "#ffeaa7"
        else:
            return
        
        # Convert to canvas coordinates
        points = []
        for x, y in zip(x_vals, y_vals):
            canvas_x = self.x_to_canvas(x)
            canvas_y = self.y_to_canvas(y)
            points.extend([canvas_x, canvas_y])
        
        # Draw the function
        self.canvas.create_line(points, fill=color, width=3, smooth=True)
        
        # Add title
        self.canvas.create_text(
            self.config.width // 2, 30,
            text=f"{func_name.upper()} Activation Function",
            fill="#ffffff",
            font=("Arial", 14, "bold")
        )
        
        # Add properties text
        properties = self.get_function_properties(func_name)
        self.canvas.create_text(
            20, self.config.height - 60,
            text=properties,
            fill="#cccccc",
            font=("Arial", 10),
            anchor="nw"
        )
    
    def draw_axes(self):
        """Draw coordinate axes"""
        # X-axis
        y_zero = self.y_to_canvas(0)
        self.canvas.create_line(
            50, y_zero, self.config.width - 50, y_zero,
            fill="#666666", width=2
        )
        
        # Y-axis
        x_zero = self.x_to_canvas(0)
        self.canvas.create_line(
            x_zero, 50, x_zero, self.config.height - 50,
            fill="#666666", width=2
        )
        
        # Add axis labels
        for x in range(int(self.x_range[0]), int(self.x_range[1]) + 1):
            if x != 0:
                canvas_x = self.x_to_canvas(x)
                self.canvas.create_line(canvas_x, y_zero - 5, canvas_x, y_zero + 5, fill="#888888")
                self.canvas.create_text(canvas_x, y_zero + 15, text=str(x), fill="#888888", font=("Arial", 8))
        
        for y in range(int(self.y_range[0]), int(self.y_range[1]) + 1):
            if y != 0:
                canvas_y = self.y_to_canvas(y)
                self.canvas.create_line(x_zero - 5, canvas_y, x_zero + 5, canvas_y, fill="#888888")
                self.canvas.create_text(x_zero - 15, canvas_y, text=str(y), fill="#888888", font=("Arial", 8))
    
    def x_to_canvas(self, x):
        """Convert x coordinate to canvas coordinate"""
        return 50 + (x - self.x_range[0]) * (self.config.width - 100) / (self.x_range[1] - self.x_range[0])
    
    def y_to_canvas(self, y):
        """Convert y coordinate to canvas coordinate"""
        return self.config.height - 50 - (y - self.y_range[0]) * (self.config.height - 100) / (self.y_range[1] - self.y_range[0])
    
    def get_function_properties(self, func_name):
        """Get properties text for the function"""
        properties = {
            "relu": "• Non-linear\n• Computationally efficient\n• Can cause dead neurons\n• Most common in hidden layers",
            "sigmoid": "• Smooth gradient\n• Output range: (0, 1)\n• Can cause vanishing gradients\n• Used in binary classification",
            "tanh": "• Zero-centered output\n• Output range: (-1, 1)\n• Better than sigmoid\n• Still suffers from vanishing gradients",
            "gelu": "• Smooth approximation of ReLU\n• Used in transformers\n• Better gradient flow\n• Probabilistic interpretation",
            "silu": "• Self-gated activation\n• Smooth and non-monotonic\n• Used in modern architectures\n• Also called Swish"
        }
        return properties.get(func_name.lower(), "")


class AttentionVisualization(BaseVisualization):
    """Visualization for self-attention mechanism"""
    
    def __init__(self, parent, config: VisualizationConfig):
        super().__init__(parent, config)
        self.tokens = ["The", "cat", "sat", "on", "the", "mat"]
        self.attention_weights = None
        
    def draw_attention_mechanism(self):
        """Draw self-attention visualization"""
        self.clear()
        
        # Generate dummy attention weights
        seq_len = len(self.tokens)
        self.attention_weights = np.random.rand(seq_len, seq_len)
        # Make it more diagonal (tokens attending to themselves more)
        for i in range(seq_len):
            self.attention_weights[i, i] += 0.5
        # Normalize
        self.attention_weights = self.attention_weights / self.attention_weights.sum(axis=1, keepdims=True)
        
        # Draw tokens
        token_positions = self.draw_tokens()
        
        # Draw attention connections
        self.draw_attention_connections(token_positions)
        
        # Add title and legend
        self.add_attention_legend()
    
    def draw_tokens(self):
        """Draw token boxes"""
        positions = []
        token_width = 80
        token_height = 40
        spacing = (self.config.width - len(self.tokens) * token_width) // (len(self.tokens) + 1)
        
        for i, token in enumerate(self.tokens):
            x = spacing + i * (token_width + spacing)
            y = self.config.height // 2 - token_height // 2
            
            # Draw token box
            self.canvas.create_rectangle(
                x, y, x + token_width, y + token_height,
                fill=self.config.primary_color,
                outline="#ffffff",
                width=2
            )
            
            # Draw token text
            self.canvas.create_text(
                x + token_width // 2, y + token_height // 2,
                text=token,
                fill="#ffffff",
                font=("Arial", 11, "bold")
            )
            
            positions.append((x + token_width // 2, y + token_height // 2))
        
        return positions
    
    def draw_attention_connections(self, positions):
        """Draw attention weight connections"""
        for i, (x1, y1) in enumerate(positions):
            for j, (x2, y2) in enumerate(positions):
                if i != j:  # Don't draw self-connections for clarity
                    weight = self.attention_weights[i, j]
                    if weight > 0.1:  # Only draw significant connections
                        # Line thickness based on attention weight
                        thickness = max(1, int(weight * 8))
                        
                        # Color intensity based on weight
                        alpha = int(weight * 255)
                        color = f"#{alpha:02x}{alpha//2:02x}00"  # Orange gradient
                        
                        # Draw curved line
                        self.draw_curved_line(x1, y1, x2, y2, color, thickness)
    
    def draw_curved_line(self, x1, y1, x2, y2, color, thickness):
        """Draw a curved line between two points"""
        # Calculate control points for bezier curve
        mid_x = (x1 + x2) // 2
        curve_height = 30 if y1 == y2 else 0
        
        # Simple curve using multiple line segments
        points = []
        for t in np.linspace(0, 1, 20):
            # Quadratic bezier curve
            x = (1-t)**2 * x1 + 2*(1-t)*t * mid_x + t**2 * x2
            y = (1-t)**2 * y1 + 2*(1-t)*t * (y1 - curve_height) + t**2 * y2
            points.extend([x, y])
        
        self.canvas.create_line(points, fill=color, width=thickness, smooth=True)
    
    def add_attention_legend(self):
        """Add legend and title"""
        self.canvas.create_text(
            self.config.width // 2, 30,
            text="Self-Attention Mechanism",
            fill="#ffffff",
            font=("Arial", 16, "bold")
        )
        
        self.canvas.create_text(
            self.config.width // 2, 60,
            text="Line thickness = attention weight strength",
            fill="#cccccc",
            font=("Arial", 10)
        )


class TokenizationVisualization(BaseVisualization):
    """Visualization for tokenization and BPE"""
    
    def draw_tokenization_process(self, text: str, method: str = "bpe"):
        """Draw tokenization process"""
        self.clear()
        
        # Add title
        self.canvas.create_text(
            self.config.width // 2, 30,
            text=f"{method.upper()} Tokenization Process",
            fill="#ffffff",
            font=("Arial", 16, "bold")
        )
        
        if method.lower() == "bpe":
            self.draw_bpe_process(text)
        elif method.lower() == "word":
            self.draw_word_tokenization(text)
        else:
            self.draw_character_tokenization(text)
    
    def draw_bpe_process(self, text: str):
        """Draw BPE tokenization process"""
        # Step 1: Show original text
        y_pos = 80
        self.canvas.create_text(
            50, y_pos,
            text="1. Original Text:",
            fill="#ffffff",
            font=("Arial", 12, "bold"),
            anchor="nw"
        )
        
        self.canvas.create_rectangle(
            50, y_pos + 25, self.config.width - 50, y_pos + 60,
            fill="#2d3748", outline="#4a5568", width=2
        )
        
        self.canvas.create_text(
            60, y_pos + 42,
            text=text,
            fill="#e2e8f0",
            font=("Arial", 11),
            anchor="nw"
        )
        
        # Step 2: Show subword tokens
        y_pos += 100
        self.canvas.create_text(
            50, y_pos,
            text="2. BPE Subword Tokens:",
            fill="#ffffff",
            font=("Arial", 12, "bold"),
            anchor="nw"
        )
        
        # Simulate BPE tokens
        tokens = self.simulate_bpe_tokens(text)
        self.draw_token_boxes(tokens, y_pos + 30)
        
        # Step 3: Show token IDs
        y_pos += 120
        self.canvas.create_text(
            50, y_pos,
            text="3. Token IDs:",
            fill="#ffffff",
            font=("Arial", 12, "bold"),
            anchor="nw"
        )
        
        token_ids = [f"[{i+100}]" for i in range(len(tokens))]
        self.draw_token_boxes(token_ids, y_pos + 30, color="#ffc107")
    
    def simulate_bpe_tokens(self, text: str) -> List[str]:
        """Simulate BPE tokenization"""
        # This is a simplified simulation
        words = text.split()
        tokens = []
        for word in words:
            if len(word) <= 3:
                tokens.append(word)
            else:
                # Split longer words into subwords
                tokens.append(word[:len(word)//2] + "▁")
                tokens.append(word[len(word)//2:])
        return tokens
    
    def draw_token_boxes(self, tokens: List[str], y_pos: int, color: str = None):
        """Draw token boxes"""
        if not color:
            color = self.config.primary_color
            
        x_pos = 60
        box_height = 35
        
        for token in tokens:
            box_width = max(60, len(token) * 10 + 20)
            
            # Draw token box
            self.canvas.create_rectangle(
                x_pos, y_pos, x_pos + box_width, y_pos + box_height,
                fill=color, outline="#ffffff", width=1
            )
            
            # Draw token text
            self.canvas.create_text(
                x_pos + box_width // 2, y_pos + box_height // 2,
                text=token,
                fill="#000000" if color == "#ffc107" else "#ffffff",
                font=("Arial", 10, "bold")
            )
            
            x_pos += box_width + 10
            
            # Wrap to next line if needed
            if x_pos > self.config.width - 100:
                x_pos = 60
                y_pos += 50
    
    def draw_word_tokenization(self, text: str):
        """Draw word-level tokenization"""
        tokens = text.split()
        y_pos = 80
        
        self.canvas.create_text(
            50, y_pos,
            text="Word Tokens:",
            fill="#ffffff",
            font=("Arial", 12, "bold"),
            anchor="nw"
        )
        
        self.draw_token_boxes(tokens, y_pos + 30)
    
    def draw_character_tokenization(self, text: str):
        """Draw character-level tokenization"""
        tokens = list(text.replace(" ", "▁"))  # Replace spaces with visible character
        y_pos = 80
        
        self.canvas.create_text(
            50, y_pos,
            text="Character Tokens:",
            fill="#ffffff",
            font=("Arial", 12, "bold"),
            anchor="nw"
        )
        
        self.draw_token_boxes(tokens, y_pos + 30)


class VisualizationManager:
    """Manager for all visualizations"""
    
    def __init__(self, parent):
        self.parent = parent
        self.config = VisualizationConfig()
        self.current_viz = None
        self.viz_frame = None
        
    def create_visualization_frame(self):
        """Create the main visualization frame"""
        self.viz_frame = ttk.Frame(self.parent, style='Left.TFrame')
        return self.viz_frame
    
    def show_visualization(self, viz_type: str, **kwargs):
        """Show specified visualization"""
        if self.viz_frame is None:
            return
            
        # Clear previous visualization
        if self.current_viz:
            self.current_viz.stop_animation()
            
        for widget in self.viz_frame.winfo_children():
            widget.destroy()
        
        # Create new visualization
        if viz_type == "neural_network":
            self.current_viz = NeuralNetworkVisualization(self.viz_frame, self.config)
            canvas = self.current_viz.create_canvas()
            canvas.pack(fill=tk.BOTH, expand=True)
            self.current_viz.draw_network(kwargs.get('layers', [4, 6, 4, 2]))
            
        elif viz_type == "backpropagation":
            self.current_viz = BackpropagationVisualization(self.viz_frame, self.config)
            canvas = self.current_viz.create_canvas()
            canvas.pack(fill=tk.BOTH, expand=True)
            self.current_viz.draw_backprop_flow()
            
        elif viz_type == "activation_function":
            self.current_viz = ActivationFunctionVisualization(self.viz_frame, self.config)
            canvas = self.current_viz.create_canvas()
            canvas.pack(fill=tk.BOTH, expand=True)
            func_name = kwargs.get('function', 'relu')
            self.current_viz.draw_activation_function(func_name)
            
        elif viz_type == "attention":
            self.current_viz = AttentionVisualization(self.viz_frame, self.config)
            canvas = self.current_viz.create_canvas()
            canvas.pack(fill=tk.BOTH, expand=True)
            self.current_viz.draw_attention_mechanism()
            
        elif viz_type == "tokenization":
            self.current_viz = TokenizationVisualization(self.viz_frame, self.config)
            canvas = self.current_viz.create_canvas()
            canvas.pack(fill=tk.BOTH, expand=True)
            text = kwargs.get('text', "Hello world! This is tokenization.")
            method = kwargs.get('method', 'bpe')
            self.current_viz.draw_tokenization_process(text, method)
            
        # Add control buttons
        self.add_visualization_controls(viz_type, **kwargs)
        
        if self.config.animate:
            self.current_viz.start_animation()
    
    def add_visualization_controls(self, viz_type: str, **kwargs):
        """Add control buttons for visualizations"""
        control_frame = ttk.Frame(self.viz_frame, style='Left.TFrame')
        control_frame.pack(side=tk.BOTTOM, fill=tk.X, padx=10, pady=5)
        
        if viz_type == "neural_network":
            ttk.Button(
                control_frame,
                text="▶ Forward Pass",
                command=lambda: self.current_viz.animate_forward_pass()
            ).pack(side=tk.LEFT, padx=5)
            
        elif viz_type == "activation_function":
            functions = ["ReLU", "Sigmoid", "Tanh", "GELU", "SiLU"]
            for func in functions:
                ttk.Button(
                    control_frame,
                    text=func,
                    command=lambda f=func: self.current_viz.draw_activation_function(f)
                ).pack(side=tk.LEFT, padx=2)
                
        elif viz_type == "tokenization":
            methods = ["BPE", "Word", "Character"]
            for method in methods:
                ttk.Button(
                    control_frame,
                    text=method,
                    command=lambda m=method: self.current_viz.draw_tokenization_process(
                        kwargs.get('text', "Hello world!"), m
                    )
                ).pack(side=tk.LEFT, padx=2)
    
    def update_config(self, **config_updates):
        """Update visualization configuration"""
        for key, value in config_updates.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)