"""
ConvAI Innovations - Main Application (Refactored)
Interactive LLM Training Academy with multi-language support and visualizations.
"""

import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
import threading
import sys
import io
import time
from typing import Optional
import concurrent.futures

# Import our modular components
from .models import Language, CodeGenRequest
from .session_manager import SessionManager
from .ai_systems import LLMAIFeedbackSystem, EnhancedKokoroTTSSystem
from .visualizations import VisualizationManager
from .ui_components import (
    ModernCodeEditor, 
    CodeGenerationPanel, 
    LanguageSelector, 
    ModelDownloader
)


class SessionBasedLLMLearningDashboard:
    """Main application class - refactored and enhanced"""
    
    def __init__(self, root: tk.Tk, model_path: Optional[str], ai_system: Optional = None):
        self.root = root
        
        # Initialize core systems
        self.session_manager = SessionManager()
        self.ai_system = ai_system or LLMAIFeedbackSystem(model_path)
        self.tts_system = EnhancedKokoroTTSSystem()
        self.visualization_manager = VisualizationManager(None)  # Will be set later
        
        # UI state
        self.is_loading = False
        self.current_language = Language.ENGLISH
        
        # Animation state
        self.animation_chars = ['|', '/', '-', '\\']
        self.animation_index = 0
        
        # Thread executor
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=3)
        
        # Setup UI
        self._configure_styles()
        self._setup_window()
        self._create_main_interface()
        self._load_current_session()
        
        # Show initial message
        self._show_initial_session_message()
        
        # Check AI system status
        if not self.ai_system.is_available:
            self.status_label.config(text="🚨 AI Mentor Offline. Check console for errors.")

    def _setup_window(self):
        """Setup main window"""
        self.root.title("🧠 ConvAI Innovations - Interactive LLM Training Academy")
        
        # Dynamic window sizing
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        
        window_width = min(1600, int(screen_width * 0.9))
        window_height = min(1000, int(screen_height * 0.9))
        
        self.root.geometry(f"{window_width}x{window_height}")
        self.root.configure(bg='#1e1e1e')
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        
        # Center window
        self.root.update_idletasks()
        x = (screen_width // 2) - (window_width // 2)
        y = (screen_height // 2) - (window_height // 2)
        self.root.geometry(f'{window_width}x{window_height}+{x}+{y}')
        
        # Window properties
        self.root.resizable(True, True)
        self.root.minsize(1400, 800)

    def _configure_styles(self):
        """Configure TTK styles"""
        style = ttk.Style()
        style.theme_use('clam')
        
        # Define color scheme
        colors = {
            'bg': '#1e1e1e',
            'fg': '#f8f9fa',
            'accent': '#00aaff',
            'success': '#28a745',
            'warning': '#ffc107',
            'danger': '#dc3545',
            'secondary': '#6c757d'
        }
        
        # Button styles
        button_styles = {
            'TButton': {'background': '#3a3d41', 'foreground': 'white'},
            'Run.TButton': {'background': colors['success'], 'foreground': 'white'},
            'Clear.TButton': {'background': colors['danger'], 'foreground': 'white'},
            'Session.TButton': {'background': '#6f42c1', 'foreground': 'white'},
            'Next.TButton': {'background': colors['accent'], 'foreground': 'white'},
            'Generate.TButton': {'background': '#17a2b8', 'foreground': 'white'},
            'Edit.TButton': {'background': '#ffc107', 'foreground': 'black'}
        }
        
        for style_name, config in button_styles.items():
            style.configure(style_name, font=('Segoe UI', 10, 'bold'), 
                          padding=8, relief='flat', **config)
            style.map(style_name, background=[('active', self._darken_color(config['background']))])
        
        # Label styles
        style.configure('TLabel', background=colors['bg'], foreground=colors['fg'], 
                       font=('Segoe UI', 10))
        style.configure('Header.TLabel', font=('Segoe UI', 16, 'bold'), 
                       foreground=colors['accent'])
        style.configure('Session.TLabel', font=('Segoe UI', 12, 'bold'), 
                       foreground=colors['warning'])
        
        # Frame styles
        style.configure('TFrame', background=colors['bg'])
        style.configure('Left.TFrame', background='#252526')
        
        # Paned window
        style.configure('TPanedwindow', background=colors['bg'])
        style.configure('TPanedwindow.Sash', sashthickness=6, relief='flat', 
                       background='#3a3d41')

    def _darken_color(self, color):
        """Darken a color for hover effects"""
        color_map = {
            '#28a745': '#218838',
            '#dc3545': '#c82333', 
            '#00aaff': '#0056b3',
            '#6f42c1': '#5a32a3',
            '#17a2b8': '#138496',
            '#3a3d41': '#4a4d51',
            '#ffc107': '#e0a800'
        }
        return color_map.get(color, color)

    def _create_main_interface(self):
        """Create the main interface"""
        # Create main paned window
        main_pane = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        main_pane.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Left panel (sessions, reference, visualizations)
        left_panel = self._create_left_panel(main_pane)
        main_pane.add(left_panel, weight=1)
        
        # Right panel (code editor and output)
        right_pane = ttk.PanedWindow(main_pane, orient=tk.VERTICAL)
        self._create_right_panel(right_pane)
        main_pane.add(right_pane, weight=2)
        
        # Status bar
        self._create_status_bar()
        
        # Configure pane sizes
        self.root.after(100, lambda: self._configure_pane_sizes(main_pane))

    def _create_left_panel(self, parent):
        """Create enhanced left panel with tabs"""
        left_frame = ttk.Frame(parent, style='Left.TFrame')
        
        # Create notebook for tabs
        notebook = ttk.Notebook(left_frame)
        notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Session tab
        session_tab = self._create_session_tab(notebook)
        notebook.add(session_tab, text="📚 Sessions")
        
        # Visualization tab
        viz_tab = self._create_visualization_tab(notebook)
        notebook.add(viz_tab, text="📊 Visualizations")
        
        # AI Code Gen tab
        if self.ai_system.is_available:
            codegen_tab = self._create_codegen_tab(notebook)
            notebook.add(codegen_tab, text="🤖 AI Code")
        
        # Settings tab
        settings_tab = self._create_settings_tab(notebook)
        notebook.add(settings_tab, text="⚙️ Settings")
        
        return left_frame

    def _create_session_tab(self, parent):
        """Create session management tab with editable reference code"""
        session_frame = ttk.Frame(parent, style='Left.TFrame')
        
        # Header
        header_frame = ttk.Frame(session_frame, style='Left.TFrame')
        header_frame.pack(fill='x', padx=8, pady=8)
        
        ttk.Label(header_frame, text="🎯 Learning Sessions", 
                 style='Header.TLabel', background='#252526').pack(anchor='w')
        
        # Current session info
        current_session = self.session_manager.get_session(
            self.session_manager.progress.current_session_id
        )
        session_title = current_session.title if current_session else "No Session"
        self.current_session_label = ttk.Label(
            header_frame, text=f"Current: {session_title}", 
            style='Session.TLabel', background='#252526'
        )
        self.current_session_label.pack(anchor='w', pady=(3, 0))
        
        # Session navigation
        nav_frame = ttk.Frame(session_frame, style='Left.TFrame')
        nav_frame.pack(fill='x', padx=8, pady=3)
        
        # Session dropdown
        session_names = [
            "🐍 Python Fundamentals", "🔢 PyTorch & NumPy", "🧠 Neural Networks",
            "⬅️ Backpropagation", "🛡️ Regularization", "📉 Loss & Optimizers", 
            "🏗️ LLM Architecture", "🔤 Tokenization & BPE", "🎯 RoPE & Attention",
            "⚖️ RMS Normalization", "🔄 FFN & Activations", "🚂 Training LLMs",
            "🎯 Inference & Generation"
        ]
        
        self.session_var = tk.StringVar(value=session_names[0])
        session_dropdown = ttk.OptionMenu(
            nav_frame, self.session_var, session_names[0], *session_names, 
            command=self._on_session_change
        )
        session_dropdown.pack(side='left', fill='x', expand=True, padx=(0, 3))
        
        self.next_session_button = ttk.Button(
            nav_frame, text="Next →", command=self._next_session, 
            style='Next.TButton'
        )
        self.next_session_button.pack(side='right', padx=(3, 0))
        
        # Reference code section with edit controls
        ref_frame = ttk.Frame(session_frame, style='Left.TFrame')
        ref_frame.pack(fill='both', expand=True, padx=8, pady=3)
        
        # Reference code header with edit toggle
        ref_header_frame = ttk.Frame(ref_frame, style='Left.TFrame')
        ref_header_frame.pack(fill='x', pady=(0, 3))
        
        ttk.Label(ref_header_frame, text="📖 Reference Code", 
                 style='TLabel', background='#252526').pack(side='left')
        
        # Edit mode toggle
        self.edit_mode_var = tk.BooleanVar(value=True)
        self.edit_toggle_button = ttk.Button(
            ref_header_frame, text="✏️ Edit", command=self._toggle_edit_mode,
            style='Edit.TButton'
        )
        self.edit_toggle_button.pack(side='right', padx=(5, 0))
        
        # Editable reference text
        self.reference_text = scrolledtext.ScrolledText(
            ref_frame, wrap=tk.WORD, font=('Consolas', 9), 
            bg='#2d3748', fg='#e2e8f0', bd=1, relief='solid', 
            padx=8, pady=8, height=18, insertbackground='white'
        )
        self.reference_text.pack(fill='both', expand=True, pady=3)
        
        # Initially in read-only mode
        self.reference_text.config(state='disabled')
        
        # Action buttons for reference code
        ref_action_frame = ttk.Frame(ref_frame, style='Left.TFrame')
        ref_action_frame.pack(fill='x', pady=3)
        
        ttk.Button(ref_action_frame, text="💡 Hint", command=self._get_hint, 
                  style='Session.TButton').pack(side='left', fill='x', expand=True, padx=(0, 2))
        ttk.Button(ref_action_frame, text="📋 Copy", command=self._copy_reference, 
                  style='TButton').pack(side='left', fill='x', expand=True, padx=2)
        ttk.Button(ref_action_frame, text="🗑️ Clear", command=self._clear_reference, 
                  style='Clear.TButton').pack(side='left', fill='x', expand=True, padx=2)
        ttk.Button(ref_action_frame, text="🔄 Reset", command=self._reset_reference, 
                  style='TButton').pack(side='left', fill='x', expand=True, padx=(2, 0))
        
        # Action buttons
        action_frame = ttk.Frame(session_frame, style='Left.TFrame')
        action_frame.pack(fill='x', padx=8, pady=5)
        
        ttk.Button(action_frame, text="→ Copy to Editor", command=self._copy_ref_to_editor, 
                  style='Session.TButton').pack(side='left', fill='x', expand=True, padx=(0, 3))
        ttk.Button(action_frame, text="💾 Save Reference", command=self._save_reference_to_file, 
                  style='TButton').pack(side='left', fill='x', expand=True, padx=3)
        
        # Audio controls at bottom of session tab
        audio_controls_frame = ttk.Frame(session_frame, style='Left.TFrame')
        audio_controls_frame.pack(fill='x', padx=8, pady=(10, 5))
        
        # Audio status and controls
        self.audio_status_label = ttk.Label(
            audio_controls_frame, text="🔇 Audio: Ready", 
            style='TLabel', background='#252526', foreground='#28a745'
        )
        self.audio_status_label.pack(side='left', padx=(0, 10))
        
        # Stop audio button - prominently placed
        self.stop_audio_button = ttk.Button(
            audio_controls_frame, text="⏹️ Stop Audio", 
            command=self.tts_system.stop_speech,
            style='Clear.TButton'
        )
        self.stop_audio_button.pack(side='right')
        
        return session_frame

    def _toggle_edit_mode(self):
        """Toggle edit mode for reference code"""
        if self.edit_mode_var.get():
            # Switch to read-only mode
            self.reference_text.config(state='disabled')
            self.edit_toggle_button.config(text="✏️ Edit")
            self.edit_mode_var.set(False)
            self.status_label.config(text="📖 Reference code is now read-only")
        else:
            # Switch to edit mode
            self.reference_text.config(state='normal')
            self.edit_toggle_button.config(text="🔒 Lock")
            self.edit_mode_var.set(True)
            self.status_label.config(text="✏️ Reference code is now editable - paste your ChatGPT code here!")

    def _clear_reference(self):
        """Clear reference code area"""
        if self.edit_mode_var.get():
            self.reference_text.delete('1.0', 'end')
            self.status_label.config(text="🗑️ Reference code cleared")
        else:
            messagebox.showinfo("Edit Mode Required", "Enable edit mode first to clear the reference code")

    def _reset_reference(self):
        """Reset reference code to original session content"""
        current_session = self.session_manager.get_session(
            self.session_manager.progress.current_session_id
        )
        if current_session:
            # Temporarily enable editing
            original_state = self.reference_text.cget('state')
            self.reference_text.config(state='normal')
            
            # Reset content
            self.reference_text.delete('1.0', 'end')
            full_content = (
                current_session.description + "\n\n" + "="*60 + 
                "\nREFERENCE CODE TO TYPE:\n" + "="*60 + "\n\n" + 
                current_session.reference_code
            )
            self.reference_text.insert('1.0', full_content)
            
            # Restore original state
            self.reference_text.config(state=original_state)
            self.status_label.config(text="🔄 Reference code reset to original session content")

    def _copy_ref_to_editor(self):
        """Copy reference code to main editor"""
        content = self.reference_text.get('1.0', 'end-1c')
        if content.strip():
            # Extract just the code part (after the last separator)
            parts = content.split("="*60)
            if len(parts) >= 3:
                code_content = parts[-1].strip()
            else:
                code_content = content
            
            result = messagebox.askyesno(
                "Copy to Editor", 
                "This will replace all code in the main editor. Continue?"
            )
            if result:
                self.code_editor.set_text(code_content)
                self.status_label.config(text="📋 Reference code copied to main editor")
        else:
            messagebox.showinfo("No Content", "Reference code area is empty")

    def _save_reference_to_file(self):
        """Save reference code to file"""
        from tkinter import filedialog
        content = self.reference_text.get('1.0', 'end-1c')
        if content.strip():
            filepath = filedialog.asksaveasfilename(
                defaultextension=".py", 
                filetypes=[("Python Files", "*.py"), ("Text Files", "*.txt"), ("All Files", "*.*")]
            )
            if filepath:
                try:
                    with open(filepath, 'w', encoding='utf-8') as f:
                        f.write(content)
                    messagebox.showinfo("Success", f"Reference code saved to {filepath}")
                    self.status_label.config(text=f"💾 Reference code saved to {filepath}")
                except Exception as e:
                    messagebox.showerror("Error", f"Failed to save file: {e}")
        else:
            messagebox.showinfo("No Content", "Reference code area is empty")

    def _create_visualization_tab(self, parent):
        """Create visualization tab"""
        viz_frame = ttk.Frame(parent, style='Left.TFrame')
        
        # Header
        ttk.Label(viz_frame, text="📊 Concept Visualizations", 
                 style='Header.TLabel', background='#252526').pack(pady=10)
        
        # Visualization controls
        control_frame = ttk.Frame(viz_frame, style='Left.TFrame')
        control_frame.pack(fill='x', padx=10, pady=5)
        
        ttk.Label(control_frame, text="Select Visualization:", 
                 style='TLabel', background='#252526').pack(anchor='w')
        
        self.viz_var = tk.StringVar(value="Neural Network")
        viz_options = [
            "Neural Network", "Backpropagation", "Activation Functions", 
            "Self-Attention", "Tokenization", "Loss Functions"
        ]
        
        viz_dropdown = ttk.OptionMenu(
            control_frame, self.viz_var, viz_options[0], *viz_options,
            command=self._on_visualization_change
        )
        viz_dropdown.pack(fill='x', pady=5)
        
        # Visualization display area
        self.visualization_manager.parent = viz_frame
        viz_display = self.visualization_manager.create_visualization_frame()
        viz_display.pack(fill='both', expand=True, padx=5, pady=5)
        
        # Default visualization
        self.visualization_manager.show_visualization("neural_network")
        
        return viz_frame

    def _create_codegen_tab(self, parent):
        """Create AI code generation tab"""
        # This will be set when we create the code editor
        self.codegen_placeholder = ttk.Frame(parent, style='Left.TFrame')
        return self.codegen_placeholder

    def _create_settings_tab(self, parent):
        """Create settings tab"""
        settings_frame = ttk.Frame(parent, style='Left.TFrame')
        
        # Language settings
        self.language_selector = LanguageSelector(settings_frame, self.tts_system)
        self.language_selector.pack(fill='x', padx=10, pady=10)
        
        # Audio settings
        audio_frame = ttk.Frame(settings_frame, style='Left.TFrame')
        audio_frame.pack(fill='x', padx=10, pady=5)
        
        ttk.Label(audio_frame, text="🔊 Audio Settings", 
                 style='TLabel', background='#252526').pack(anchor='w')
        
        self.audio_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(audio_frame, text="Enable Sandra's Voice", 
                       variable=self.audio_var, style='TCheckbutton').pack(anchor='w', pady=2)
        
        # Visualization settings
        viz_settings_frame = ttk.Frame(settings_frame, style='Left.TFrame')
        viz_settings_frame.pack(fill='x', padx=10, pady=5)
        
        ttk.Label(viz_settings_frame, text="📊 Visualization Settings", 
                 style='TLabel', background='#252526').pack(anchor='w')
        
        self.animation_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(viz_settings_frame, text="Enable Animations", 
                       variable=self.animation_var, style='TCheckbutton',
                       command=self._toggle_animations).pack(anchor='w', pady=2)
        
        return settings_frame

    def _create_right_panel(self, parent):
        """Create right panel with code editor and output"""
        # Code editor section
        editor_frame = ttk.Frame(parent)
        self._create_editor_section(editor_frame)
        parent.add(editor_frame, weight=3)
        
        # Output section  
        output_frame = ttk.Frame(parent)
        self._create_output_section(output_frame)
        parent.add(output_frame, weight=2)

    def _create_editor_section(self, parent):
        """Create code editor section"""
        # Header
        header_frame = ttk.Frame(parent)
        header_frame.pack(fill='x', pady=(10, 5), padx=10)
        
        ttk.Label(header_frame, text="💻 Code Practice Area", 
                 style='Header.TLabel').pack(side='left')
        ttk.Label(header_frame, text="Type code manually for better learning!", 
                 style='TLabel', foreground='#ffc107').pack(side='right')
        
        # Code editor
        self.code_editor = ModernCodeEditor(parent)
        
        # Now create the code generation panel
        if self.ai_system.is_available and hasattr(self, 'codegen_placeholder'):
            codegen_panel = CodeGenerationPanel(
                self.codegen_placeholder, self.ai_system, self.code_editor
            )
            codegen_panel.pack(fill='both', expand=True)
        
        # Button toolbar
        button_bar = ttk.Frame(parent)
        button_bar.pack(fill='x', pady=5, padx=10)
        
        # Main action buttons
        self.run_button = ttk.Button(
            button_bar, text="▶ Run Code", command=self._run_code, 
            style='Run.TButton'
        )
        self.run_button.pack(side='left', padx=2)
        
        ttk.Button(button_bar, text="🧹 Clear", command=self.code_editor.clear, 
                  style='Clear.TButton').pack(side='left', padx=2)
        
        # Editor utilities
        ttk.Button(button_bar, text="↶ Undo", 
                  command=lambda: self.code_editor.text_widget.edit_undo()).pack(side='left', padx=2)
        ttk.Button(button_bar, text="↷ Redo", 
                  command=lambda: self.code_editor.text_widget.edit_redo()).pack(side='left', padx=2)
        
        # File operations
        ttk.Button(button_bar, text="💾 Save", command=self._save_code).pack(side='right', padx=2)
        ttk.Button(button_bar, text="📁 Load", command=self._load_code).pack(side='right', padx=2)
        
        self.code_editor.pack(fill='both', expand=True, pady=5, padx=10)

    def _create_output_section(self, parent):
        """Create output section"""
        self.output_frame = parent
        
        ttk.Label(self.output_frame, text="📤 Output & AI Feedback from Sandra", 
                 style='Header.TLabel').pack(pady=10)
        
        # Loading animation
        self.loading_label = ttk.Label(
            self.output_frame, text="", font=('Consolas', 14, 'bold'), 
            foreground="#00aaff", background='#1e1e1e'
        )
        
        # Output text
        self.output_text = scrolledtext.ScrolledText(
            self.output_frame, wrap=tk.WORD, font=('Consolas', 11), 
            bg='#282c34', fg='#f8f9fa', bd=0, relief='flat', 
            state='disabled', padx=10, pady=10
        )
        self.output_text.pack(fill='both', expand=True, pady=5, padx=10)
        
        # Configure text tags
        self._configure_output_tags()

    def _create_status_bar(self):
        """Create status bar"""
        status_frame = ttk.Frame(self.root, style='Left.TFrame', height=35)
        status_frame.pack(side='bottom', fill='x', padx=5, pady=(0, 5))
        status_frame.pack_propagate(False)
        
        self.status_label = ttk.Label(
            status_frame, text="🧠 ConvAI Innovations Ready.", 
            background='#252526', anchor='w', style='TLabel'
        )
        self.status_label.pack(side='left', padx=10, pady=5)
        
        # Progress indicator
        completion_pct = self.session_manager.progress.get_completion_percentage()
        self.progress_label = ttk.Label(
            status_frame, text=f"Progress: {completion_pct:.0f}% Complete", 
            background='#252526', anchor='e', 
            style='TLabel', foreground='#28a745'
        )
        self.progress_label.pack(side='right', padx=10, pady=5)

    def _configure_output_tags(self):
        """Configure output text tags"""
        tags = {
            'success': {'foreground': '#28a745', 'font': ('Consolas', 11, 'bold')},
            'error': {'foreground': '#dc3545', 'font': ('Consolas', 11, 'bold')},
            'ai_feedback': {'foreground': '#00aaff', 'font': ('Consolas', 12, 'italic')},
            'hint': {'foreground': '#ffc107', 'font': ('Consolas', 11, 'bold')},
            'info': {'foreground': '#17a2b8', 'font': ('Consolas', 11)},
            'session_msg': {'foreground': '#6f42c1', 'font': ('Consolas', 12, 'bold')}
        }
        
        for tag, config in tags.items():
            self.output_text.tag_config(tag, **config)

    def _configure_pane_sizes(self, main_pane):
        """Configure pane sizes after window display"""
        try:
            total_width = self.root.winfo_width()
            if total_width > 100:
                left_width = min(550, int(total_width * 0.35))
                main_pane.sashpos(0, left_width)
        except tk.TclError:
            pass

    # Event handlers
    def _on_session_change(self, selected_session):
        """Handle session change"""
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

    def _on_visualization_change(self, selected_viz):
        """Handle visualization change"""
        viz_mapping = {
            "Neural Network": "neural_network",
            "Backpropagation": "backpropagation", 
            "Activation Functions": "activation_function",
            "Self-Attention": "attention",
            "Tokenization": "tokenization",
            "Loss Functions": "loss_functions"
        }
        
        viz_type = viz_mapping.get(selected_viz, "neural_network")
        
        if viz_type == "activation_function":
            self.visualization_manager.show_visualization(viz_type, function="relu")
        elif viz_type == "tokenization":
            self.visualization_manager.show_visualization(
                viz_type, text="Hello world! This is tokenization.", method="bpe"
            )
        else:
            self.visualization_manager.show_visualization(viz_type)

    def _toggle_animations(self):
        """Toggle visualization animations"""
        self.visualization_manager.update_config(animate=self.animation_var.get())

    def _next_session(self):
        """Move to next session"""
        next_session_id = self.session_manager.get_next_session()
        if next_session_id:
            self.session_manager.mark_session_complete(
                self.session_manager.progress.current_session_id
            )
            self.session_manager.progress.current_session_id = next_session_id
            self._load_current_session()
            self._clear_output()
            self._show_initial_session_message()
            
            next_session = self.session_manager.get_session(next_session_id)
            self.status_label.config(text=f"🎉 Advanced to: {next_session.title}")
        else:
            messagebox.showinfo(
                "Congratulations!", 
                "🎉 You've completed all sessions! You're now ready to build your own LLMs!"
            )

    def _get_hint(self):
        """Get hint for current session"""
        current_session = self.session_manager.get_session(
            self.session_manager.progress.current_session_id
        )
        if current_session and current_session.hints:
            import random
            hint = random.choice(current_session.hints)
            self._log_output(f"\n💡 Sandra's Hint: {hint}", 'hint')

    def _copy_reference(self):
        """Copy reference code to clipboard"""
        content = self.reference_text.get('1.0', 'end-1c')
        if content.strip():
            self.root.clipboard_clear()
            self.root.clipboard_append(content)
            self.status_label.config(
                text="📋 Reference content copied! But try typing it manually for better learning."
            )
        else:
            messagebox.showinfo("No Content", "Reference code area is empty")

    def _run_code(self):
        """Run the code in the editor"""
        if self.is_loading:
            return
            
        code = self.code_editor.get_text()
        if not code.strip():
            messagebox.showwarning("Input Error", "Code editor is empty. Try typing some code first!")
            return
        
        self._set_ui_loading(True)
        self.tts_system.stop_speech()
        self._clear_output()
        self._start_loading_animation()
        
        # Run in background thread
        threading.Thread(target=self._execute_code, args=(code,), daemon=True).start()

    def _execute_code(self, code: str):
        """Execute code in background thread"""
        output_buffer = io.StringIO()
        error_buffer = io.StringIO()
        
        # Redirect stdout and stderr
        sys.stdout, sys.stderr = output_buffer, error_buffer
        error_msg = ""
        
        try:
            exec(code, {'__builtins__': __builtins__})
        except Exception as e:
            error_msg = f"{type(e).__name__}: {e}"
        finally:
            sys.stdout, sys.stderr = sys.__stdout__, sys.__stderr__
        
        # Process on main thread
        self.root.after(0, self._process_execution_result, code, 
                       output_buffer.getvalue(), error_msg)

    def _process_execution_result(self, code, output, error):
        """Process code execution results"""
        self._stop_loading_animation()
        
        # Get current language from the language selector (FIXED)
        current_language = self.language_selector.get_current_language()  # This was the issue!
        
        # Generate AI feedback with the selected language
        session_id = self.session_manager.progress.current_session_id
        feedback_text = self.ai_system.generate_feedback(code, error, session_id, current_language)
        
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
                self._speak_with_stop_button(feedback_text, current_language)
        
        self.status_label.config(text="🧠 ConvAI Innovations Ready.")
        self._set_ui_loading(False)

    def _save_code(self):
        """Save code to file"""
        from tkinter import filedialog
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
        """Load code from file"""
        from tkinter import filedialog
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

    # UI state management
    def _set_ui_loading(self, is_loading):
        """Set UI loading state"""
        self.is_loading = is_loading
        state = tk.DISABLED if is_loading else tk.NORMAL
        self.run_button.config(state=state)
        
        if not is_loading:
            self.run_button.config(text="▶ Run Code")

    def _start_loading_animation(self):
        """Start loading animation"""
        self.loading_label.pack(pady=20)
        self.output_text.pack_forget()
        self._animate_loading()

    def _stop_loading_animation(self):
        """Stop loading animation"""
        self.loading_label.pack_forget()
        self.output_text.pack(fill='both', expand=True, pady=5, padx=10)

    def _animate_loading(self):
        """Animate loading indicator"""
        if self.is_loading:
            char = self.animation_chars[self.animation_index]
            self.loading_label.config(text=f"Sandra is analyzing your code... {char}")
            self.animation_index = (self.animation_index + 1) % len(self.animation_chars)
            self.root.after(150, self._animate_loading)

    def _speak_with_stop_button(self, text, language=None):
        """Speak text and manage stop button"""
        if not self.tts_system.is_available:
            self.audio_status_label.config(text="🔇 Audio: Not Available", foreground='#dc3545')
            return
            
        self.stop_audio_button.config(state=tk.NORMAL)
        self.audio_status_label.config(text="🔊 Audio: Speaking...", foreground='#ffc107')
        self.tts_system.speak(text, language)
        
        def check_status():
            if not self.tts_system.is_speaking:
                self.stop_audio_button.config(state=tk.NORMAL)  # Keep enabled for interrupting
                self.audio_status_label.config(text="🔇 Audio: Ready", foreground='#28a745')
            else:
                self.root.after(100, check_status)
        
        self.root.after(100, check_status)

    # Session management
    def _load_current_session(self):
        """Load current session content"""
        current_session = self.session_manager.get_session(
            self.session_manager.progress.current_session_id
        )
        if not current_session:
            return
        
        # Update session display
        self.current_session_label.config(text=f"Current: {current_session.title}")
        
        # Load reference code - temporarily enable editing to load content
        was_edit_mode = self.edit_mode_var.get()
        self.reference_text.config(state='normal')
        self.reference_text.delete('1.0', 'end')
        
        full_content = (
            current_session.description + "\n\n" + "="*60 + 
            "\nREFERENCE CODE TO TYPE:\n" + "="*60 + "\n\n" + 
            current_session.reference_code
        )
        self.reference_text.insert('1.0', full_content)
        
        # Restore edit mode state
        if not was_edit_mode:
            self.reference_text.config(state='disabled')
        
        # Clear editor
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
        
        # Update visualization if available
        if current_session.visualization_type:
            self.visualization_manager.show_visualization(current_session.visualization_type)
        
        # Update progress
        self._update_progress_display()

    def _update_progress_display(self):
        """Update progress display"""
        completion_pct = self.session_manager.progress.get_completion_percentage()
        completed_count = len(self.session_manager.progress.completed_sessions)
        total_count = self.session_manager.progress.total_sessions
        
        self.progress_label.config(
            text=f"Progress: {completed_count}/{total_count} sessions ({completion_pct:.0f}%)"
        )

    def _show_initial_session_message(self):
        """Show initial session message"""
        current_language = self.language_selector.get_current_language()
        
        if self.ai_system.is_available:
            initial_msg = self.ai_system.initial_session_message(
                self.session_manager.progress.current_session_id, current_language
            )
            self._log_output(f"🤖 Sandra: {initial_msg}", 'session_msg')
        else:
            messages = {
                Language.ENGLISH: "👋 Welcome! Start by typing the reference code from the left panel to practice.",
                Language.SPANISH: "👋 ¡Bienvenido! Comienza escribiendo el código de referencia del panel izquierdo para practicar.",
                Language.FRENCH: "👋 Bienvenue ! Commencez par taper le code de référence du panneau de gauche pour vous entraîner.",
                Language.HINDI: "👋 स्वागत! अभ्यास के लिए बाएं पैनल से संदर्भ कोड टाइप करना शुरू करें।",
                Language.ITALIAN: "👋 Benvenuto! Inizia digitando il codice di riferimento dal pannello di sinistra per esercitarti.",
                Language.PORTUGUESE: "👋 Bem-vindo! Comece digitando o código de referência do painel esquerdo para praticar."
            }
            message = messages.get(current_language, messages[Language.ENGLISH])
            self._log_output(message, 'session_msg')

    def _clear_output(self):
        """Clear output area"""
        self.output_text.config(state='normal')
        self.output_text.delete('1.0', 'end')
        self.output_text.config(state='disabled')

    def _log_output(self, message, tag=None):
        """Log message to output"""
        self.output_text.config(state='normal')
        self.output_text.insert('end', message + '\n', tag)
        self.output_text.config(state='disabled')
        self.output_text.see('end')

    def on_closing(self):
        """Handle application closing"""
        if messagebox.askokcancel("Quit", "Do you want to exit ConvAI Innovations?"):
            self.tts_system.stop_speech()
            self.executor.shutdown(wait=False)
            self.root.destroy()


def main():
    """Main application entry point"""
    # Check dependencies
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch
        TRANSFORMERS_AVAILABLE = True
    except ImportError:
        TRANSFORMERS_AVAILABLE = False
        messagebox.showerror(
            "Dependency Error", 
            "Transformers library not found. Please install:\npip install transformers torch"
        )
        return
        
    try:
        from kokoro.pipeline import KPipeline
        import torch
        import sounddevice as sd
        KOKORO_TTS_AVAILABLE = True
    except ImportError:
        KOKORO_TTS_AVAILABLE = False
        messagebox.showwarning(
            "Audio Warning", 
            "Kokoro TTS not available. Audio features will be limited.\n"
            "To enable full audio: pip install kokoro-tts torch sounddevice"
        )

    # Initialize main window
    root = tk.Tk()
    root.withdraw()  # Hide until setup complete

    def on_setup_complete(model_path, ai_system=None):
        # Model path can be None when using default Hugging Face models
        root.deiconify()
        app = SessionBasedLLMLearningDashboard(root, model_path, ai_system)

    # Start model download and setup
    downloader = ModelDownloader(on_complete=on_setup_complete)
    downloader.run()
    
    # Start main event loop
    root.mainloop()


if __name__ == "__main__":
    main()