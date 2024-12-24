import tkinter as tk
import customtkinter as ctk
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
import matplotlib as mpl
import matplotlib.cm as cmap
import matplotlib.colors as mpl_colors
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
# import multiprocessing as mp
import time
import joblib

import config

from copy import deepcopy
from PIL import Image
from sklearn.metrics import confusion_matrix, accuracy_score
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from tkinter.messagebox import showerror
from tkinter import filedialog
from sklearn.svm import LinearSVC
from sklearn.model_selection import learning_curve
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression

class TrainFrame(ctk.CTkFrame):
    def __init__(self, container):
        # Setting appearance
        self.container = container
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("dark-blue")
        super().__init__(container, corner_radius=0)
        self.image_path = config.IMAGE_PATH
        

        self.button_setting = {
            'corner_radius': 10, 
            'compound': 'left',  
            'anchor': 'w', 
            'border_spacing': 10, 
            'height': 20,
            'text_color': 'gray90', 
            'hover_color': 'gray30', 
            'fg_color': '#3d85c6',
            'font': ctk.CTkFont(size=18, weight="bold")
        }
        def hex_to_rgb(h):
            h = h.lstrip('#')
            return tuple(int(h[i:i+2], 16)/255 for i in (0, 2, 4))

        palette = ['#b4d2b1', '#568f8b', '#1d4a60', '#cd7e59', '#ddb247', '#d15252']
        palette_rgb = [hex_to_rgb(x) for x in palette]
        cmap = mpl_colors.ListedColormap(palette_rgb)
        colors = cmap.colors
        bg_color= '#fdfcf6'

        custom_params = {
            "axes.spines.right": False,
            "axes.spines.top": False,
            'grid.alpha':0.3,
            # 'figure.figsize': (16, 6),
            'axes.titlesize': 'Large',
            'axes.labelsize': 'Large',
            'figure.facecolor': bg_color,
            'axes.facecolor': bg_color
        }

        sns.set_theme(
            style='whitegrid',
            palette=sns.color_palette(palette),
            rc=custom_params
        )

        # Add 2x2 grid
        self.grid_columnconfigure(0, weight=4)
        self.grid_columnconfigure(1, weight=4)
        self.grid_rowconfigure(0, weight=4)
        self.grid_rowconfigure(1, weight=2)

        # Add param frame
        self.param_frame = ctk.CTkFrame(self,
                                        border_width=3,     # Độ dày đường viền
                                        border_color="#5A5A5A"    # Màu viền trắng
                                        )
        self.param_frame.grid(row=0, column=0, sticky='nsew', padx=10)
        # Adjusting the layout for better balance and alignment
        # Update grid configuration and frame dimensions
        self.param_frame.grid_rowconfigure(0, weight=1)  # Adjust overall row height
        self.param_frame.grid_columnconfigure((0, 1, 2, 3), weight=1)  # Set equal width for all columns

        # KNN Frame
        # Tạo frame cho KNN
        self.knn_frame = ctk.CTkFrame(
            self.param_frame,
            border_width=1,
            border_color="#CB4343",
            corner_radius=10
        )
        self.knn_frame.grid(row=0, column=0, sticky='nsew', padx=10, pady=10)

        # Cấu hình lưới (grid)
        self.knn_frame.grid_rowconfigure((0, 1, 2, 3), weight=1)  # Các hàng có chiều cao cân bằng
        self.knn_frame.grid_columnconfigure(0, weight=1)  # Cột đầu tiên co giãn theo chiều ngang

        # Tiêu đề KNN
        self.knn_label = ctk.CTkLabel(
            self.knn_frame, 
            text="KNN",
            font=ctk.CTkFont(size=18, weight="bold")
        )
        self.knn_label.grid(row=0, column=0, padx=10, pady=5, sticky="ns")  # Gắn nhãn phía trên, căn giữa

        # Nhập số n_neighbors
        self.k_choice_entry = ctk.CTkEntry(
            self.knn_frame, 
            placeholder_text='n_neighbors'
        )
        self.k_choice_entry.grid(row=1, column=0, padx=10, pady=5, sticky="ew")  # Co giãn ngang

        # Menu chọn Algorithm
        self.algo_menu = ctk.CTkComboBox(
            self.knn_frame, 
            values=['auto', 'ball_tree', 'kd_tree', 'brute'],
            state='readonly'
        )
        self.algo_menu.set('Algorithm')
        self.algo_menu.grid(row=2, column=0, padx=10, pady=5, sticky="ew")  # Co giãn ngang

        # Menu chọn Weights
        self.metrics_menu = ctk.CTkComboBox(
            self.knn_frame, 
            values=['uniform', 'distance'],
            state='readonly'
        )
        self.metrics_menu.set('Weights')
        self.metrics_menu.grid(row=3, column=0, padx=10, pady=5, sticky="ew")  # Co giãn ngang


        # SVM Frame
        self.svm_frame = ctk.CTkFrame(
            self.param_frame,
            border_width=1,
            border_color="#CB4343",
            corner_radius=10
        )
        self.svm_frame.grid(row=0, column=1, sticky='nsew', padx=10, pady=10)

        # Configure grid
        self.svm_frame.grid_rowconfigure((0, 1, 2, 3), weight=1)  # Equal row height
        self.svm_frame.grid_columnconfigure(0, weight=1)  # Column 0 stretches horizontally

        # Title SVM
        self.svm_label = ctk.CTkLabel(
            self.svm_frame, 
            text="SVM",
            font=ctk.CTkFont(size=18, weight="bold")
        )
        self.svm_label.grid(row=0, column=0, padx=10, pady=5, sticky="ns")  # Top-centered label

        # Input for C
        self.svm_choice_1 = ctk.CTkEntry(
            self.svm_frame, 
            placeholder_text='C'
        )
        self.svm_choice_1.grid(row=1, column=0, padx=10, pady=5, sticky="ew")  # Horizontally expandable

        # Menu to choose Gamma
        self.svm_gamma_menu = ctk.CTkComboBox(
            self.svm_frame, 
            values=['scale', 'auto'],
            state='readonly'
        )
        self.svm_gamma_menu.set('Gamma')
        self.svm_gamma_menu.grid(row=2, column=0, padx=10, pady=5, sticky="ew")  # Horizontally expandable

        # Menu to choose Kernel
        self.svm_kernel_menu = ctk.CTkComboBox(
            self.svm_frame, 
            values=['linear', 'poly', 'rbf', 'sigmoid'],
            state='readonly'
        )
        self.svm_kernel_menu.set('Kernel')
        self.svm_kernel_menu.grid(row=3, column=0, padx=10, pady=5, sticky="ew")  # Horizontally expandable

        # RF Frame
        self.rf_frame = ctk.CTkFrame(
            self.param_frame,
            border_width=1,
            border_color="#CB4343",
            corner_radius=10
        )
        self.rf_frame.grid(row=0, column=2, sticky='nsew', padx=10, pady=10)

        # Configure grid
        self.rf_frame.grid_rowconfigure((0, 1, 2, 3, 4), weight=1)  # Equal row height
        self.rf_frame.grid_columnconfigure(0, weight=1)  # Column 0 stretches horizontally

        # Title RF
        self.rf_label = ctk.CTkLabel(
            self.rf_frame, 
            text="RF",
            font=ctk.CTkFont(size=18, weight="bold")
        )
        self.rf_label.grid(row=0, column=0, padx=10, pady=5, sticky="ns")  # Top-centered label

        # Input for max_depth
        self.rf_max_depth = ctk.CTkEntry(
            self.rf_frame, 
            placeholder_text='max_depth'
        )
        self.rf_max_depth.grid(row=1, column=0, padx=10, pady=5, sticky="ew")  # Horizontally expandable

        # Input for min_samples_split
        self.rf_min_samples_split = ctk.CTkEntry(
            self.rf_frame, 
            placeholder_text='min_samples_split'
        )
        self.rf_min_samples_split.grid(row=2, column=0, padx=10, pady=5, sticky="ew")  # Horizontally expandable

        # Input for min_samples_leaf
        self.rf_min_samples_leaf = ctk.CTkEntry(
            self.rf_frame, 
            placeholder_text='min_samples_leaf'
        )
        self.rf_min_samples_leaf.grid(row=3, column=0, padx=10, pady=5, sticky="ew")  # Horizontally expandable

        # Input for n_estimators
        self.rf_n_estimators = ctk.CTkEntry(
            self.rf_frame, 
            placeholder_text='n_estimators'
        )
        self.rf_n_estimators.grid(row=4, column=0, padx=10, pady=5, sticky="ew")  # Horizontally expandable

        # LR Frame
        self.lr_frame = ctk.CTkFrame(
            self.param_frame,
            border_width=1,
            border_color="#CB4343",
            corner_radius=10
        )
        self.lr_frame.grid(row=0, column=3, sticky='nsew', padx=10, pady=10)

        # Configure grid
        self.lr_frame.grid_rowconfigure((0, 1, 2), weight=1)  # Equal row height
        self.lr_frame.grid_columnconfigure(0, weight=1)  # Column 0 stretches horizontally

        # Title LR
        self.lr_label = ctk.CTkLabel(
            self.lr_frame, 
            text="LR",
            font=ctk.CTkFont(size=18, weight="bold")
        )
        self.lr_label.grid(row=0, column=0, padx=10, pady=5, sticky="ns")  # Top-centered label

        # Menu to choose Solver
        self.lr_solver_menu = ctk.CTkComboBox(
            self.lr_frame, 
            values=['lbfgs', 'newton-cg', 'newton-cholesky', 'sag', 'saga'],
            state='readonly'
        )
        self.lr_solver_menu.set('Solver')
        self.lr_solver_menu.grid(row=1, column=0, padx=10, pady=5, sticky="ew")  # Horizontally expandable

        # Input for C
        self.lr_c = ctk.CTkEntry(
            self.lr_frame, 
            placeholder_text='C'
        )
        self.lr_c.grid(row=2, column=0, padx=10, pady=5, sticky="ew")  # Horizontally expandable

        # Add button frame and buttons
        self.button_frame = ctk.CTkFrame(self,
                                        border_width=1,     # Độ dày đường viền
                                        border_color="white"    # Màu viền trắng
                                        )
        self.button_frame.grid(row=0, column=1, sticky='nsew')
        self.button_frame.grid_rowconfigure(0, weight=1)
        self.button_frame.grid_rowconfigure(1, weight=1)
        self.button_frame.grid_rowconfigure(2, weight=1)
        self.button_frame.grid_rowconfigure(3, weight=1)
        self.button_frame.grid_rowconfigure(4, weight=1)

        # Add train button
        self.model_image_icon = ctk.CTkImage(Image.open(self.image_path + 'ai-icon.png'), size=(25, 25))
        self.train_button = ctk.CTkButton(self.button_frame, image=self.model_image_icon, text='Train', 
                                            **self.button_setting, 
                                            #command=self.train_button_event
                                            )
        self.train_button.grid(row=0, column=0, padx=10, pady=10, sticky='ew')

        # Add learning curve
        self.paint_icon = ctk.CTkImage(Image.open(self.image_path + 'paint.png'), size=(25, 25))
        self.draw_button = ctk.CTkButton(self.button_frame, image=self.paint_icon,
                                         text='Learning curve', 
                                         #command=self.draw_button_event, 
                                         **self.button_setting)
        self.draw_button.grid(row=1, column=0, padx=10, pady=10, sticky='ew')


        # Add save button
        self.save_icon = ctk.CTkImage(Image.open(self.image_path + 'save.png'), size=(25, 25))
        self.save_button = ctk.CTkButton(self.button_frame, image=self.save_icon,
                                         text='Save', 
                                         #command=self.save_button_event, 
                                         **self.button_setting)
        self.save_button.grid(row=2, column=0, padx=10, pady=10, sticky='ew')
        
        # Add choose model
        self.model_menu = ctk.CTkComboBox(self.button_frame, height=30, 
                                          values=[name for name in config.MODEL_LIST], 
                                          state='readonly'
                                          #command=self.model_menu_event
                                          )
        self.model_menu.set('Choose Current Model')
        self.model_menu.grid(row=3, column=0, padx=10, pady=10, sticky='ew')
        
        # Add model name
        self.name_entry = ctk.CTkEntry(self.button_frame, corner_radius=10, placeholder_text='Enter the model name', height=30)
        self.name_entry.grid(row=4, column=0, padx=10, pady=10, sticky='ew')

        # Add bottom frame
        self.bottom_frame = ctk.CTkFrame(self)
        self.bottom_frame.grid(row=1, column=0, sticky='nsew')
        self.bottom_frame.grid_columnconfigure(0, weight=1)
        self.bottom_frame.grid_columnconfigure(1, weight=1)

        # Add Confusion matrix
        self.figure_confusion_mat, self.ax_confusion_mat = plt.subplots()
        self.canvas_confusion_mat = FigureCanvasTkAgg(self.figure_confusion_mat, self.bottom_frame)
        self.canvas_confusion_mat.draw()
        self.canvas_confusion_mat.get_tk_widget().grid(row=0, column=0, sticky='nsew', padx=10, pady=10)

        # Add learning curve
        self.figure_learning_curve, self.ax_learning_curve = plt.subplots()
        self.canvas_learning_curve = FigureCanvasTkAgg(self.figure_learning_curve, self.bottom_frame)
        self.canvas_learning_curve.draw()
        self.canvas_learning_curve.get_tk_widget().grid(row=0, column=1, sticky='nsew', padx=10, pady=10)

        # Add result box
        self.result_box = ctk.CTkTextbox(
            self, 
            font=ctk.CTkFont(size=15, weight="bold"),
            border_width=3,        # Độ dày viền
            border_color="white"   # Màu viền trắng
        )
        self.result_box.insert(tk.INSERT, 'RESULT BOX\n')
        self.result_box.grid(row=1, column=1, sticky='nsew', padx=10, pady=10)
        self.result_box.configure(state="disabled")

        # Add model
        self.model_type = None
        self.model = None

        # Add progress bar
        self.progress_bar = ctk.CTkProgressBar(self.param_frame, mode='indeterminnate')
        self.progress_bar.grid(row=1, column=0, columnspan=3, padx=10, pady=10, sticky='ew')

        # self.queue = mp.Queue()

