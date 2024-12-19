import string
import tkinter as tk
from tkinter.messagebox import showerror

import customtkinter as ctk
from PIL import Image, ImageTk
from tkinter import messagebox

import config
from models.utils import load_model, preprocess_image, predict_characters
from utils import process_image


class HandwritingRecognitionFrame(ctk.CTkFrame):
    def __init__(self, container):
        #Color and appearance
        self.uploaded_image = None
        self.filename = None
        self.container = container
        ctk.set_appearance_mode('dark')
        ctk.set_default_color_theme('green')
        
        #Basic setup
        super().__init__(master=container,corner_radius=0, fg_color='transparent')
        self.icon_image_path = config.IMAGE_PATH
        
        #Setting up the images
        self.hand_rec_image = ctk.CTkImage(Image.open(self.icon_image_path + r'\hand_predict.png'), size=(20,20))
        self.hand_predict_button = ctk.CTkImage(Image.open(self.icon_image_path + r'\hand_predict_button.png'), size=(20,20))
        self.reset_button = ctk.CTkImage(Image.open(self.icon_image_path + r'\reset.png'), size=(20,20))
        
        #Setting up the grid
        self.grid_rowconfigure(index=0, weight=2)
        self.grid_rowconfigure(index=1, weight=1)
        self.grid_columnconfigure(index=0, weight=1)
        
        
        #Setting up the button universal settings
        self.button_setting = {
            'corner_radius': 15,
            'height': 40,
            'border_spacing': 10,
            'fg_color': '#4A90E2',
            'text_color': 'black',
            'hover_color': "#FFA07A",
            'anchor': 'center',
            'compound': 'left',
            'font': ctk.CTkFont(size=17, weight="bold")
        }
        
        #Setting up the first row
        self.first_row = ctk.CTkFrame(master=self, corner_radius=0)
        self.first_row.grid(row=0,column=0,sticky='nsew')
        self.first_row.grid_rowconfigure(index=0, weight=1)
        self.first_row.grid_columnconfigure(index=0, weight=1)
        
        #Setting up the image upload frame
        self.image_upload_frame = ctk.CTkFrame(master=self.first_row, corner_radius=0, 
                                               border_width=5, fg_color='black',)
        self.image_upload_frame.grid_rowconfigure(index=0, weight=1)
        self.image_upload_frame.grid_columnconfigure(index=0, weight=1)
        self.image_upload_frame.grid(row=0,column=0,sticky='nsew',padx=10,pady=10)
        
        self.img_upload_button = ctk.CTkButton(master=self.image_upload_frame, text='Upload Image',
                                               command=self.on_upload_image,
                                               **self.button_setting)
        self.img_upload_button.grid(row=0,column=0)
        
        #Setting up the canvas frame
        self.canvas_frame = ctk.CTkFrame(master=self.first_row, corner_radius=0, border_width=10)
        self.canvas_frame.grid_rowconfigure(index=0, weight=1)
        self.canvas_frame.grid_columnconfigure(index=0, weight=1)
        self.canvas_label = ctk.CTkLabel(master=self.canvas_frame, text='', width=600, height=400)
        self.canvas_label.grid(row=0,column=0,sticky='nsew')
        
        #Setting up the navigation frame
        self.prediction_frame = ctk.CTkFrame(master=self.first_row, corner_radius=10)
        
        self.prediction_frame.grid_rowconfigure(index=0, weight=1)
        self.prediction_frame.grid_rowconfigure(index=1, weight=1)
        self.prediction_frame.grid_rowconfigure(index=2, weight=1)
        self.prediction_frame.grid_rowconfigure(index=3, weight=1)
        self.prediction_frame.grid_columnconfigure(index=0, weight=1)
        
        self.prediction_frame.grid(row=0,column=1,sticky='nsew', padx=10,pady=10)
        
            #-------------------------------- Prediction Label
        self.hand_predict_label = ctk.CTkLabel(master=self.prediction_frame, text='Handwriting Prediction',
                                               image=self.hand_rec_image, font=('Ariel', 13, 'bold'), 
                                               fg_color="#4A90E2", corner_radius=15, pady=5,
                                               compound='top')
        self.hand_predict_label.grid(row=0,column=0,pady=30,padx=10,sticky='n')
        
            #-------------------------------- Reset Button
        self.hand_reset_button = ctk.CTkButton(master=self.prediction_frame, text='Reset',
                                               image=self.reset_button,
                                               command=self.on_reset,
                                               **self.button_setting)
        self.hand_reset_button.grid(row=1,column=0,sticky='n')
        
            #-------------------------------- Prediction Button
        self.hand_predict_button = ctk.CTkButton(master=self.prediction_frame, text='Predict',
                                                 image=self.hand_predict_button, command=self.predict_button_event,
                                                **self.button_setting)
        self.hand_predict_button.grid(row=2,column=0,sticky='n')
        
            #-------------------------------- Dropdown
        self.selected_option = ctk.StringVar()
        self.selected_option.set('Select an algorithm')
        self.options = config.MODEL_LIST
        self.dropdown = ctk.CTkOptionMenu(master=self.prediction_frame,
                                          variable=self.selected_option,
                                          values=self.options,command=self.model_menu_event)
        self.dropdown.grid(row=3,column=0,sticky='n')
        
        #Setting up the second row
        self.second_row = ctk.CTkFrame(master=self, corner_radius=0)
        self.second_row.grid(row=1,column=0,sticky='nsew')
        self.second_row.grid_rowconfigure(index=0, weight=1)
        self.second_row.grid_columnconfigure(index=0, weight=1)
        
        self.message_box = ctk.CTkTextbox(master=self.second_row, corner_radius=15, border_width=0,
                                         font=('Ariel', 15, 'bold'),
                                         fg_color='#767676')
        self.message_box.insert(tk.INSERT, 'LOG BOX\n')
        self.message_box.configure(state='disabled')
        self.message_box.grid(row=0,column=0,sticky='nsew', padx=10, pady=10)
        # Model backend
        self.model_type = None
        self.model = None
        self.model_name = ''

    #Function to upload the image
    def on_upload_image(self):
        self.filename = ctk.filedialog.askopenfilename(initialdir='/', title='Select an Image',
                                                       filetypes=[('png files','*.png'),
                                                                  ('jpg files','*.jpg'),
                                                                  ('jpeg files','*.jpeg'),
                                                                  ('all files','*.*')])
        if self.filename == '':
            messagebox.showerror('Error','No file selected')
        self.uploaded_image = ctk.CTkImage(Image.open(self.filename),size= (600,400))
        self.canvas_label.configure(image=self.uploaded_image)
        self.image_path = self.filename
        self.image_upload_frame.grid_forget()
        self.canvas_frame.grid(row=0,column=0)
    
    #Function to reset the frame
    def on_reset(self):
        self.canvas_frame.grid_forget()
        self.image_upload_frame.grid(row=0,column=0,sticky='nsew')

    def model_menu_event(self, choice):
        self.model_type = choice

        # Open models in the dir
        result = " "
        if len(result) == 0:
            showerror(message="Please choose a models", title='Error')
        else:
            # Load the models
            self.model = load_model(model_name=str(self.model_type).lower())

    def predict_button_event(self):
        if self.model is None:
            showerror(message="Please choose a models first", title='Error')
        else:

            img_array = process_image(self.image_path)

            if self.model_type != 'CNN':
                characters = preprocess_image(self.image_path)
                predictions = predict_characters(characters, self.model)
                predictions = [string.ascii_lowercase[label - 1] for label in predictions]
                word = ''.join(map(str, predictions))
            else:
                word = self.model.predict(x=img_array.reshape((1, 28, 28, 1))).argmax()

            # Print out the result
            self.message_box.configure(state="normal")
            self.message_box.insert(tk.END, f'\n- Your image predicted label is: {word}')
            self.message_box.configure(state="disabled")