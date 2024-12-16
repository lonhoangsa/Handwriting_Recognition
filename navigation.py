import tkinter as tk
import customtkinter as ctk
from PIL import Image

import config
from exit import ExitFrame

class NavigationFrame(ctk.CTkFrame):
    def __init__ (self, container):
        #Color and appearance
        self.container = container
        ctk.set_appearance_mode('dark')
        ctk.set_default_color_theme('green')
        
        #Basic setup
        super().__init__(master=container,corner_radius=0)
        self.image_path = config.IMAGE_PATH
        
        #Setting up the images
        self.menu_image = ctk.CTkImage(Image.open(self.image_path + r'\menu_icon.png'), size=(30,30))
        self.home_image = ctk.CTkImage(Image.open(self.image_path + r'\home.png'), size=(20,20))
        self.draw_image = ctk.CTkImage(Image.open(self.image_path + r'\draw.png'), size=(20,20))
        self.train_image = ctk.CTkImage(Image.open(self.image_path + r'\train.png'), size=(20,20))
        self.predict_image = ctk.CTkImage(Image.open(self.image_path + r'\predict.png'), size=(20,20))
        self.hand_rec_image = ctk.CTkImage(Image.open(self.image_path + r'\hand_rec.png'), size=(20,20))
        self.exit_image = ctk.CTkImage(Image.open(self.image_path + r'\exit.png'), size=(20,20))
        
        #Setting up the grid
        self.grid_columnconfigure(index=0, weight=1)
        
        #Setting up the label/button universal settings
        self.button_setting = {
            'corner_radius': 15,
            'height': 90,
            'border_spacing': 10,
            'fg_color': 'transparent',
            'text_color': 'gray90',
            'hover_color': "#FFA07A",
            'anchor': 'center',
            'font': ctk.CTkFont(size=17, weight="bold")
        }
        
        #Setting up the menu label
        self.menu_label = ctk.CTkLabel(master=self, text = '   MENU',font=("Ariel", 20, "bold"), 
                                       image=self.menu_image, height = 90, compound='left',
                                       fg_color="#4A90E2", corner_radius=15)
        self.menu_label.grid(row=0,column=0,padx=20,pady=20,sticky='nsew')
        
        #Setting up the Home button
        self.home_button = ctk.CTkButton(master=self, text='Home', 
                                         image=self.home_image, 
                                         command=self.home,
                                         **self.button_setting)
        self.home_button.grid(row=1,column=0,sticky='nsew')
        
        #Setting up the Draw button
        self.draw_button = ctk.CTkButton(master=self, text='Draw',
                                         image=self.draw_image, command=self.draw, 
                                         **self.button_setting)
        self.draw_button.grid(row=2,column=0,sticky='nsew')
        
        #Setting up the Train button
        self.train_button = ctk.CTkButton(master=self, text='Train',
                                         image=self.train_image, 
                                         command=self.train,
                                         **self.button_setting)
        self.train_button.grid(row=3,column=0,sticky='nsew')
        
        #Setting up the Predict button
        self.predict_button = ctk.CTkButton(master=self, text='Predict',
                                         image=self.predict_image, 
                                         command=self.predict,
                                         **self.button_setting)
        self.predict_button.grid(row=4,column=0,sticky='nsew')
        
        #Setting up the Handwriting Recognition button
        self.hand_rec_button = ctk.CTkButton(master=self, text='Handwriting Recognition',
                                         image=self.hand_rec_image, 
                                         command=self.handwriting_recognition, 
                                         **self.button_setting)
        self.hand_rec_button.grid(row=5,column=0,sticky='nsew')
        
        #Setting up the Exit button
        self.exit_button = ctk.CTkButton(master=self, text='Exit',
                                         image=self.exit_image, 
                                         command=self.exit, 
                                         **self.button_setting)
        self.exit_button.grid(row=6,column=0,sticky='nsew')
        
        #Setting up the version label
        self.version_label = ctk.CTkLabel(master=self, text='Version 1.0', 
                                          font=("Ariel", 40, "bold"), compound="center", 
                                          anchor='center',bg_color='gray')
        self.version_label.grid(row=7,column=0,pady=0,sticky='nsew')
    
    #Function to close every frames and open only the frame needed
    def select_frame_by_name(self, name):
        #Closing all frames
        self.container.home_frame.grid_forget()
        self.container.draw_main_frame.grid_forget()
        self.container.draw_start_frame.grid_forget()
        self.container.predict_frame.grid_forget()
        self.container.predict_live_frame.grid_forget()
        self.container.predict_batch_frame.grid_forget()
        self.container.train_frame.grid_forget()
        self.container.handwriting_recognition_frame.grid_forget()
        self.container.exit_frame.grid_forget()
        
        #Checking the name of the frame to open
        if name == 'home':
            self.container.home_frame.grid(row=0,column=1,sticky='nsew')
        
        if name == 'draw':
            self.container.draw_start_frame.grid(row=0,column=1,sticky='nsew')
            
        if name == 'predict':
            self.container.predict_frame.grid(row=0,column=1,sticky='nsew')
        
        if name == 'train':
            self.container.train_frame.grid(row=0,column=1,sticky='nsew')
            
        if name == 'handwriting_recognition':
            self.container.handwriting_recognition_frame.grid(row=0,column=1,sticky='nsew')
        
        if name == 'exit':
            self.container.exit_frame.grid(row=0,column=1,sticky='nsew')
    
    def home(self):
        self.select_frame_by_name('home')
    
    def draw(self):
        self.select_frame_by_name('draw')
        
    def train(self):
        self.select_frame_by_name('train')
    
    def predict(self):
        self.select_frame_by_name('predict')
        
    def handwriting_recognition(self):
        self.select_frame_by_name('handwriting_recognition')
        
    def handwriting_recognition(self):
        self.select_frame_by_name('handwriting_recognition')
    
    def exit(self):
        self.select_frame_by_name('exit')