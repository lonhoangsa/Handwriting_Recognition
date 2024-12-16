import tkinter as tk
import customtkinter as ctk

import config

class ExitFrame(ctk.CTkFrame):
    def __init__(self, container):
        #Color and appearance
        self.container = container
        ctk.set_appearance_mode('dark')
        ctk.set_default_color_theme('dark-blue')
        self.config_path = config.IMAGE_PATH
        super().__init__(master=container,corner_radius=0, bg_color='transparent')
                
        #Setting up the grid
        self.grid_rowconfigure(index=0,weight=1)
        self.grid_rowconfigure(index=1,weight=1)
        self.grid_columnconfigure(index=0,weight=1)
        
        #Setting up the question label
        self.question_label = ctk.CTkLabel(master=self, text='Are you sure you want to exit?', 
                                           font=("Ariel", 30, "bold"), compound="bottom", 
                                           corner_radius=0)
        self.question_label.grid(row=0, column = 0, pady=10, sticky='s')
        
        #Setting up the yes and no buttons (with universal settings)
        self.button_setting = {
            'corner_radius': 15,
            'height': 40,
            'border_spacing': 10,
            'fg_color': '#D7EBE7',
            'text_color': 'black',
            'hover_color': "#FFA07A",
            'anchor': 'center',
            'font': ctk.CTkFont(size=17, weight="bold")
        }
        
        self.button_frame = ctk.CTkFrame(master=self, corner_radius=0, fg_color='transparent')
        self.button_frame.grid(row=1, column=0, sticky='nsew')
        
        self.button_frame.grid_columnconfigure(index=0, weight=1)
        self.button_frame.grid_columnconfigure(index=1, weight=1)
        
        self.yes_button = ctk.CTkButton(master=self.button_frame, text='Yes', **self.button_setting,
                                        command=self.container.destroy)
        self.yes_button.grid(row=0,column=0,padx=10,pady=10,sticky='e')
        
        self.no_button = ctk.CTkButton(master=self.button_frame, text='No' ,**self.button_setting,
                                       command=self.close)
        self.no_button.grid(row=0,column=1,padx=10,pady=10,sticky='w')
    
    def close(self):
        self.grid_forget()
        self.container.home_frame.grid(row=0,column=1,sticky='nsew')
        