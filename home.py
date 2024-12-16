import tkinter as tk
import customtkinter as ctk

import config

from tkVideoPlayer import TkinterVideo

class HomeFrame(ctk.CTkFrame):
    def __init__(self, container):
        #Color and appearance
        self.container = container
        ctk.set_appearance_mode('dark')
        ctk.set_default_color_theme('green')
        self.config_path = config.IMAGE_PATH
        
        #Play video
        super().__init__(master=container,corner_radius=0,border_width=100, border_color='#4AE2C6')
        self.video = TkinterVideo(master=self, scaled=True)
        self.video.load(self.config_path + r'\welcome.mp4')
        self.video.pack(expand=True)
        self.video.place(relx=0.5, rely=0.5, relheight=0.98, relwidth=0.98, anchor='center')
        self.video.play()
        self.video.bind('<<Ended>>', self.loop_video)
        
    def loop_video(self, event):
        self.video.play()
