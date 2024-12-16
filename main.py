import tkinter as tk
import customtkinter as ctk

import config
from navigation import NavigationFrame
from home import HomeFrame
from exit import ExitFrame
from handwriting_recognition import HandwritingRecognitionFrame
from draw import DrawMainFrame, DrawStartFrame  
from predict import PredictStartFrame, PredictLivePredictionFrame, PredictBatchPredictionFrame
from train import TrainFrame

class MainApp(ctk.CTk):
    #Mainframe
    def __init__ (self):
        #Color and appearance
        ctk.set_appearance_mode('dark')
        ctk.set_default_color_theme('green')
        
        #Basic setup
        super().__init__()
        self.image_path = config.IMAGE_PATH
        
        #Window attributes
        self.geometry('1080x720')
        self.title('Handwriting Recognition')
        
        #Setting icon
        self.iconbitmap(self.image_path + r'\icon.ico')
        
        #Setting up the grid
        self.grid_columnconfigure(index=0, weight=0)
        self.grid_columnconfigure(index=1, weight=1)
        self.grid_rowconfigure(index=0,weight=1)
        
        #Home frame
        self.home_frame = HomeFrame(self)
        self.home_frame.grid(row=0,column=1,sticky='nsew')
        
        #Draw frame
        self.draw_start_frame = DrawStartFrame(self)
        self.draw_main_frame = DrawMainFrame(self)
        
        # Predict Frame
        self.predict_frame = PredictStartFrame(self)
        self.predict_live_frame = PredictLivePredictionFrame(self)
        self.predict_batch_frame = PredictBatchPredictionFrame(self)
        
        # Train Frame
        self.train_frame = TrainFrame(self)
        
        #Handwriting Recognition frame
        self.handwriting_recognition_frame = HandwritingRecognitionFrame(self)
        
        #Exit frame
        self.exit_frame = ExitFrame(self)
        
        #Navigation frame
        self.navigation_frame = NavigationFrame(self)
        self.navigation_frame.grid(row=0,column=0,sticky='nsew')
        
        
if __name__ == '__main__':
    app = MainApp()
    #app.resizable(False,False)
    app.mainloop()