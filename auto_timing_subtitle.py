from concurrent.futures import process
from curses import window
from optparse import Option
from tkinter import *
from tkinter import filedialog
from pathlib import Path
import os
from tqdm import tqdm
import time
from urllib.parse import quote_plus
import pysubs2
import torch
import whisper
from faster_whisper import WhisperModel

def browseFiles():
    global filename
    filename = filedialog.askopenfilename(initialdir="/",
                                          title="Select a File",
                                          filetypes=(("MP4 Files", "*.mp4*"),("All Types", "*.*")))
    path = Label(window, text=f"Selected files:\n{filename}", width=100, height=5, bg='white')
    path.grid(row=0, column=1)

def list_model():
    model_list = ["large-v2 (Faster-whisper)", "large-v3 (Whisper)"]
    value_inside = StringVar(window)
    value_inside.set("Select Model")
    model_menu = OptionMenu(window, value_inside, *model_list)
    model_menu.grid(row=2, column=1)
    button_process = Button(window, text="Click to Process")
    button_process.grid(row=3, column=1)

window = Tk()
window.title('Image Classification')
window.geometry('800x600')
window.config(background='white')

button_explore = Button(window, text="Browse Files", command=browseFiles)
button_explore.grid(row=1,column=1)
