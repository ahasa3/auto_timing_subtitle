from enum import auto
from tkinter import *
from tkinter import filedialog
from process import auto_sub_jp

def callback():
    global process
    if process is not None:
        process.destroy()
    process = Label(window,text="procssing.....")
    process.pack()
    
    type_ = value_type.get()
    model = value_model.get()
    split = value_split.get()
    method = value_method.get()
    beam = beam_size.get(1.0, "end-1c")
    file_name = filename
    
    split_.destroy()
    method_.destroy()
    model_menu.destroy()
    model_attention.destroy()
    button_process.destroy()
    type_menu.destroy()
    split_menu.destroy()
    method_menu.destroy()
    beam_size.destroy()
    time_consumtion = auto_sub_jp(type_, model, split, method, beam, file_name)
    if time_ is not None:
        time_.destroy()
    time_ = Label(window, text=f'{time_consumtion}')
    time_.pack()
def browseFiles():
    global filename
    global path_
    filename = filedialog.askopenfilename(initialdir="/",
                                          title="Select a File",
                                          filetypes=(("MP4 Files", "*.mp4*"),("All Types", "*.*")))
    if path_ is not None:
        path_.destroy()
    path_ = Label(window, text=f"Selected files:\n{filename}", width=100, height=5, bg='white')
    path_.pack()
    required_list()
    advanced_settings()
    list_model()
    

def list_model():
    global model_menu
    global model_attention
    global button_process
    global value_model
    model_list = ["large-v2", "large-v3"]
    model_attention = Label(window,text="Model size will affect the processing time and transcribe quality\nmodel large-v2 is from faster-whisper\nmodel large-v3 is from whisper")
    model_attention.pack()
    value_model = StringVar(window)
    value_model.set("Select Model")
    model_menu = OptionMenu(window, value_model, *model_list)
    model_menu.pack()
    button_process = Button(window, text="Click to Process",command=callback)
    button_process.pack()

def required_list():
    global type_menu
    global value_type
    type_list = ["audio","video"]
    value_type = StringVar(window)
    value_type.set("Select uploaded file type")
    type_menu = OptionMenu(window, value_type, *type_list)
    type_menu.pack()
def advanced_settings():
    global split_
    global method_
    global split_menu
    global value_method
    global value_split
    global method_menu
    global beam_
    global beam_size
    if split_ is not None:
        split_.destroy()
    split_ = Label(window, text="Option for split line text by spaces. The splited lines all use the same time stamp, with 'adjust_required' label as remark for manual adjustment")
    split_.pack
    split_list = ["No","Yes"]
    value_split=StringVar(window)
    value_split.set("Use Split?")
    split_menu = OptionMenu(window, value_split, *split_list)
    split_menu.pack
    
    if method_ is not None:
        method_.destroy()
    method_ = Label(window, text="Normal segmentation (Modest): When the length of the text after the space exceeds 5 characters, start a new line\nSplit all (Aggressive): start a new line as soon as a space is encountered")
    method_.pack()
    split_method_list=["Modest","Aggressive"]
    value_method = StringVar(window)
    value_method.set("Select split method")
    method_menu = OptionMenu(window, value_method, *split_method_list)
    method_menu.pack()
    
    if beam_ is not None:
        beam_.destroy()
    beam_ = Label(window, text="The higher the Beam Size value, the more paths are explored during recognition, which can help improve recognition accuracy within a certain range,\nbut the relative VRAM usage will also be higher. At the same time, the Beam Size may decrease after exceeding 5-10.\nDefaut Beam size is 5")
    beam_.pack()
    beam_size = Text(window, height=1, width=20)
    beam_size.pack()
    
window = Tk()
window.title('Auto timing subtitle')
window.geometry('800x600')
window.config(background='white')

path_ = None
process = None
split_ = None
method_ = None
beam_ = None
time_ = None
button_explore = Button(window, text="Browse Files", command=browseFiles)
button_explore.pack()

window.mainloop()