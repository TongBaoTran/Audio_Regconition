

import os.path
import tkinter as tk
import tkinter.font as tkFont
from tkinter import *
from tkinter import filedialog as fd

import librosa
from librosa import display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pygame
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from PIL import Image, ImageTk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from pandas import DataFrame
from pathlib import Path


WIDTH = 1050
HEIGHT = 530
app = tk.Tk()
app.title('Sound Classification App')
app.geometry('{}x{}'.format(WIDTH, HEIGHT))
currentDir = os.path
backgroundimg = Image.open("Background.png")
backgroundimg = backgroundimg.resize((WIDTH, HEIGHT), Image.LANCZOS)
resized_backgroundimg = ImageTk.PhotoImage(backgroundimg)
canvas = Canvas(app, width=WIDTH, height=HEIGHT)
canvas.pack(fill=BOTH, expand=True)
canvas.create_image(0, 0, image=resized_backgroundimg, anchor=NW)


#q = queue.Queue()
#recording = False
#file_exists = False
#recordingcount = 0
colmap = "magma"


# # Functions to play, stop and record audio in Python voice recorder
# # The recording is done as a thread to prevent it being the main process
# def threading_rec(x):
#     if x == 1:
#         # If recording is selected, then the thread is activated
#         global t1
#         t1 = threading.Thread(target=record_audio)
#         t1.start()
#     elif x == 2:
#         # To stop, set the flag to false
#         global recording
#         recording = False
#         messagebox.showinfo(title= "Recording", message="Recording finished")
#     elif x == 3:
#         # To play a recording, it must exist.
#         if file_exists:
#             # Read the recording if it exists and play it
#             data, fs = sf.read("trial.wav", dtype='float32')
#             sd.play(data, fs)
#             sd.wait()
#         else:
#             messagebox.showerror(message="Record something to play")
#
#
# def callback(indata, frames, time, status):
#     q.put(indata.copy())
#
#
# def record_audio():
#     global recording
#     recording = True
#     global file_exists
#     messagebox.showinfo(title = "Recording", message="Recording starts")
#     global recordingcount
#     recordingcount += 1
#     newfile = "recording_" + str(recordingcount) + ".wav"
#     with sf.SoundFile(newfile, mode='w', samplerate=44100, channels=2) as file:
#         with sd.InputStream(samplerate=44100, channels=2, callback=callback):
#             while recording == True:
#                 file_exists = True
#                 file.write(q.get())
#     # # the file name output you want to record into
#     # filename = "recording_" + str(recordingcount) + ".wav"
#     # # set the chunk size of 1024 samples
#     # chunk = 1024
#     # # sample format
#     # FORMAT = pyaudio.paInt16
#     # # mono, change to 2 if you want stereo
#     # channels = 1
#     # # 44100 samples per second
#     # sample_rate = 44100
#     # record_seconds = 10
#     # # initialize PyAudio object
#     # p = pyaudio.PyAudio()
#     # # open stream object as input & output
#     # stream = p.open(format=FORMAT,
#     #                 channels=channels,
#     #                 rate=sample_rate,
#     #                 input=True,
#     #                 output=True,
#     #                 frames_per_buffer=chunk)
#     # frames = []
#     # print("Recording...")
#     # for i in range(int(44100 / chunk * record_seconds)):
#     #     data = stream.read(chunk)
#     #     # if you want to hear your voice while recording
#     #     # stream.write(data)
#     #     frames.append(data)
#     # print("Finished recording.")
#     # # stop and close stream
#     # stream.stop_stream()
#     # stream.close()
#     # # terminate pyaudio object
#     # p.terminate()
#     # # save audio file
#     # # open the file in 'write bytes' mode
#     # wf = wave.open(filename, "wb")
#     # # set the channels
#     # wf.setnchannels(channels)
#     # # set the sample format
#     # wf.setsampwidth(p.get_sample_size(FORMAT))
#     # # set the sample rate
#     # wf.setframerate(sample_rate)
#     # # write the frames as bytes
#     # wf.writeframes(b"".join(frames))
#     # # close the file
#     # wf.close()
#
#     #top.destroy()
#     global audiofile
#     audiofile = newfile
#     textstr = newfile
#     updateiputtextbox(textstr)
#     global imagefile
#     clmap = getcolormap(color_var.get())
#     imagefile = create_spectrogram(audiofile, clmap)
#     img = ImageTk.PhotoImage(Image.open(imagefile))
#     spectroimg.configure(image=img)
#     spectroimg.image = img


def importinput(inputstr):
    inputstr = input_var.get()
    global audiofile
    global imagefile
    if inputstr == "Upload Audio":
        filetypes = [("Sound files", "*.wav *.mp3 *.ogg")]
        filename = fd.askopenfilename(title='Open audio file', initialdir='/', filetypes=filetypes)
        audiofile = filename
        textstr = filename.split('/')[-1]
        updateiputtextbox(textstr)
        clmap = getcolormap(color_var.get())
        imagefile = create_spectrogram(audiofile, clmap)
        im = Image.open(imagefile)
        im = im.resize((223, 217))
        img = ImageTk.PhotoImage(im)
        spectroimg.configure(image=img)
        spectroimg.image = img
    elif inputstr == "Upload Spectrogram":
        filetypes = [("Image files", "*.png *.jpg")]
        filename = fd.askopenfilename(title='Open image file', initialdir='/', filetypes=filetypes)
        imagefile = filename
        textstr = filename.split('/')[-1]
        updateiputtextbox(textstr)
        im = Image.open(imagefile)
        im = im.resize((223, 217))
        img = ImageTk.PhotoImage(im)
        spectroimg.configure(image=img)
        spectroimg.image = img
    predict_btn.configure(bg="yellow")
    # elif inputstr == "Real-time Recording":
    #     global top
    #     top = Toplevel(app)
    #     top.geometry("300x100")
    #     top.title("Recording")
    #     Button(top, text="Start Recording", font=('Helvetica', 12), command=lambda m=1: threading_rec(m)).place(x=20,y=20)
    #     Button(top, text="Stop Recording", font=('Helvetica', 12), command=lambda m=2: threading_rec(m)).place(x=150,y=20)


def play():
    if input_textbox.get("1.0", END) != "\n" and (
            input_var.get() == "Upload Audio" ):
        pygame.init()
        pygame.mixer.init()
        pygame.mixer.music.load(audiofile)
        pygame.mixer.music.play()


def stop():
    if input_textbox.get("1.0", END) != "\n" and (
            input_var.get() == "Upload Audio" ):
        pygame.init()
        pygame.mixer.init()
        pygame.mixer.music.stop()


def importmodel():
    global modelfile
    filetypes = [("Model files", "*.h5")]
    filename = fd.askopenfilename(title='Open model file', initialdir='/', filetypes=filetypes)
    modelfile = filename
    textstr = filename.split('/')[-1]
    model_textbox.config(state=NORMAL)
    model_textbox.delete('1.0', END)
    model_textbox.insert('1.0', textstr)
    model_textbox.config(state=DISABLED)


def importlabel():
    global labelfile
    filetypes = [("Label files", "*.csv")]
    filename = fd.askopenfilename(title='Open label file', initialdir='/', filetypes=filetypes)
    labelfile = filename
    textstr = filename.split('/')[-1]
    label_textbox.config(state=NORMAL)
    label_textbox.delete('1.0', END)
    label_textbox.insert('1.0', textstr)
    label_textbox.config(state=DISABLED)


def importcolormap(inputstr):
    global imagefile
    inputstr = color_var.get()
    global colmap
    clmap = getcolormap(inputstr)
    if input_textbox.get("1.0", END) != "\n" and (input_var.get() == "Upload Audio" or input_var.get() == "Real-time Recording"):
        imagefile = create_spectrogram(audiofile, clmap)
    colmap = clmap
    if input_textbox.get("1.0", END) != "\n":
        img = ImageTk.PhotoImage(Image.open(imagefile))
        spectroimg.configure(image=img)
        spectroimg.image = img
    if input_textbox.get("1.0", END) != "\n" and model_textbox.get("1.0", END) != "\n" and label_textbox.get("1.0", END) != "\n" and output_textbox.index("end") !=0:
        predict()


def getcolormap(inputstr):
    if inputstr == "Plasma":
        cl = "plasma"
    elif inputstr == "Magma":
        cl = "magma"
    elif inputstr == "Grayscale":
        cl = "gray"
    elif inputstr == "Viridis":
        cl = "viridis"
    elif inputstr == "Inferno":
        cl = "inferno"
    else:
        cl = "magma"
    return cl

def create_spectrogram(audiofile, cl):
    plt.interactive(False)
    signal, sample_rate = librosa.load(audiofile)
    fig = plt.figure(figsize=[0.72, 0.72])
    ax = fig.add_subplot(111)
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)
    ax.set_frame_on(False)
    S = librosa.feature.melspectrogram(y=signal, sr=sample_rate)
    librosa.display.specshow(librosa.power_to_db(S, ref=np.max), cmap = cl)
    imagefile = 'spectrogram123.png'
    plt.savefig(imagefile, dpi=400, bbox_inches='tight', pad_inches=0)
    plt.close()
    fig.clf()
    plt.close(fig)
    plt.close('all')
    del signal, sample_rate, fig, ax, S, audiofile,cl
    return imagefile

def updateiputtextbox(textstr):
    input_textbox.config(state=NORMAL)
    input_textbox.delete('1.0', END)
    input_textbox.insert('1.0', textstr)
    input_textbox.config(state=DISABLED)

def setmodeltype(inputstr):
    inputstr = model_type_var.get()
    global modeltype
    x = 1
    if inputstr=="Multi Label":
        x=2
    else:
        x=1
    modeltype =x
    if input_textbox.get("1.0", END) != "\n" and model_textbox.get("1.0", END) != "\n" and label_textbox.get("1.0", END) != "\n" and output_textbox.index("end") !=0:
        predict()


def predict():
    if input_textbox.get("1.0", END) != "\n" and model_textbox.get("1.0", END) != "\n" and label_textbox.get("1.0",END) != "\n" :
        loadedmodel = tf.keras.models.load_model(modelfile)
        x = loadedmodel.layers[0].input_shape
        input2D = (x[1], x[2])
        classes = pd.read_csv(labelfile).loc[:, "Class/Label"]
        cut = 1
        if modeltype == 1:
            cut = 1
        else:
            if len(classes)>10:
                cut=len(classes)//10
            else:
                cut = len(classes)

        img = image.load_img(imagefile, target_size=input2D)
        img_tensor = image.img_to_array(img)
        img_tensor = np.expand_dims(img_tensor, axis=0)
        img_tensor /= 255.
        result = loadedmodel.predict(img_tensor)[0]
        pairlist = zip(classes, result)
        global result1
        result1 =list(pairlist)
        result1.sort(key = lambda x: x[1])
        result1.reverse()
        output_textbox.delete(0, END)
        label_list = []
        percent_list = []
        for i in range(cut):
            output_textbox.insert(i, str(result1[i][0]) + "   --->   " + str(round(result1[i][1]*100,2)) + "%")
            label_list.append(str(result1[i][0]))
            percent_list.append(round(result1[i][1]*100,2))
        predict_btn.configure(bg="green")

        data1 = {'Label': label_list,'Percentage': percent_list}
        df1 = DataFrame(data1, columns=['Label', 'Percentage'])
        figure1 = plt.figure(figsize=(3, 4), dpi=100)
        ax1 = figure1.add_subplot(111)
        bar1 = FigureCanvasTkAgg(figure1, app)
        bar1.get_tk_widget().place(x = 700, y = 10)
        df1 = df1[['Label', 'Percentage']].groupby('Label').sum()
        df1.plot(kind='bar', legend=None, ax=ax1)
        ax1.set_title('Prediction')
        ax1.set_xlabel('Labels', color='k', labelpad=30)
        fig = plt.gcf()
        fig.tight_layout()
        plt.gcf().subplots_adjust(bottom=0.4)
        #plt.savefig("chart.png", bbox_index='tight')
        plt.tick_params(axis='both', which='major', labelsize=10)
        del ax1, fig


# fig = plt.figure(figsize=[0.72, 0.72])
#     ax = fig.add_subplot(111)
#     ax.
#     ax.axes.get_xaxis().set_visible(False)
#     ax.axes.get_yaxis().set_visible(False)
#     ax.set_frame_on(False)
#     S = librosa.feature.melspectrogram(y=signal, sr=sample_rate)
#     librosa.display.specshow(librosa.power_to_db(S, ref=np.max), cmap=cl)
#     imagefile = 'spectrogram.png'
#     plt.savefig(imagefile, dpi=400, bbox_inches='tight', pad_inches=0)
#     plt.close()

# Import Input
InputOptionList = ["Upload Audio", "Upload Spectrogram"]
input_var = tk.StringVar(app)
input_var.set("Import Input")
opt = tk.OptionMenu(app, input_var, *InputOptionList, command=importinput)
opt.config(width=17, font=('Helvetica', 11), bg="white")
opt.place(x=10, y=10)
input_textbox = Text(app, width=40, height=1.5)
input_textbox.config(state=DISABLED)
input_textbox.place(x=200, y=10)

# Play Button
play_img = Image.open("playbutton.png")
play_img = play_img.resize((35, 35), Image.LANCZOS)
resized_play_img = ImageTk.PhotoImage(play_img)
play_btn = Button(app, image=resized_play_img, width=30, height=30, command=play)
play_btn.place(x=527, y=10)

# Stop Button
stop_img = Image.open("stop.png")
stop_img = stop_img.resize((30, 30), Image.LANCZOS)
resized_stop_img = ImageTk.PhotoImage(stop_img)
stop_btn = Button(app, image=resized_stop_img, width=30, height=30, bg="white", command=stop)
stop_btn.place(x=563, y=10)

# Import model
import_model_btn = Button(app, width=19, height=1, bg="white", text="Import model", font=('Helvetica', 12),
                          command=importmodel)
import_model_btn.place(x=10, y=50)
model_textbox = Text(app, width=40, height=1.5)
model_textbox.config(state=DISABLED)
model_textbox.place(x=200, y=50)

# Import labels
import_label_btn = Button(app, width=19, height=1, bg="white", text="Import labels", font=('Helvetica', 12),
                          command=importlabel)
import_label_btn.place(x=10, y=90)
label_textbox = Text(app, width=40, height=1.5)
label_textbox.config(state=DISABLED)
label_textbox.place(x=200, y=90)

# Import Model Type
ModelTypeOptionList = ["Single Label", "Multi Label"]
model_type_var = tk.StringVar(app)
model_type_var.set("Choose model type")
model_type_opt = tk.OptionMenu(app, model_type_var, *ModelTypeOptionList, command= setmodeltype)
model_type_opt.config(width=17, font=('Helvetica', 11), bg="white")
model_type_opt.place(x=10, y=130)

# Import ColorMap
ColorMapOptionList = ["Plasma", "Grayscale", "Inferno", "Magma", "Viridis"]
color_var = tk.StringVar(app)
color_var.set("Select Color Map")
color_opt = tk.OptionMenu(app, color_var, *ColorMapOptionList, command=importcolormap)
color_opt.config(width=17, font=('Helvetica', 11), bg="white")
color_opt.place(x=10, y=170)

# Spectrogram Frame
spectro_frame = Frame(app, bg="white", width=280, height=286)
spectro_frame.place(x=10, y=220)
spectroimg = Label(app)
spectroimg.place(x=30, y=250)

# Plot frame
plot_frame = Frame(app, bg="white", width=330, height=495)
plot_frame.place(x=690, y=10)


# Output Textbox
output_textbox = Listbox(app, height=14, width=42, font= tkFont.Font(family="Arial", size=13, weight="bold", slant="italic"))
output_textbox.place(x=300, y=220)

# Predict
predict_btn = Button(app, text="Predict", width=10, height=1, bg="yellow", font=('Helvetica', 25), command = predict)
predict_btn.place(x=200, y=140)


# Function to resize the window
def resize_image(e):
    global image, resized, image2
    image = Image.open("Background.png")
    resized = image.resize((e.width, e.height), Image.LANCZOS)
    image2 = ImageTk.PhotoImage(resized)
    canvas.create_image(0, 0, image=image2, anchor='nw')


app.resizable(False, False)
app.mainloop()
