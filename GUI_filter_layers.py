

import tkinter as tk
from PIL import ImageTk, Image
from tkinter import filedialog
import numpy as np
import tensorflow
from tensorflow.keras.applications import vgg16
from tensorflow.keras.preprocessing.image import img_to_array
from init_variables import *

from apply_gradcam import Get_GradCam_activation

from look_into_layers import load_model_for_filter_visualsation , look_into_layers
import cv2
import os

physical_devices = tensorflow.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
   tensorflow.config.experimental.set_memory_growth(physical_devices[0], True)

CLASS_INDEX = ['class1' ,'class2','class3','class4','class5','class6','class7','class8','class9']

#https://github.com/tensorflow/tensorflow/blob/fcc4b966f1265f466e82617020af93670141b009/tensorflow/python/keras/applications/imagenet_utils.py#L129
def Sort_Tuple(tup):
    # reverse = None (Sorts in Ascending order)
    # key is set to sort using second element of
    # sublist lambda has been used
    tup.sort(key=lambda x: x[1] , reverse= True)
    return tup

def decode_predictions(preds, top=9):
  global CLASS_INDEX
  results = []
  for pred in preds:
    top_indices = pred.argsort()[-top:][::-1]
    for i in top_indices:
        result = [CLASS_INDEX[i] , (pred[i],) ]
        results.append(result)
        #print(results)
    results = Sort_Tuple(results)
  return results


def load_img():
    global img, image_data
    for img_display in frame.winfo_children():
        img_display.destroy()
    image_data = filedialog.askopenfilename(initialdir="/", title="Choose an image",
                                       filetypes=(("all files", "*.*"), ("png files", "*.png")))
    basewidth = 150
    img = Image.open(image_data).convert('RGB')
    wpercent = (basewidth / float(img.size[0]))
    hsize = int((float(img.size[1]) * float(wpercent)))
    img = img.resize((basewidth, hsize), Image.ANTIALIAS)
    img = ImageTk.PhotoImage(img)
    file_name = image_data.split('/')
    panel = tk.Label(frame, text= str(file_name[len(file_name)-1]).upper()).pack()
    panel_image = tk.Label(frame, image=img).pack()

def show_filters():
    global img, image_data , predictions , edge
    for img_display in frame.winfo_children():
        img_display.destroy()

    image_name_list = look_into_layers(model , image_data , ixs )

    print(image_name_list)

    #for filename in os.listdir(folder):
    for filename in image_name_list:
        heatmap_array = cv2.imread(filename)
        print(type(heatmap_array))
        img1 = Image.fromarray(heatmap_array)
        print(img1)

        #basewidth = 150
        #wpercent = (basewidth / float(img1.size[0]))
        #hsize = int((float(img1.size[1]) * float(wpercent)))
        #img1 = img1.resize((basewidth, hsize), Image.ANTIALIAS)
        img1 = ImageTk.PhotoImage(img1)

        #file_name = image_data.split('/')
        #panel = tk.Label(frame, text=str(file_name[len(file_name) - 1]).upper()).pack()
        panel = tk.Label(frame, text='amrit').pack()
        panel_image = tk.Label(frame, image=img1).pack()


root = tk.Tk()
root.title('Skin lesions Classifier')
root.geometry("1000x1000")
#root.iconbitmap('class.ico')
root.resizable(True, True)

tit = tk.Label(root, text="Skin lesions Classifier", padx=25, pady=6, font=("", 12)).pack()

canvas = tk.Canvas(root, height=900, width=800, bg='grey')
canvas.pack()

frame = tk.Frame(root, bg='white')
frame.place(relwidth=0.8, relheight=0.8, relx=0.1, rely=0.1)

chose_image = tk.Button(root, text='Choose Image',padx=35, pady=10, fg="white", bg="grey", command=load_img)
chose_image.pack(side=tk.LEFT)

class_image = tk.Button(root, text='Show_filters', padx=35, pady=10,fg="white", bg="grey", command=show_filters)
class_image.pack(side=tk.RIGHT)

#model = vgg16.VGG16(weights='imagenet')
model , ixs = load_model_for_filter_visualsation(model_architecture)

root.mainloop()
