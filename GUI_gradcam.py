

#https://pierpaolo28.github.io/blog/blog40/
#https://github.com/grey-ninja/Machine-Learning-Tkinter-GUI/blob/master/gui-ml.py

import tkinter as tk
from PIL import ImageTk, Image
from tkinter import filedialog
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import vgg16
from tensorflow.keras.preprocessing.image import img_to_array
from apply_gradcam import *
from init_variables import *

physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
   tf.config.experimental.set_memory_growth(physical_devices[0], True)

model_architecture = model_architecture
CLASS_INDEX = ['class1' ,'class2','class3','class4','class5','class6','class7','class8','class9']

#https://github.com/tensorflow/tensorflow/blob/fcc4b966f1265f466e82617020af93670141b009/tensorflow/python/keras/applications/imagenet_utils.py#L129

def load_img():
    global img, image_data , model_architecture
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

def classify():
    global img, image_data , predictions , edge
    for img_display in frame.winfo_children():
        img_display.destroy()

    stacked_output_image , resized_overlapped_img , label = gradCAM(image_data,layer_index, img_size, model_architecture, intensity= intensity, gradcam_save_res = gradcam_save_res)

    results = label
    #img = Image.fromarray((stacked_output_image * 255).astype(np.uint8))
    #edge = Image.fromarray((stacked_output_image * 255).astype(np.uint8))

    img = Image.fromarray((stacked_output_image ).astype(np.uint8))
    edge = Image.fromarray((stacked_output_image).astype(np.uint8))

    basewidth = 150
    wpercent = (basewidth / float(img.size[0]))
    hsize = int((float(img.size[1]) * float(wpercent)))
    img = img.resize((basewidth, hsize), Image.ANTIALIAS)

    img = ImageTk.PhotoImage(img)
    file_name = image_data.split('/')
    panel = tk.Label(frame, text=str(file_name[len(file_name) - 1]).upper()).pack()
    panel_image = tk.Label(frame, image=img).pack()

    #print(predictions)

    table = tk.Label(frame, text="Top image class predictions and confidences").pack()
    '''for i in range(0, len(label)):
        result = tk.Label(frame, text= (str(label[i][0]).upper() + ': ' + str(round(label[i][1] * 100, 3)) + '%')).pack()
        print(i)'''
    for i in range(0, len(results[0])):
        result = tk.Label(frame, text= (str(results[0][i][1]).upper() + ': ' + str(round(( results[0][i][2] * 100) , 3)) + '%')).pack()

    #result = tk.Label(frame, text=label).pack()

def savefile():
    global edge
    filename = filedialog.asksaveasfile(mode='w', defaultextension=".jpg")
    if not filename:
        return
    edge.save(filename)


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

class_image = tk.Button(root, text='Classify Image', padx=35, pady=10,fg="white", bg="grey", command=classify)
class_image.pack(side=tk.RIGHT)

button = tk.Button(text="Save as",padx=35, pady=10,fg="white", bg="grey", command=savefile)
button.pack(side=tk.BOTTOM)


#model = vgg16.VGG16(weights='imagenet')
'''model = tf.keras.models.load_model('/home/amritpal/PycharmProjects/100-days-of-code/100_days_of_code/Skin_disease_working_version_minimal'
                                           '/saved_model/model_name_RGB_224__0.55__1.61__12-3_2:59')'''

root.mainloop()
