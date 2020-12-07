

#https://pierpaolo28.github.io/blog/blog40/
#https://github.com/grey-ninja/Machine-Learning-Tkinter-GUI/blob/master/gui-ml.py

import tkinter as tk
from PIL import ImageTk, Image
from tkinter import filedialog
import numpy as np
import tensorflow
from tensorflow.keras.applications import vgg16
from tensorflow.keras.preprocessing.image import img_to_array
from apply_gradcam_old import Get_GradCam_activation
from init_variables import *
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
  print(preds)   ## [0. 0. 0. 0. 0. 1. 0. 0. 0.]
  for pred in preds:
    print(pred)  # [0. 0. 0. 0. 0. 1. 0. 0. 0.]
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

def classify():
    global img, image_data , predictions , edge
    for img_display in frame.winfo_children():
        img_display.destroy()

    original1 = Image.open(image_data).convert('RGB')
    original = original1.resize((224, 224), Image.ANTIALIAS)
    numpy_image = img_to_array(original)
    image_batch = np.expand_dims(numpy_image, axis=0)
    processed_image = vgg16.preprocess_input(image_batch.copy())
    predictions = model.predict(processed_image)
    label = decode_predictions(predictions)

    heatmap_array = Get_GradCam_activation(model, image_data , predictions, annotate_output=False, label=None)

    print(heatmap_array.shape)
    print(type(heatmap_array))

    img = Image.fromarray(heatmap_array)
    edge = Image.fromarray(heatmap_array)

    basewidth = 250
    #img = Image.open(heatmap).convert('RGB')
    wpercent = (basewidth / float(img.size[0]))
    hsize = int((float(img.size[1]) * float(wpercent)))
    img = img.resize((basewidth, hsize), Image.ANTIALIAS)

    img = ImageTk.PhotoImage(img)
    file_name = image_data.split('/')
    panel = tk.Label(frame, text=str(file_name[len(file_name) - 1]).upper()).pack()
    panel_image = tk.Label(frame, image=img).pack()

    #print(predictions)

    table = tk.Label(frame, text="Top image class predictions and confidences").pack()
    for i in range(0, len(label)):
        result = tk.Label(frame, text= (str(label[i][0]).upper() + ': ' + str(round(label[i][1][0] * 100, 3)) + '%')).pack()


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
model = tensorflow.keras.models.load_model(saved_cbr_weights_directory)

root.mainloop()
