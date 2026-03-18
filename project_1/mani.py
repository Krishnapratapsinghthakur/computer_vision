from cProfile import label
import cv2
import numpy as np
import tkinter as tkinter
from tkinter import filedialog
from tkinter import messagebox
from PIL import Image, ImageTk

# dictionayr to store pil images
image = {'original': None, 'sketch': None}


def open_file():
    filepath=filedialog.askopenfilename()
    if not filepath:
        return 
    img=cv2.imread(filepath)
    display_image(img, original=True)
    sketch=convert_to_sketch(img)
    display_image(sketch, original=False)

def convert_to_sketch(img):
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Invert the image
    inverted = cv2.bitwise_not(gray)
    # Apply Gaussian blur
    blurred = cv2.GaussianBlur(inverted, (21, 21), 0)
    # Invert the blurred image
    inverted_blurred = cv2.bitwise_not(blurred)
    # Create the sketch
    sketch = cv2.divide(gray, inverted_blurred, scale=256.0)
    return sketch

def display_image(img, original=True):
    img_rgb=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_pil=Image.fromarray(img_rgb)
    img_tk=ImageTk.PhotoImage(img_pil)
    if original:
        image['original']=img_tk
    else:
        image['sketch']=img_tk
    label=original_image_label if original else sketch_image_label
    label.config(image=img_tk)
    label.image=img_tk

def save_sketch():
    if image['sketch'] is None:
        messagebox.showwarning("Warning", "No sketch to save")
        return
    sketch_filepath=filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG files", "*.png"), ("All files", "*.*")])
    if not sketch_filepath:
        return
    image['sketch'].save(sketch_filepath)
    messagebox.showinfo("Info", "Sketch saved successfully")    


app=tkinter.Tk()
app.title("Sketch Converter")

frame=tkinter.Frame(app)
frame.pack(pady=10,padx=10)

original_image_label=tkinter.Label(frame)
original_image_label.pack(side="left")

sketch_image_label=tkinter.Label(frame)
sketch_image_label.pack(side="right")

button_frame=tkinter.Frame(app)
button_frame.pack()

open_button=tkinter.Button(button_frame, text="Open Image", command=open_file)
open_button.pack(side="left")

save_button=tkinter.Button(button_frame, text="Save Sketch", command=save_sketch)
save_button.pack(side="right")

app.mainloop()



