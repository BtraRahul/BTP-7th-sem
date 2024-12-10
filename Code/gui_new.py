import tkinter as tk
from tkinter import filedialog, ttk, messagebox
from keras.models import load_model
from PIL import Image, ImageTk, ImageOps
import numpy as np
import cv2
import threading  # For running classification in a separate thread

# Disable scientific notation for clarity
np.set_printoptions(suppress=True)

# Load the model
model = load_model("keras_model.h5", compile=False)

# Load the labels
class_names = [line.strip() for line in open("labels.txt", "r").readlines()]

def classify_image(image):
    # Display loading spinner during processing
    spinner_label.place(relx=0.5, rely=0.5, anchor=tk.CENTER)
    root.update()

    # Resize the image to be at least 224x224 and then crop from the center
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.LANCZOS)

    # Convert image to RGB if it is not already in that mode
    if image.mode != "RGB":
        image = image.convert("RGB")

    # Display the input image
    input_img = ImageTk.PhotoImage(image)
    input_img_label.config(image=input_img)
    input_img_label.image = input_img

    # Turn the image into a numpy array
    image_array = np.asarray(image)

    # Normalize the image   
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1

    # Load the image into the array
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    data[0] = normalized_image_array

    # Predict using the model
    prediction = model.predict(data)
    index = np.argmax(prediction)

    # Hide the spinner after processing
    spinner_label.place_forget()

    # Display the results
    if index < len(class_names):
        class_name = class_names[index]
        confidence_score = prediction[0][index]
        result_label.config(text=f"Predicted Class: {class_name}", fg="green")
        confidence_label.config(text=f"Confidence Score: {confidence_score:.4f}", fg="green")
    else:
        result_label.config(text="Not Detected", fg="red")
        confidence_label.config(text="")

def select_image():
    file_path = filedialog.askopenfilename()
    if file_path:
        image = Image.open(file_path)
        threading.Thread(target=classify_image, args=(image,)).start()

# Create the main window
root = tk.Tk()
root.title("Brain Tumor Classification Tool")
root.geometry("800x600")
root.configure(bg="#2b3e50")  # Dark background for contrast

# Title Frame
title_frame = tk.Frame(root, bg="#34495e")
title_frame.pack(pady=20, fill=tk.X)

title_label = tk.Label(title_frame, text="Brain Tumor Classification Tool", font=("Arial", 20, "bold"), bg="#34495e", fg="white")
title_label.pack()

subtitle_label = tk.Label(title_frame, text="Classifying Brain Tumor Types with AI", font=("Arial", 14, "italic"), bg="#34495e", fg="lightgrey")
subtitle_label.pack()

# Input Image Frame
input_frame = tk.Frame(root, bg="#95a5a6")
input_frame.pack(pady=20)

input_label = tk.Label(input_frame, text="Input Image", font=("Arial", 16, "bold"), bg="#95a5a6", fg="black")
input_label.pack()

input_img_label = tk.Label(input_frame, bg="#95a5a6")
input_img_label.pack(pady=10)

# Result Frame
result_frame = tk.Frame(root, bg="#bdc3c7")
result_frame.pack(pady=20)

result_label = tk.Label(result_frame, text="", font=("Arial", 16, "bold"), bg="#bdc3c7")
result_label.pack()

confidence_label = tk.Label(result_frame, text="", font=("Arial", 14), bg="#bdc3c7")
confidence_label.pack()

# Loading spinner animation
spinner = Image.open("a.gif")  # Replace with path to spinner GIF
spinner = spinner.resize((50, 50), Image.LANCZOS)
spinner_img = ImageTk.PhotoImage(spinner)
spinner_label = tk.Label(root, image=spinner_img, bg="#2b3e50")

# Select Image Button
select_button = tk.Button(root, text="Select Image", command=select_image, bg="#3498db", fg="white", font=("Arial", 12, "bold"), width=20)
select_button.pack(pady=10)
select_button.config(borderwidth=2, relief="raised")

# Run the Tkinter application
root.mainloop()
