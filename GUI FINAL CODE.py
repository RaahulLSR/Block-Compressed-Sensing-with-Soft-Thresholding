# -*- coding: utf-8 -*-
"""
Created on Fri May 24 20:54:01 2024

@author: Asus
"""

# -*- coding: utf-8 -*-
"""
Created on Tue May 21 12:40:15 2024
    
@author: Asus
"""

import tkinter as tk
from tkinter import filedialog, ttk
from PIL import Image, ImageTk
import pywt
import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import dct, idct
from sklearn.metrics import mean_squared_error
from skimage.metrics import structural_similarity as ssim
import pandas as pd
import threading
import queue

def brek():
    pass

# Define necessary functions (same as in your original code)
def undersampler(undersample_rate, original1):
    n = original1.shape[0] * original1.shape[1]
    original_undersampled = (original1.reshape(-1) * np.random.permutation(
        np.concatenate((np.ones(int(n * undersample_rate)), np.zeros(int(n * (1-undersample_rate))))))
    ).reshape(original1.shape)
    return original_undersampled

def flat_wavelet_transform2(x, method='bior1.3'):
    coeffs = pywt.wavedec2(x, method)
    output = coeffs[0].reshape(-1)
    for tups in coeffs[1:]:
        for c in tups:
            output = np.concatenate((output, c.reshape(-1)))
    return output

def inverse_flat_wavelet_transform2(X, shape, method='bior1.3'):
    shapes = pywt.wavedecn_shapes(shape, method)
    nx = shapes[0][0]
    ny = shapes[0][1]
    n = nx * ny
    coeffs = [X[:n].reshape(nx, ny)]
    for i, d in enumerate(shapes[1:]):
        vals = list(d.values())
        nx = vals[0][0]
        ny = vals[0][1]
        coeffs.append((X[n: n + nx * ny].reshape(nx, ny),
                       X[n + nx * ny: n + 2 * nx * ny].reshape(nx, ny),
                       X[n + 2 * nx * ny: n + 3 * nx * ny].reshape(nx, ny)))
        n += 3 * nx * ny
    return pywt.waverec2(coeffs, method)

def soft_thresh(x, lam):
    if not isinstance(x[0], complex):
        return np.zeros(x.shape) + (x + lam) * (x < -lam) + (x - lam) * (x > lam)
    else:
        return np.zeros(x.shape) + (abs(x) - lam) / abs(x) * x * (abs(x) > lam)

def dct2(a):
    return dct(dct(a.T, norm='ortho').T, norm='ortho')

def idct2(a):
    return idct(idct(a.T, norm='ortho').T, norm='ortho')

def distance(x, y):
    return sum(abs(x.reshape(-1) - y.reshape(-1)))

def reconstructor(ind ,lam, y, domain, original, epoch,output_queue):
    eps = 1e-2
    lam_decay = 0.995
    minlam = 1
    err2 = []

    xhat = y.copy()
    if domain == "wavelet":
        for i in range(epoch):
            method = 'haar'
            xhat_old = xhat
            Xhat_old = flat_wavelet_transform2(xhat, method)
            Xhat = soft_thresh(Xhat_old, lam)
            xhat = inverse_flat_wavelet_transform2(Xhat, (1024, 1024), method)
            xhat[y != 0] = y[y != 0]

            xhat = xhat.astype(int)
            xhat[xhat < 0] = 0
            xhat[xhat > 255] = 255
            err2.append(distance(original, xhat))
            lam *= lam_decay
    elif domain == "fourier":
        for i in range(epoch):
            xhat_old = xhat
            Xhat_old = np.fft.fft2(xhat)
            Xhat = soft_thresh(Xhat_old, lam)
            xhat = np.fft.ifft2(Xhat).real
            xhat[y != 0] = y[y != 0]

            xhat = xhat.astype(int)
            xhat[xhat < 0] = 0
            xhat[xhat > 255] = 255
            err2.append(distance(original, xhat))
            lam *= lam_decay
    elif domain == "dct":
        for i in range(epoch):
            xhat_old = xhat
            Xhat_old = dct2(xhat)
            Xhat = soft_thresh(Xhat_old, lam)
            xhat = idct2(Xhat)
            xhat[y != 0] = y[y != 0]

            xhat = xhat.astype(int)
            xhat[xhat < 0] = 0
            xhat[xhat > 255] = 255
            err2.append(distance(original, xhat))
            lam *= lam_decay
    xhat.reshape(original.shape)

    output_queue.put((ind,xhat, err2))
    return

def reconstructor1(lam, y, domain, original, epoch,output_queue):
    eps = 1e-2
    lam_decay = 0.995
    minlam = 1
    err2 = []

    xhat = y.copy()
    if domain == "wavelet":
        for i in range(epoch):
            method = 'haar'
            xhat_old = xhat
            Xhat_old = flat_wavelet_transform2(xhat, method)
            Xhat = soft_thresh(Xhat_old, lam)
            xhat = inverse_flat_wavelet_transform2(Xhat, (1024, 1024), method)
            xhat[y != 0] = y[y != 0]

            xhat = xhat.astype(int)
            xhat[xhat < 0] = 0
            xhat[xhat > 255] = 255
            err2.append(distance(original, xhat))
            lam *= lam_decay
    elif domain == "fourier":
        for i in range(epoch):
            xhat_old = xhat
            Xhat_old = np.fft.fft2(xhat)
            Xhat = soft_thresh(Xhat_old, lam)
            xhat = np.fft.ifft2(Xhat).real
            xhat[y != 0] = y[y != 0]

            xhat = xhat.astype(int)
            xhat[xhat < 0] = 0
            xhat[xhat > 255] = 255
            err2.append(distance(original, xhat))
            lam *= lam_decay
    elif domain == "dct":
        for i in range(epoch):
            xhat_old = xhat
            Xhat_old = dct2(xhat)
            Xhat = soft_thresh(Xhat_old, lam)
            xhat = idct2(Xhat)
            xhat[y != 0] = y[y != 0]

            xhat = xhat.astype(int)
            xhat[xhat < 0] = 0
            xhat[xhat > 255] = 255
            err2.append(distance(original, xhat))
            lam *= lam_decay
    xhat.reshape(original.shape)

    return xhat, err2
    

def calculate_metrics(original, reconstructed):
    mse_value = mean_squared_error(original, reconstructed)
    psnr_value = 20 * np.log10(np.max(original) / np.sqrt(mse_value))
    data_range = original.max() - original.min()
    ssim_value, _ = ssim(original, reconstructed, data_range=data_range, full=True)
    ncc_value = np.sum(original * reconstructed) / np.sqrt(np.abs(np.sum(original**2) * np.sum(reconstructed**2)))
    return mse_value, psnr_value, ssim_value, ncc_value

def update_metrics_table(metrics_df):
    for i in metrics_table.get_children():
        metrics_table.delete(i)
    for _, row in metrics_df.iterrows():
        metrics_table.insert('', 'end', values=list(row))

# GUI Functions
def load_image():
    global original_image
    file_path = filedialog.askopenfilename()
    if file_path:
        image = Image.open(file_path)
        image = image.convert('L')
        image = image.resize((1024,1024), Image.LANCZOS)
        original_image = np.array(image, dtype=np.float32)

        img_tk = ImageTk.PhotoImage(image)
        original_img_label.configure(image=img_tk)
        original_img_label.image = img_tk
        

def perform_undersampling():
    global undersampled_image
    try:
        compression_ratio = float(compression_ratio_entry.get())
    except ValueError:
        tk.messagebox.showerror("Invalid Input", "Please enter a valid compression ratio.")
        return

    undersampled_image = undersampler(compression_ratio, original_image)

    undersampled_image_pil = Image.fromarray(undersampled_image).convert('L')
    img_tk = ImageTk.PhotoImage(undersampled_image_pil)
    undersampled_img_label.configure(image=img_tk)
    undersampled_img_label.image = img_tk
def split_image(image):
    height, width = image.shape[:2]
    part_height, part_width = height // 8, width // 8
    parts = []
    for i in range(8):
        for j in range(8):
            parts.append(image[i * part_height:(i + 1) * part_height, j * part_width:(j + 1) * part_width])
    return parts


# Define a function to combine four parts into a single image
def combine_parts(parts):
    rows = []
    for i in range(8):
        row = np.concatenate(parts[i * 8:(i + 1) * 8], axis=1)
        rows.append(row)
    combined_image = np.concatenate(rows, axis=0)
    return combined_image


import time
# Initialize an empty list to store the time taken for each divided image
divided_image_times = []
# Create and start threads
threads = []

# Create a queue to store the output from threads
output_queue = queue.Queue()


def perform_reconstruction():
    method = method_var.get()
    lam = 100  # initial lambda value
    epoch = 600  # number of epochs
    undersampled_parts = split_image(undersampled_image)

    # Split original image into four parts
    original_parts = split_image(original_image)
    
    # Initialize an empty list to store reconstructed parts
    reconstructed_parts = [None]*64
    
    # Iterate over each pair of undersampled and original parts
    start_time = time.time() 
    for i in range(64):
        thread = threading.Thread(target=reconstructor, args=(i,lam,undersampled_parts[i], method, original_parts[i], epoch,output_queue))
        thread.start()
        threads.append(thread)

    # Join all threads to wait for them to finish
    for thread in threads:
        thread.join()
    while not output_queue.empty():
        co,a, _ = output_queue.get()
        reconstructed_parts[co] = a
    # Combine reconstructed parts into a single image
    reconstructed_image = combine_parts(reconstructed_parts)
    end_time = time.time()  # Record end time
    # Calculate time taken
    time_taken = end_time - start_time
    print("Time taken for a divided image:", time_taken, "seconds")
    
    # Convert reconstructed image to PIL format
    reconstructed_image_pil = Image.fromarray(reconstructed_image).convert('L')
    # Display the reconstructed image
    img_tk = ImageTk.PhotoImage(reconstructed_image_pil)
    reconstructed_img_label.configure(image=img_tk)
    reconstructed_img_label.image = img_tk
    
    start_time_full = time.time()  # Record start time
    # Full image reconstruction
    reconstructed_image_full, _ = reconstructor1(lam, undersampled_image, method, original_image, epoch,output_queue)
    end_time_full = time.time()  # Record end time
    # Calculate time taken for full image reconstruction
    time_taken_full = end_time_full - start_time_full
    print("Time taken for the full image reconstruction:", time_taken_full, "seconds")
    
    

    # Calculate metrics for the entire reconstructed image
    mse, psnr, ssim_value, ncc = calculate_metrics(original_image, reconstructed_image)
    # Create a DataFrame to store metrics
    metrics_df = pd.DataFrame({
        'Metric': ['MSE', 'PSNR', 'SSIM', 'NCC'],
        'Value': [mse, psnr, ssim_value, ncc]
})
    
    mse, psnr, ssim_value, ncc = calculate_metrics(original_image, reconstructed_image_full)
    # Create a DataFrame to store metrics
    metrics_df1 = pd.DataFrame({
        'Metric': ['MSE', 'PSNR', 'SSIM', 'NCC'],
        'Value': [mse, psnr, ssim_value, ncc]
})
    print(metrics_df)
    print("full image:")
    print(metrics_df1)
    
    # Update metrics table
    update_metrics_table(metrics_df)

# Create the main window
root = tk.Tk()
root.title("Image Reconstruction GUI")
screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()

# Calculate the x and y coordinates for the Tk root window
x_coordinate = (screen_width / 2) - (1600 / 2)  # Assuming the width of the window is 1600
y_coordinate = (screen_height / 2) - (1000 / 2)  # Assuming the height of the window is 1000

# Set the geometry of the root window to center it on the screen
root.geometry(f"1600x1000+{int(x_coordinate)}+{int(y_coordinate)}")
root.configure(bg="#333333")

# Create and place widgets
load_img_button = tk.Button(root, text="Load Image", command=load_image,bg="blue", fg="white")
load_img_button.grid(row=0, column=0, padx=10, pady=10)

method_var = tk.StringVar(root)
method_var.set("dct")
method_frame = tk.Frame(root, bg="blue")
method_frame.grid(row=0, column=2, padx=10, pady=10)

# Create the OptionMenu inside the frame
method_menu = tk.OptionMenu(method_frame, method_var, "wavelet", "fourier", "dct")
method_menu.config(bg="blue",fg='white')  # Set background color of the dropdown menu
method_menu.grid(row=0, column=0, padx=10, pady=10)

compression_ratio_label = tk.Label(root, text="Compression Ratio:",bg="blue", fg="white")
compression_ratio_label.grid(row=0, column=4, padx=10, pady=10)
compression_ratio_entry = tk.Entry(root)
compression_ratio_entry.grid(row=0, column=4, padx=10, pady=10)
compression_ratio_entry.insert(0, "0.5")

def draw_photo_frame():
    # Create a frame for photo display
    photo_frame = tk.Frame(root, bg="white", width=1024, height=1024)
    photo_frame.grid(row=1, column=1, columnspan=3, padx=10, pady=10)
    photo_frame1 = tk.Frame(root, bg="white", width=1024, height=1024)
    photo_frame1.grid(row=1, column=3, columnspan=3, padx=10, pady=10)
    photo_frame3 = tk.Frame(root, bg="white", width=1024, height=1024)
    photo_frame3.grid(row=1, column=5, columnspan=3, padx=10, pady=10)
# Call the function to draw the frame initially
draw_photo_frame()

original_img_label = tk.Label(root)
original_img_label.grid(row=1, column=2, padx=10, pady=10)

undersampled_img_label = tk.Label(root)
undersampled_img_label.grid(row=1, column=4, padx=10, pady=10)

reconstructed_img_label = tk.Label(root)
reconstructed_img_label.grid(row=1, column=6, padx=10, pady=10)

undersample_button = tk.Button(root, text="Perform Undersampling", command=perform_undersampling,bg="blue", fg="white")
undersample_button.grid(row=2, column=2, padx=10, pady=10)

reconstruct_button = tk.Button(root, text="Perform Reconstruction", command=perform_reconstruction,bg="blue", fg="white")
reconstruct_button.grid(row=2, column=4, padx=10, pady=10)

metrics_table = ttk.Treeview(root, columns=('Metric', 'Value'), show='headings', height=10)
metrics_table.heading('Metric', text='Metric')
metrics_table.heading('Value', text='Value')
metrics_table.column('Metric', width=400)
metrics_table.column('Value', width=400)
style = ttk.Style()
style.configure("Treeview.Heading", font=("Arial", 16))
style.configure("Treeview", font=("Arial",14), rowheight=120)
metrics_table.grid(row=3, column=2, columnspan=3, padx=10, pady=10)

# Start the main loop
root.mainloop()
