import tkinter as tk
from PIL import Image, ImageTk
import torch
from diffusers import StableDiffusionPipeline

# Create the Stable Diffusion model pipeline
pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", torch_dtype=torch.float16)
pipe = pipe.to("cuda")

def generate_image():
    prompt_text = prompt_entry.get()
    image = pipe(prompt_text).images[0]

    # Display the generated image in the tkinter window
    display_image_in_tkinter(image)

def display_image_in_tkinter(image):
    window = tk.Toplevel()
    window.title("Generated Image")

    # Resize the image to fit the tkinter window using the LANCZOS resampling method
    width, height = image.size
    aspect_ratio = width / height
    new_width = 512
    new_height = int(new_width / aspect_ratio)
    resized_image = image.resize((new_width, new_height), Image.LANCZOS)

    # Convert the PIL image to a PhotoImage to display in a tkinter Label
    img_tk = ImageTk.PhotoImage(resized_image)

    # Create a Label and display the image
    label = tk.Label(window, image=img_tk)
    label.image = img_tk  # Store the PhotoImage to prevent garbage collection
    label.pack()

# Create the GUI window
root = tk.Tk()
root.title("Text-to-Image Generator")

# Create a Label and Entry widget to get the prompt input from the user
prompt_label = tk.Label(root, text="Enter your prompt:")
prompt_label.pack()
prompt_entry = tk.Entry(root, width=50)
prompt_entry.pack()

# Create a button to generate the image
generate_button = tk.Button(root, text="Generate Image", command=generate_image)
generate_button.pack()

# Start the tkinter main loop
root.mainloop()










# import torch
# from diffusers import StableDiffusionPipeline
# import tkinter

# pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", torch_dtype=torch.float16)
# pipe = pipe.to("cuda")
# prompt = input("your prompt text:")
# image = pipe(prompt).images[0]  # image here is in [PIL format](https://pillow.readthedocs.io/en/stable/)






# # Now to display an image you can either save it such as:
# image.save(f"{prompt}.png")

# # or if you're in a google colab you can directly display it with 
# print(image)

