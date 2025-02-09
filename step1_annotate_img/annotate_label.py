import os
import json
from tkinter import Tk, Button, Label, StringVar, Radiobutton, Frame
from PIL import Image, ImageTk


class ImagePairLabeler:
    def __init__(self, image_directory, json_path):
        self.root = Tk()
        self.root.title("Image Pair Labeler")

        self.root.geometry("800x600")  # Increase window size

        self.label = Label(self.root, text="Label image pairs")
        self.label.pack()

        self.canvas_frame = Frame(self.root)
        self.canvas_frame.pack()

        self.canvas_before = None
        self.canvas_after = None
        self.current_images = []
        self.image_index = 0
        self.image_files = []
        self.labels = {}
        self.meal_count = 1
        self.json_file = json_path
        self.image_directory = image_directory

        self.meal_type_var = StringVar(value="NA")  # Default meal type
        self.setup_buttons()
        self.load_directory()

    def setup_buttons(self):
        # Buttons for leftover
        Button(self.root, text="Pair as Full", command=lambda: self.save_label("Full", is_pair=True)).pack()
        Button(self.root, text="Pair as Some Leftover", command=lambda: self.save_label("Some Leftover", is_pair=True)).pack()
        Button(self.root, text="Pair as Little Leftover", command=lambda: self.save_label("Little Leftover", is_pair=True)).pack()
        Button(self.root, text="Pair as No Leftover", command=lambda: self.save_label("No Leftover", is_pair=True)).pack()

        # Horizontal layout for meal type selection
        meal_type_frame = Frame(self.root)
        meal_type_frame.pack()
        Label(meal_type_frame, text="Select Meal Type:").pack(side="left")
        Radiobutton(meal_type_frame, text="Breakfast", variable=self.meal_type_var, value="Breakfast").pack(side="left")
        Radiobutton(meal_type_frame, text="Lunch/Dinner", variable=self.meal_type_var, value="Lunch/Dinner").pack(side="left")
        Radiobutton(meal_type_frame, text="Snacks", variable=self.meal_type_var, value="Snacks").pack(side="left")
        Radiobutton(meal_type_frame, text="NA", variable=self.meal_type_var, value="NA").pack(side="left")

        # Button for single image
        Button(self.root, text="Single as Exception", command=lambda: self.save_label("Exception", is_pair=False)).pack()
        Button(self.root, text="Next", command=self.next_image).pack()

    def load_directory(self):
        # Get all .jpg, .png, and .jpeg files sorted by name (assuming timestamps are part of names)
        self.image_files = sorted(
            [os.path.join(self.image_directory, file) for file in os.listdir(self.image_directory)
             if file.lower().endswith((".jpg", ".png", ".jpeg"))]
        )

        if not self.image_files:
            self.label.config(text="No .jpg, .png, or .jpeg files found in the specified directory.")
            return

        # Load existing labels if JSON file exists
        if os.path.exists(self.json_file):
            with open(self.json_file, "r") as f:
                self.labels = json.load(f)

        self.label.config(text=f"{len(self.image_files)} images found.")
        self.image_index = 0
        self.display_images()

    def format_image_path(self, image_path):
        # Convert full path to a user-understandable relative path with folder and file name
        folder_name = os.path.basename(self.image_directory)
        file_name = os.path.basename(image_path)
        return f"{folder_name}/{file_name}"

    def display_images(self):
        if self.image_index >= len(self.image_files):
            self.label.config(text="All images labeled!")
            return

        self.current_images = [self.image_files[self.image_index]]

        # If a consecutive image exists, allow pairing
        if self.image_index + 1 < len(self.image_files):
            self.current_images.append(self.image_files[self.image_index + 1])

        self.display_image(self.current_images[0], "Image 1 (Before Meal)", is_before=True)

        if len(self.current_images) > 1:
            self.display_image(self.current_images[1], "Image 2 (After Meal)", is_before=False)
        else:
            if self.canvas_after:
                self.canvas_after.destroy()
                self.canvas_after = None

        self.label.config(
            text=f"Labeling images: {os.path.basename(self.current_images[0])}"
            + (f" and {os.path.basename(self.current_images[1])}" if len(self.current_images) > 1 else "")
        )

    def display_image(self, image_path, title, is_before):
        image = Image.open(image_path)
        desired_size = (300, 300)
        image = image.resize(desired_size)

        canvas = self.canvas_before if is_before else self.canvas_after
        if canvas:
            canvas.destroy()

        canvas = Label(self.canvas_frame, text=title)
        canvas.pack(side="left")

        img = ImageTk.PhotoImage(image)
        canvas.img = img  # Keep a reference to avoid garbage collection
        canvas.config(image=img)

        if is_before:
            self.canvas_before = canvas
        else:
            self.canvas_after = canvas

    def save_label(self, label, is_pair):
        if self.image_index >= len(self.image_files):
            return

        meal_type = self.meal_type_var.get().strip()

        if is_pair and len(self.current_images) == 2:
            image_before, image_after = self.current_images
            self.labels[f"meal{self.meal_count}"] = {
                "meal_type": meal_type,
                "before image": self.format_image_path(image_before),
                "after image": self.format_image_path(image_after),
                "leftover": label
            }
            self.meal_count += 1
        else:
            image_single = self.current_images[0]
            self.labels[f"single_image_{self.meal_count}"] = {
                "meal_type": meal_type,
                "image": self.format_image_path(image_single),
                "description": label
            }
            self.meal_count += 1

        # Save to JSON file
        with open(self.json_file, "w") as f:
            json.dump(self.labels, f, indent=4)

        self.next_image(is_pair)

    def next_image(self, is_pair=True):
        self.image_index += 2 if is_pair and len(self.current_images) == 2 else 1
        self.display_images()

    def run(self):
        self.root.mainloop()


if __name__ == "__main__":
    # Set the directory to scan for images
    image_directory = "C:/Users/mahim/PycharmProjects/PythonProject/img_dir/CaM01-049"

    # Set the path to save the JSON file
    json_path = "label_output_CaM01-049.json"

    labeler = ImagePairLabeler(image_directory, json_path)
    labeler.run()
