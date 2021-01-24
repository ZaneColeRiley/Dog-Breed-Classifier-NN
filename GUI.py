# Imports
import string

from PIL import Image, ImageTk
from NN import model, predict_breed
from tkinter import Tk, ttk, filedialog, Label
from tkinter import LEFT, BOTTOM, TOP, RIGHT


class GUI(object):
    def __init__(self):
        self.root = Tk()
        self.root.geometry("900x900")
        self.root.title("Dog Breed Classifier")

        self.breed_exact = ttk.Label(self.root)
        self.breed_predicted = ttk.Label(self.root)
        self.nn_score = ttk.Label(self.root)

        self.predict_button = ttk.Button(self.root, text="Predict", command=self.edit_text_labels)

        self.breed_exact.pack(side=TOP)
        self.breed_predicted.pack(side=TOP)
        self.nn_score.pack(side=TOP)
        self.predict_button.pack(side=BOTTOM)

    def get_img_path(self):
        dog_img_path = filedialog.askopenfile()
        return dog_img_path

    def get_dir(self):
        dog_breed = filedialog.askdirectory()
        return dog_breed

    def get_breed(self):
        breed = self.get_dir()
        breed = breed.replace(breed[0], '').replace(str(string.digits), '').replace('-', '').replace('_', ' ')
        return str(breed)

    def get_predicted_breed(self):
        model.load_weights("Models/Dog_Classifier.h5")
        breed, score = predict_breed(input("Image Path: ").replace("'\'", '/'))
        return str(breed), score

    def edit_text_labels(self):
        predicted_breed, score = self.get_predicted_breed()
        exact_breed = self.get_breed()
        self.breed_exact.configure(text=str(exact_breed))
        self.breed_predicted.configure(text=str(predicted_breed))
        self.nn_score.configure(text="{:.2f}%".format(score))


if __name__ == '__main__':
    classifier = GUI()
    classifier.root.mainloop()
