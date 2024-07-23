
# Interactive Learning Digit Recogonizer from Scratch

This project shows how to make a number recognizer using a neural network that was built from scratch with numpy. The interactive learning module is one of the best parts of this project. It lets users draw their own numbers and check the model's estimates. If the model's guess is wrong, it uses what the user types as an example to get better at recogonizing the digit. This project is for people looking to learn how to make their own neural networks from scratch but get confused by the math.

## Libraries Used

1. numpy
2. matplotlib
3. opencv
4. scikit-learn (for train-test split)


## References 

The neural network in this project was inspired by [Samson Zhang's video](https://www.youtube.com/watch?v=w8yWXqWQYmU&pp=ygUbbmV1cmFsIG5ldHdvcmsgZnJvbSBzY3JhdGNo), where he beautifully explains the concepts and demonstrates the code. While the dataset remains the same, this repository presents my own interpretation and implementation of the model.


## Neural Network

![Neural Network Diagram](https://github.com/PhantomX256/Interactive-Learning-Digit-Recognizer-from-Scratch/blob/1b85b170dcbc1dba2bea74d6a4f9a3f37a406448/Neural_Net.png?raw=true)


## Installing necessary libraries

Run the following command in order to install the required libraries and dependencies

`pip install -r requirements.txt`

## How to Use the Project

1. After cloning the repository, open `test_image.png` in Paint or your preferred image editor and clear the canvas.
2. Draw your own digit using white color on a black canvas.
3. Save the image, open `test_model.ipynb`, and run all the code cells to check the model's prediction.

## How to Train the Model

If you wish to train the model to suit your needs, follow these steps:

1. Open `train_model.ipynb` and modify the dataset to be loaded into the notebook.
2. Run all the code cells with your changes and wait for the model to finish training.
3. Once satisfied with the model's accuracy, head to `test_image.ipynb` to test it out!