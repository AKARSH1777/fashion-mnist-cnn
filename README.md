# FashionMNIST CNN Classifier

A beginner-friendly computer vision project that classifies clothing images from the **FashionMNIST** dataset using a Convolutional Neural Network (CNN) built with **PyTorch**.

This project explores the core CNN workflow—from image tensors and convolution layers to model training, evaluation, and feature-map visualization—alongside the interactive [CNN Explainer](https://poloclub.github.io/cnn-explainer/).

## Project Goals

- Load and explore the FashionMNIST dataset.
- Understand image tensor dimensions and batches.
- Build a CNN using PyTorch.
- Inspect tensor shapes throughout the network.
- Train the model using backpropagation and an optimizer.
- Evaluate performance on the test set.
- Visualize predictions and convolutional feature maps.
- Save the trained model weights.

## Dataset

FashionMNIST contains 70,000 grayscale clothing images:

- **Training set:** 60,000 images
- **Test set:** 10,000 images
- **Image dimensions:** 28 × 28 pixels
- **Channels:** 1 (grayscale)
- **Classes:** 10

The classes are:

1. T-shirt/top
2. Trouser
3. Pullover
4. Dress
5. Coat
6. Sandal
7. Shirt
8. Sneaker
9. Bag
10. Ankle boot

The notebook downloads the dataset automatically through `torchvision.datasets.FashionMNIST`.

## Model Architecture

The notebook implements this CNN:

```text
Input image: 1 × 28 × 28
        |
Conv2d (1 → 16, kernel size 3, padding 1)
        |
      ReLU
        |
MaxPool2d (2 × 2)
        |
Feature maps: 16 × 14 × 14
        |
Conv2d (16 → 32, kernel size 3, padding 1)
        |
      ReLU
        |
MaxPool2d (2 × 2)
        |
Feature maps: 32 × 7 × 7
        |
      Flatten
        |
1,568 features
        |
Linear layer (1,568 → 10)
        |
10 class scores (logits)
```

### Key concepts

- **Convolution:** learns local visual patterns using trainable filters.
- **ReLU:** adds nonlinearity by setting negative activations to zero.
- **Max pooling:** reduces the spatial dimensions of feature maps.
- **Flattening:** converts feature maps into a one-dimensional vector.
- **Fully connected layer:** produces scores for the 10 clothing classes.
- **Cross-entropy loss:** measures prediction error against the correct labels.
- **Backpropagation:** calculates gradients used to update the model parameters.

## Repository Structure

```text
fashion-mnist-cnn/
├── FashionMNIST_CNN.ipynb   # Main Google Colab/Jupyter notebook
├── README.md                # Project documentation
└── .gitignore               # Optional Git ignore rules
```

The dataset is downloaded when the notebook runs, so it does not need to be committed to the repository.

## Run the Project in Google Colab

**Recommended: Google Colab**—no local Python environment is required.

1. Open [Google Colab](https://colab.research.google.com/).
2. Select **File → Upload notebook**.
3. Upload `FashionMNIST_CNN.ipynb` from this repository.
4. (Optional) Select **Runtime → Change runtime type** and choose a GPU if one is available. A CPU is also sufficient for this small model.
5. Run the notebook cells in order.

The notebook imports PyTorch, downloads FashionMNIST, displays sample images, builds the CNN, trains it, evaluates it, visualizes predictions and feature maps, and saves the model weights.

### Requirements

The notebook uses:

- Python 3
- PyTorch
- torchvision
- NumPy
- Matplotlib

These packages are generally available in Colab. For local use, install them with:

```bash
pip install torch torchvision numpy matplotlib
```

## Training and Evaluation

The starter notebook uses:

- **Batch size:** 64
- **Optimizer:** Adam
- **Learning rate:** 0.001
- **Loss function:** CrossEntropyLoss
- **Epochs:** 5

These are initial settings for experimentation, not a claim of optimal performance. The notebook prints training loss and accuracy for each epoch, then calculates accuracy on the separate test set.

**Test accuracy:** Run the notebook and replace this line with your actual measured result. Do not report an estimated value as an experimental result.

## Outputs

When you run the notebook, it:

- Displays sample FashionMNIST images and labels.
- Prints tensor shapes at each CNN stage.
- Plots training loss and training accuracy.
- Reports test-set accuracy.
- Displays example predictions beside their true labels.
- Visualizes activations from the first convolutional layer.
- Saves trained weights to `fashion_cnn.pth`.

The checkpoint is created in the active Colab runtime. Download it if you want to keep a local copy; Colab runtime files may disappear when the session ends.

## Learning Resources

- [CNN Explainer — Interactive CNN visualization](https://poloclub.github.io/cnn-explainer/)
- [PyTorch: Learn the Basics](https://docs.pytorch.org/tutorials/beginner/basics/intro.html)
- [FashionMNIST dataset documentation](https://pytorch.org/vision/stable/generated/torchvision.datasets.FashionMNIST.html)

## Possible Improvements

- Compare models with different numbers of convolutional layers.
- Try different batch sizes and learning rates.
- Train for more epochs and compare test performance.
- Build a confusion matrix to inspect commonly confused classes.
- Compare the CNN against a fully connected neural network.
- Add an inference interface using FastAPI and a web frontend.

## What I Learned

Update this section as you complete the project. For example:

- How image data is represented as PyTorch tensors.
- How convolution, activation functions, and pooling transform images.
- How a CNN learns through loss calculation, backpropagation, and optimization.
- How to evaluate a model on unseen data.
- How to visualize predictions and intermediate feature maps.

## License

Choose a license before reusing or distributing this project. If you intend to make it open source, add a `LICENSE` file with your chosen license.
