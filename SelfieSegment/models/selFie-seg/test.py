import functools
import time
import numpy as np
import tensorflow as tf

import IPython.display as display

import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['figure.figsize'] = (12, 12)
mpl.rcParams['axes.grid'] = False

selfie_path = "./selfie_test.png"

selfie_model_path = "./selfie_segmentation_landscape.tflite"

# Function to load an image from a file, and add a batch dimension.


def load_img(path_to_img):
    img = tf.io.read_file(path_to_img)
    img = tf.io.decode_image(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)
    img = img[tf.newaxis, :]

    return img

# Function to pre-process by resizing an central cropping it.


def preprocess_image(image, target_dim):
    # Resize the image so that the shorter dimension becomes 256px.
    shape = tf.cast(tf.shape(image)[1:-1], tf.float32)
    short_dim = min(shape)
    scale = target_dim / short_dim
    new_shape = tf.cast(shape * scale, tf.int32)
    print("new_shape", new_shape)
    image = tf.image.resize(image, new_shape)

    # Central crop the image.
    image = tf.image.resize_with_crop_or_pad(image, target_dim, target_dim)

    return image


# Load the input images.
content_image = load_img(selfie_path)

# Preprocess the input images.
preprocessed_selfie_image = preprocess_image(content_image, 256)

print('Content Image Shape:', preprocessed_selfie_image.shape)


def imshow(image, title=None):
    if len(image.shape) > 3:
        image = tf.squeeze(image, axis=0)

    plt.imshow(image)
    if title:
        plt.title(title)


plt.subplot(1, 2, 1)
imshow(preprocessed_selfie_image, 'Content Image')


# Function to run style prediction on preprocessed style image.
def run_style_predict(preprocessed_style_image):
    # Load the model.
    interpreter = tf.lite.Interpreter(model_path=selfie_model_path)

    # Set model input.
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    interpreter.set_tensor(input_details[0]["index"], preprocessed_style_image)

    # Calculate style bottleneck.
    interpreter.invoke()
    style_bottleneck = interpreter.tensor(
        interpreter.get_output_details()[0]["index"]
    )()

    return style_bottleneck


# Calculate style bottleneck for the preprocessed style image.
style_bottleneck = run_style_predict(preprocessed_selfie_image)
print('Style Bottleneck Shape:', style_bottleneck.shape)
print("value :\n", style_bottleneck)
plt.subplot(1, 2, 2)
imshow(style_bottleneck, 'style_bottleneck')


plt.show()
