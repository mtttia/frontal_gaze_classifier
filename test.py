#!/usr/bin/env python3

import argparse
import sys
import keras
import numpy as np
import tensorflow as tf

def load_model(model_path):
    print("reading model from: " + model_path)
    model = keras.models.load_model(model_path)
    return model


def test_model(image_path, model_path):
    model = load_model(model_path)
    img = keras.utils.load_img(image_path, target_size=(224, 224))
    img_array = keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)
    img_array = img_array / 255.0
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions[0])
    confidence = np.max(predictions[0])
    print(f"Predictions: {predictions}")
    return predicted_class, confidence

def get_class_name(index):
    classes = [
        "closed_look",
        "forward_look",
        "left_look",
        "right_look",
    ]
    class_name = classes[index]
    return class_name

def main():
    parser = argparse.ArgumentParser(
        description="Testa il modello"
    )

    parser.add_argument(
        "image_path",  type=str, help="Path dell'immagine da visualizzare"
    )

    parser.add_argument(
        "-m", "--model", type=str, help="Path to keras model file",
    )

    args = parser.parse_args()
    image_path = args.image_path
    model_path = args.model or "./model.keras"
    predicted_class, confidence = test_model(image_path, model_path)
    class_name = get_class_name(predicted_class)
    print(f"Predicted class: {class_name}")
    print(f"Confidence: {confidence:.2f}")

if __name__ == "__main__":
    main()
    sys.exit(0)