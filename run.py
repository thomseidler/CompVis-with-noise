import tensorflow as tf
import tensorflow.keras.applications as tf_apps
from keras import metrics
import matplotlib.pyplot as plt
import numpy as np
import tensorflow_datasets as tfds
import copy
import typing as t
import keras
import functools

import numpy as np
import argparse

import mantik

class NoiseSchedule:
    def __init__(self, timesteps: int):
        self.timesteps = timesteps
        self.schedule = self._generate_noise_schedule(timesteps)
        self.cumulative_product_noise = np.cumprod(1.0 - self.schedule, axis=0)
        self.sqrt_cumulative_product_noise = np.sqrt(
            self.cumulative_product_noise
        )
        self.sqrt_one_minus_cumulative_product_noise = np.sqrt(
            1.0 - self.cumulative_product_noise
        )

    def _generate_noise_schedule(self, timesteps: int):
        return linear_noise_schedule(timesteps)

def get_noisy_sample(
    dataset: np.ndarray, step: int, noise_schedule: NoiseSchedule
):
    return noise_schedule.sqrt_cumulative_product_noise[
        step
    ] * dataset + noise_schedule.sqrt_one_minus_cumulative_product_noise[
        step
    ] * np.random.normal(
        size=dataset.shape
    )


def cosine_noise_schedule(timesteps, s=0.008):
    """
    cosine schedule as proposed in https://arxiv.org/abs/2102.09672
    """
    steps = timesteps + 1
    x = np.linspace(0, timesteps, steps)
    alphas_cumprod = np.cos(((x / timesteps) + s) / (1 + s) * np.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return np.clip(betas, 0.0001, 0.9999)


def linear_noise_schedule(timesteps):
    beta_start = 0.0001
    beta_end = 0.02
    return np.linspace(beta_start, beta_end, timesteps)


def quadratic_noise_schedule(timesteps):
    beta_start = 0.0001
    beta_end = 0.02
    return np.linspace(beta_start**0.5, beta_end**0.5, timesteps) ** 2


def _sigmoid(x: float) -> float:
    return 1 / (1 + np.exp(-x))


def sigmoid_noise_schedule(timesteps):
    beta_start = 0.0001
    beta_end = 0.02
    betas = np.linspace(-6, 6, timesteps)
    return _sigmoid(betas) * (beta_end - beta_start) + beta_start


def get_accuracy_score(predicted_labels, true_str_labels, top_n):
  binary_results = []
  predicted_decoded = keras.applications.resnet.decode_predictions(preds, top=-1)
  for predicted, true in zip(predicted_decoded, true_str_labels):
    relevant_preds = set(map(lambda x: x[0], sorted(predicted, key=lambda x: x[2], reverse=True)[:top_n]))
    if str(true) in relevant_preds:
      binary_results.append(1)
    else:
      binary_results.append(0)
  return np.average(binary_results)


if __name__=="__main__":

    # Configure hyperparameters
    IMG_SIZE = 224
    BATCH_SIZE = 64
    VALIDATION_SIZE = 32*BATCH_SIZE

    model = tf_apps.resnet50.ResNet50(weights="imagenet", include_top=True)

    noise_schedule = NoiseSchedule(timesteps=5000)

    (ds_test, ), ds_info = tfds.load(
    "imagenette/320px-v2", split=["validation"], with_info=True, as_supervised=True,
    )

    label_info = ds_info.features["label"]
    str_labels = [label_info.int2str(label) for (_, label) in ds_test.take(-1)]


    # Note: Use 1000 classes because model is pretrained on ImageNet1k (with 1k classes)
    NUM_CLASSES = 1000#ds_info.features["label"].num_classes
    size = (IMG_SIZE, IMG_SIZE)
    ds_test = ds_test.map(lambda image, label: (tf.image.resize(image, size), label))
    ds_test.shuffle(10)

    def input_preprocess_test(image, label):
        label = tf.one_hot(label, NUM_CLASSES)
        return image, label


    ds_test = ds_test.map(input_preprocess_test, num_parallel_calls=tf.data.AUTOTUNE)
    ds_test = ds_test.batch(batch_size=BATCH_SIZE, drop_remainder=True)

    noise_steps = [0, 100, 250, 500, 750, 1000, 1250, 1500, 1750, 2000, 2500, 3000, 3500, 4000, 4999]


    for noise_step in noise_steps:

        noisify_inner = functools.partial(get_noisy_sample, step=noise_step, noise_schedule=noise_schedule)

        def noisify(image, label):
            noisy_image = noisify_inner(image)
            return (noisy_image, label)

        def preprocess_data(image, label):
            processed_image = keras.applications.resnet.preprocess_input(image)
            return (processed_image, label)

        with mantik.mlflow.start_run():
            # Apply noise
            ds_noise = ds_test.map(preprocess_data)
            ds_noise = ds_noise.map(noisify)

            mlflow.log_param({"noise_step": noise_step})
            mlflow.log_param({"image_size": IMG_SIZE})
            mlflow.log_param({"batch_size": BATCH_SIZE})
            mlflow.log_param({"validation_size": VALIDATION_SIZE})
            mlflow.log_param({"weights": "imagenet"})
            mlflow.log_param({"validation_data": "imagenette"})
            mlflow.log_param({"model": "Resnet50"})

            validation_subset = ds_noise.take(VALIDATION_SIZE)
            
            preds = model.predict(validation_subset)
            
            for i in range(7):
                metric_order = pow(2,i)
                mlflow.log_metric({f"Top{metric_order}": get_accuracy_score(preds, str_labels[:VALIDATION_SIZE], metric_order)})
