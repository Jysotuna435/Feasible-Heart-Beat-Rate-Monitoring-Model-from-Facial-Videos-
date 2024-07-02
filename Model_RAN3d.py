import tensorflow as tf
from tensorflow.keras import layers
import numpy as np

def conv3d_block(inputs, filters, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='same'):
    x = layers.Conv3D(filters, kernel_size, strides=strides, padding=padding)(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    return x


def residual_block(inputs, filters):
    x = conv3d_block(inputs, filters)
    x = conv3d_block(x, filters)
    residual = layers.Conv3D(filters, kernel_size=(1, 1, 1), padding='same')(inputs)
    output = layers.add([x, residual])
    output = layers.ReLU()(output)
    return output


def attention_module(inputs):
    x = layers.Conv3D(filters=1, kernel_size=(1, 1, 1), padding='same')(inputs)
    x = layers.Activation('sigmoid')(x)
    output = layers.multiply([inputs, x])
    return output


def build_3d_ran(input_shape, num_classes):
    inputs = layers.Input(shape=input_shape)

    # Initial convolutional layer
    x = conv3d_block(inputs, filters=64, kernel_size=(7, 7, 7), strides=(2, 2, 2))
    x = layers.MaxPooling3D(pool_size=(3, 3, 3), strides=(2, 2, 2), padding='same')(x)

    # Stack of residual blocks with attention modules
    for _ in range(4):
        x = residual_block(x, filters=64)
        x = attention_module(x)

    # Global average pooling
    x = layers.GlobalAveragePooling3D()(x)

    # Fully connected layers
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.5)(x)

    # Output layer
    outputs = layers.Dense(num_classes, activation='softmax')(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name='3d_ran')
    return model


def Model_RAN3d(Images,Target):
    input_shape = (64, 64, 64, 3)  # Example input shape
    num_classes = Target.shape[1]  # Example number of classes
    model = build_3d_ran(input_shape, num_classes)
    model.summary()
    weight = model.layers[-1].get_weights()[0]
    Feat = np.resize(weight, [ Target.shape[0],100])
    return Feat

