import sys
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
import numpy as np
import tensorflow_probability as tfp

tfd = tfp.distributions

# --- [1. تعريف طبقة المخرجات البايزية] ---
@tf.keras.utils.register_keras_serializable()
class SovereignBayesianLayer(layers.Layer):
    def __init__(self, **kwargs):
        super(SovereignBayesianLayer, self).__init__(**kwargs)

    def call(self, inputs):
        # تقسيم المخرجات إلى (المتوسط) و (الاشتقاق المعياري)
        loc = inputs[..., :3]
        scale = 1e-3 + tf.keras.activations.softplus(0.05 * inputs[..., 3:])
        dist = tfd.Normal(loc=loc, scale=scale)
        
        # حيلة التوافق لإرفاق التوزيع بالـ Tensor
        mean = dist.mean()
        mean._dist = dist 
        return mean

    def compute_output_shape(self, input_shape):
        return input_shape[:-1] + (3,)

# --- [2. بناء الدماغ السيادي المصحح] ---
def build_sovereign_model():
    acoustic_input = layers.Input(shape=(500, 1), name="Acoustic_Stream")
    env_input = layers.Input(shape=(3,), name="Environmental_Context")
    
    # استخدام أسماء ثابتة للطبقات لضمان مطابقة الأوزان
    x = layers.Conv1D(64, kernel_size=5, activation='relu', name="conv_feature_extractor")(acoustic_input)
    x = layers.BatchNormalization(name="batch_norm_1")(x)
    x = layers.MaxPooling1D(pool_size=2, name="max_pool_1")(x)
    x = layers.Flatten(name="flatten")(x)
    
    y = layers.Dense(16, activation='relu', name="env_dense")(env_input)
    
    combined = layers.Concatenate(name="fusion_layer")([x, y])
    z = layers.Dense(128, activation='relu', name="dense_deep_1")(combined)
    z = layers.Dense(64, activation='relu', name="dense_deep_2")(z)
    
    params = layers.Dense(6, name="Param_Layer")(z)
    sovereign_output = SovereignBayesianLayer(name="Bayesian_Output")(params)
    
    model = models.Model(inputs=[acoustic_input, env_input], outputs=sovereign_output)
    
    def negloglik(y_true, y_pred):
        dist = y_pred._dist
        return -tf.reduce_mean(dist.log_prob(y_true))

    model.compile(optimizer='adam', loss=negloglik, run_eagerly=True)
    return model