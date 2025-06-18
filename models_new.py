from tensorflow.keras.layers import *
from tensorflow.keras.initializers import TruncatedNormal
from tensorflow.keras.layers import *
from tensorflow.keras.initializers import TruncatedNormal
import numpy as np
from tensorflow.keras.models import Model
from model_config import dropout_rate, stddev, num_experts, expert_units
import importlib
import tensorflow as tf
import layers
importlib.reload(layers)
from layers import _apply_attention, positional_encoding
from tensorflow.keras.layers import Input, Embedding, LSTM, GRU, Bidirectional, Dense, Reshape, GlobalAveragePooling1D, Dropout, BatchNormalization, Add
from model_config import dropout_rate, stddev, num_experts, expert_units
import matplotlib.pyplot as plt


def get_model(model_name, feature_config, tf_config, is_training=False):
    if model_name=='base':
        return build_base_model(feature_config, tf_config, is_training)
    else:
        raise NotImplementedError


def build_base_model(feature_config, tf_config, is_training):
    