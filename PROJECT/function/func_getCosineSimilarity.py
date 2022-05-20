import tensorflow as tf
import numpy as np

def getCosineSimilarity(A, B):
    A = np.expand_dims(A, axis=0)
    B = np.expand_dims(B, axis=0)
    A = tf.convert_to_tensor(A)
    B = tf.convert_to_tensor(B)
    dis = tf.compat.v1.losses.cosine_distance(tf.nn.l2_normalize(A, axis = 1), tf.nn.l2_normalize(B, axis = 1), axis = 1)
    cosine_similarity = 1 - dis
    return cosine_similarity.numpy()