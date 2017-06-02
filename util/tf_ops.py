import tensorflow as tf

def get_solver_adam(learning_rate, beta1):
    return tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=beta1)