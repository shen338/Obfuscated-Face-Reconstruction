import tensorflow as tf

t = tf.random_uniform((100,222,222,3))
t1 = tf.random_uniform((100,222,222,3))
sample = tf.concat([t, t1], axis=3)

def interpolate(a, b):
    shape = tf.concat((tf.shape(a)[0:1], tf.tile([1], [a.shape.ndims - 1])), axis=0)
    alpha = tf.random_uniform(shape=shape, minval=0., maxval=1.)
    inter = a + alpha * (b - a)
    inter.set_shape(a.get_shape().as_list())
    return inter

t1 = interpolate(t, t1)

print(type(t))
with tf.Session() as sess:
    value = sess.run(t1)
print(type(value))
print(value.shape)

