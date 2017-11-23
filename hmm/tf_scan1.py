import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf


x = tf.placeholder(tf.int32, shape=(None,), name='x')

def square(last, current):
    return current*current

square_op = tf.scan(
    fn=square,
    elems=x
)

with tf.Session() as session:
    o_val = session.run(square_op, feed_dict={x: [1,2,3,4,5]})
    print('square_op: {}'.format(o_val))


N = tf.placeholder(tf.int32, shape=(), name='N')


def fib(last, current):
    return (last[1], last[0] + last[1])

fib_op = tf.scan(
    fn=fib,
    elems=tf.range(N),
    initializer=(0, 1)
)

with tf.Session() as session:
    o_val = session.run(fib_op, feed_dict={N: 8})
    print('fib_op: {}'.format(o_val))


s = tf.placeholder(tf.float32, shape=(None,), name='s')
decay = tf.placeholder(tf.float32, shape=(), name='decay')

def filt(last, current):
    return decay * last + (1-decay) * current

filt_op = tf.scan(
    fn=filt,
    elems=s,
    initializer=0.0
)

np.random.seed(42)

raw_in = np.sin(np.linspace(0,10,200))
noise_in = raw_in + np.random.randn(len(raw_in))

plt.plot(raw_in)
plt.plot(noise_in)
plt.show()

with tf.Session() as session:
    o_val = session.run(filt_op, feed_dict={s: noise_in, decay: 0.95})
    print('filt_op: {}'.format(o_val))

plt.plot(o_val)
plt.show()