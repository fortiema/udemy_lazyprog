import numpy as np
import theano
import theano.tensor as T

N = T.iscalar('N')

def recurrence(n, fn_1, fn_2):
    return fn_1 + fn_2, fn_1

outputs, updates = theano.scan(
    fn=recurrence,
    sequences=T.arange(N),
    n_steps=N,
    outputs_info=[1., 1.]
)

fib = theano.function(
    inputs=[N],
    outputs=outputs
)

o_val = fib(8)

print(f'output: {o_val}')