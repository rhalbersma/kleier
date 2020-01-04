#          Copyright Rein Halbersma 2019-2020.
# Distributed under the Boost Software License, Version 1.0.
#    (See accompanying file LICENSE_1_0.txt or copy at
#          http://www.boost.org/LICENSE_1_0.txt)

import jax
import jax.numpy as np
import scipy

jax.config.update("jax_enable_x64", True)

def nls(fun, x0, args=()):
    return scipy.optimize.least_squares(fun, x0, jac=jax.jacobian(fun), method='lm', args=args)

def gaussian_decay(x, *args):
    w, t = args
    return w - np.exp(-(t / x)**2)

def exponential_decay(x, *args):
    w, t = args
    return w - np.exp(-(t / x))

def main():
    # https://www.kleier.net/txt/rating_23.html#SEC23
    w = np.array([ 1.0, 0.8, 0.55, 0.3, 0.05 ])
    t = np.array([   0,   1,    2,   3,    4 ])
    x0 = 1

    nls(gaussian_decay, x0, args=(w, t))
    nls(exponential_decay, x0, args=(w, t))
