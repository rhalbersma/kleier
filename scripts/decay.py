#          Copyright Rein Halbersma 2019-2020.
# Distributed under the Boost Software License, Version 1.0.
#    (See accompanying file LICENSE_1_0.txt or copy at
#          http://www.boost.org/LICENSE_1_0.txt)

import jax
import jax.numpy as jnp
import scipy

jax.config.update("jax_enable_x64", True)

def gaussian_decay(x, *args):
    w, t = args
    return w - jnp.exp(-(t / x)**2)

def exponential_decay(x, *args):
    w, t = args
    return w - jnp.exp(-(t / x))

def nls(fun, x0, args=()):
    return scipy.optimize.least_squares(fun, x0, jac=jax.jacobian(fun), method='lm', args=args)

def mle(fun, x0, args=()):
    logl = lambda x, *args: 0.5 * jnp.sum(fun(x, *args)**2)
    return scipy.optimize.minimize(logl, x0, args=args, method='Newton-CG', jac=jax.grad(logl), hess=jax.hessian(logl))

dist_norm = {
    'cdf'   : lambda x: jax.scipy.stats.norm.cdf(x),
    'logcdf': lambda x: jax.scipy.stats.norm.logcdf(x),
    'scale' : 200 * jnp.sqrt(2)
}

dist_logistic = {
    'cdf'   : lambda x: jax.scipy.special.expit(x),
    'logcdf': lambda x: jnp.log(jax.scipy.special.expit(x)),
    'scale' : 400 / jnp.log(10)
}

def score_balance(x, *args):
    W, R, dist = args
    delta = (x - R) / dist['scale']
    We = dist['cdf']
    return W - We(delta)

def squared_error(x, *args):
    return score_balance(x, *args)**2

def log_likelihood(x, *args):
    W, R, dist = args
    delta = (x - R) / dist['scale']
    logWe = dist['logcdf']
    return W * logWe(delta) + (1 - W) * logWe(-delta)

def tpr_root(x0, args=()):
    f = lambda x, *args: jnp.sum(score_balance(x, *args))
    return scipy.optimize.root_scalar(f, args=args, method='newton', fprime=jax.grad(f), fprime2=jax.hessian(f), x0=x0)

def tpr_nls(x0, args=()):
    fun = lambda x, *args: score_balance(x, *args)
    return scipy.optimize.least_squares(fun, x0, jac=jax.jacobian(fun), method='lm', args=args)

def tpr_mle(x0, args=()):
    fun = lambda x, *args: -jnp.sum(log_likelihood(x, *args))
    return scipy.optimize.minimize(fun, x0, args=args, method='Newton-CG', jac=jax.grad(fun), hess=jax.hessian(fun))

def tpr_ci(x0, args=()):
    fun = lambda x, *args: -jnp.sum(log_likelihood(x, *args))
    mle = scipy.optimize.minimize(fun, x0, args=args, method='Newton-CG', jac=jax.grad(fun), hess=jax.hessian(fun))
    se = scipy.stats.norm.ppf(.975) * jnp.sqrt(jnp.diag(jnp.linalg.inv(jax.hessian(fun)(mle.x, *args))))
    LR = 0.5 * scipy.stats.chi2.ppf(.95, 1)
    f = lambda x, *args: mle.fun + LR - fun(x, *args)
    lb = scipy.optimize.root_scalar(f, args=args, method='newton', fprime=jax.grad(f), fprime2=jax.hessian(f), x0=mle.x - se).root
    ub = scipy.optimize.root_scalar(f, args=args, method='newton', fprime=jax.grad(f), fprime2=jax.hessian(f), x0=mle.x + se).root
    return lb, mle.x, ub

def main():
    # https://www.kleier.net/txt/rating_23.html#SEC23
    w = jnp.array([ 1.0, 0.8, 0.55, 0.3, 0.05 ])
    t = jnp.array([   0,   1,    2,   3,    4 ])
    x0 = 1

    nls(gaussian_decay, x0, args=(w, t))
    nls(exponential_decay, x0, args=(w, t))
    mle(gaussian_decay, x0, args=(w, t))
    mle(exponential_decay, x0, args=(w, t))

    W = jnp.array([  0.0,    1.0])
    R = jnp.array([900.0, 1100.0])
    R0 = R.mean() * 1.1
    tpr_root(R0, args=(W, R, dist_norm    ))
    tpr_root(R0, args=(W, R, dist_logistic))
    tpr_nls(R0, args=(W, R, dist_norm    ))
    tpr_nls(R0, args=(W, R, dist_logistic))
    tpr_mle(R0, args=(W, R, dist_norm    ))
    tpr_mle(R0, args=(W, R, dist_logistic))
    tpr_ci(R0, args=(W, R, dist_norm))
    tpr_ci(R0, args=(W, R, dist_logistic))
