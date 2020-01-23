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

class logistic:
    def cdf(self, x, loc=0, scale=1):
        y = (x - loc) / scale
        return jax.scipy.special.expit(y)
    
    def logcdf(self, x, loc=0, scale=1): 
        y = (x - loc) / scale
        return jnp.log(self.cdf(y))
    
    def pdf(self, x, loc=0, scale=1):
        y = (x - loc) / scale
        return self.cdf(y) * (1.0 - self.cdf(y))

    def logpdf(self, x, loc=0, scale=1):
        y = (x - loc) / scale
        return jnp.log(self.pdf(y))

    def ppf(self, q, loc=0, scale=1): 
        return jnp.array(jax.scipy.special.logit(q) * scale + loc, 'float64')

dist_norm     = (jax.scipy.stats.norm, 200 * jnp.sqrt(2))
dist_logistic = (logistic()          , 400 / jnp.log(10))

def score_balance(x, *args):
    W, R, (dist, scale) = args
    We = dist.cdf
    return W - We(x, R, scale)

def squared_error(x, *args):
    return score_balance(x, *args)**2

def log_likelihood(x, *args):
    W, R, (dist, scale) = args
    logWe = dist.logcdf
    return jnp.sum(W * logWe(x, R, scale) + (1.0 - W) * logWe(-x, -R, scale))

def L2_reg(x, *args):
    Ra = jnp.mean(args[1])
    dist, scale = dist_norm
    logXe = dist.logpdf
    return jnp.sum(logXe(x, Ra, scale))

def tpr_ppf(args):
    W, R, (dist, scale) = args
    return dist.ppf(W.mean(), R.mean(), scale)

def tpr_root(args, x0=None):
    f = lambda x, *args: jnp.sum(score_balance(x, *args))
    if x0 is None:
        x0 = tpr_ppf(args)
    return scipy.optimize.root_scalar(f, args=args, method='newton', fprime=jax.grad(f), fprime2=jax.hessian(f), x0=x0)

def tpr_nls(args, fun=None, x0=None):
    fun = lambda x, *args: score_balance(x, *args)
    if x0 is None:
        x0 = tpr_ppf(args)
    return scipy.optimize.least_squares(fun, x0, jac=jax.jacobian(fun), method='lm', args=args)

def tpr_mle(args, x0=None, ci=False):
    fun = lambda x, *args: -log_likelihood(x, *args) #- L2_reg(x, *args)
    if x0 is None:
        x0 = tpr_ppf(args)
    mle = scipy.optimize.minimize(fun, x0, args=args, method='trust-ncg', jac=jax.grad(fun), hess=jax.hessian(fun), options={'gtol': 1e-8})
    if not ci:
        return mle
    LR = 0.5 * scipy.stats.chi2.ppf(.95, 1)
    f = lambda x, *args: mle.fun + LR - fun(x, *args)
    se = scipy.stats.norm.ppf(.975) * jnp.sqrt(jnp.diag(jnp.linalg.inv(mle.hess)))
    lb = scipy.optimize.root_scalar(f, args=args, method='newton', fprime=jax.grad(f), fprime2=jax.hessian(f), x0=mle.x - se).root
    ub = scipy.optimize.root_scalar(f, args=args, method='newton', fprime=jax.grad(f), fprime2=jax.hessian(f), x0=mle.x + se).root
    return mle, lb, ub

def main():
    # https://www.kleier.net/txt/rating_23.html#SEC23
    w = jnp.array([ 1.0, 0.8, 0.55, 0.3, 0.05 ])
    t = jnp.array([   0,   1,    2,   3,    4 ])
    x0 = 1

    nls(gaussian_decay, x0, args=(w, t))
    nls(exponential_decay, x0, args=(w, t))
    mle(gaussian_decay, x0, args=(w, t))
    mle(exponential_decay, x0, args=(w, t))

    #W = jnp.array([   1.0,    1.0,    0.0,    0.0,    1.0,    1.0,    0.0])
    #R = jnp.array([1436.0, 1162.0, 1782.0, 1708.0, 1297.0, 1342.0, 1813.0])
    W = jnp.array([   1.0,    1.0,    1.0,    1.0,    1.0,    1.0,    0.5])
    R = jnp.array([1446.0, 1687.0, 1798.0, 1860.0, 1917.0, 1756.0, 1805.0])
    tpr_ppf((W, R, dist_norm    ))
    tpr_ppf((W, R, dist_logistic))
    tpr_root((W, R, dist_norm    ))
    tpr_root((W, R, dist_logistic))
    tpr_nls((W, R, dist_norm    ))
    tpr_nls((W, R, dist_logistic))
    tpr_mle((W, R, dist_norm    ))
    tpr_mle((W, R, dist_logistic))
