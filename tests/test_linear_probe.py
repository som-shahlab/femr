from __future__ import annotations

import random as pyrandom

import pytest

try:
    import jax
    import jax.numpy as jnp
    import numpy as np
    from jax import device_put, devices, grad, jit, random, value_and_grad

    from femr.models.linear_probe import *

except ImportError:
    pytest.skip("jax package not available?", allow_module_level=True)

def test_logistic_model():
    rng = jax.random.PRNGKey(42)

    keys = jax.random.split(rng, 5)

    truth_beta = jax.random.uniform(keys[0], shape=(10,))
    beta = jax.random.uniform(keys[1], shape=(10,))
    reprs = jax.random.uniform(keys[2], shape=(2000, 10))

    hazards = jnp.dot(reprs, truth_beta)
    probs = jax.nn.sigmoid(hazards)

    labels = jax.random.bernoulli(keys[3], probs).astype(jnp.float16)
    
    u = jax.random.uniform(keys[4], shape=beta.shape)

    data = {'reprs': reprs, 'labels': labels}


    for l in [0, 3]:
        print(l)

        my_grad = compute_logistic_grad(beta, data, l=l)
        auto_grad = jax.grad(compute_logistic_loss)(beta, data, l=l)
        assert jnp.allclose(my_grad, auto_grad)


        auto_hessian = jax.hessian(compute_logistic_loss)(beta, data, l=l)
        auto_val = u @ (auto_hessian @ u.T)
        my_val = compute_logistic_hessian(beta, u, data, l=l)
        assert jnp.allclose(my_val, auto_val)
    
    random_loss = compute_logistic_loss(beta, data)
    truth_loss = compute_logistic_loss(truth_beta, data)

    assert truth_loss < random_loss

    g = None
    u = None
    while True:
        beta, g, u = conjugate_gradient(beta, g, u, data, 0, compute_hessian=compute_logistic_hessian, compute_grad=compute_logistic_grad)
        grad_norm = jnp.linalg.norm(g, ord=2)

        if grad_norm < 0.0001:
            break

    best_loss = compute_logistic_loss(beta, data)

    assert best_loss <= truth_loss
