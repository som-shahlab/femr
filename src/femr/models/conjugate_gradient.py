import functools
import os
from typing import TypeVar

import haiku as hk
import jax
import jax.numpy as jnp
import jax.scipy.optimize
import numpy as np
import optax
import sklearn.metrics

os.environ["JAX_NUMPY_RANK_PROMOTION"] = "raise"


T = TypeVar("T")


@functools.partial(jax.jit, donate_argnums=(0, 1, 2), static_argnames=("compute_hessian", "compute_grad"))
def conjugate_gradient(last_w, last_gradient, last_u, data, l2, compute_hessian, compute_grad):
    g = compute_grad(last_w, data, l2=l2)
    if last_gradient is None:
        u = g
    else:
        delta = g - last_gradient
        beta = jnp.dot(g, delta) / jnp.dot(last_u, delta)
        u = g - last_u * beta
    w = last_w - (jnp.dot(g, u) / compute_hessian(last_w, u, data, l2=l2)) * u
    return w, g, u


def compute_logistic_loss(beta, data, l2=0):
    reprs = data["reprs"]
    labels = data["labels"]

    hazards = jnp.dot(reprs, beta)

    return (
        optax.sigmoid_binary_cross_entropy(hazards, labels).mean(dtype=jnp.float32) + 0.5 * l2 * (beta[:-1] ** 2).sum()
    )


def compute_logistic_grad(beta, data, l2=0):
    reprs = data["reprs"]
    labels = data["labels"]

    hazards = jnp.dot(reprs, beta)

    assert hazards.shape == labels.shape

    logit = jax.nn.sigmoid(hazards)
    inverse_logit = jax.nn.sigmoid(-hazards)

    weights = -labels * inverse_logit + (1 - labels) * logit
    weights = jnp.expand_dims(weights, axis=-1)

    mask = beta.at[-1].set(0)

    return (weights * reprs).mean(axis=0, dtype=jnp.float32) + l2 * mask


def compute_logistic_hessian(beta, u, data, l2=0):
    reprs = data["reprs"]

    hazards = jnp.dot(reprs, beta)

    logit = jax.nn.sigmoid(hazards)
    inverse_logit = jax.nn.sigmoid(-hazards)

    factor = jnp.dot(reprs, u) ** 2

    val = u.at[-1].set(0)

    return (factor * logit * inverse_logit).mean(axis=0, dtype=jnp.float32) + l2 * jnp.dot(val, val)


def compute_survival_loss(beta, data, l2=0):
    reprs = data["reprs"]
    log_times = data["log_times"]
    is_events = data["is_events"]

    hazards = jnp.dot(reprs, beta)

    assert hazards.shape == is_events.shape

    event_loss = jnp.log(2) * (hazards * is_events).mean(dtype=jnp.float32)
    event_loss = -event_loss

    survival_loss = jnp.exp2(hazards + log_times).mean(dtype=jnp.float32)

    return survival_loss + event_loss + 0.5 * l2 * (beta[:-1] ** 2).sum()


def compute_survival_grad(beta, data, l2=0):
    reprs = data["reprs"]
    log_times = data["log_times"]
    is_events = data["is_events"]
    hazards = jnp.dot(reprs, beta)

    assert hazards.shape == is_events.shape

    event_loss = jnp.log(2) * (reprs * is_events.reshape(is_events.shape + (1,))).mean(axis=(0, 1), dtype=jnp.float32)
    event_loss = -event_loss

    survival_loss = jnp.log(2) * (jnp.exp2(hazards + log_times).reshape(hazards.shape + (1,)) * reprs).mean(
        axis=(0, 1), dtype=jnp.float32
    )

    mask = beta.at[-1].set(0)

    return survival_loss + event_loss + l2 * mask


def compute_survival_hessian(beta, u, data, l2=0):
    reprs = data["reprs"]
    log_times = data["log_times"]
    is_events = data["is_events"]

    hazards = jnp.dot(reprs, beta)
    factor = jnp.dot(reprs, u) ** 2

    assert hazards.shape == is_events.shape

    scale = jnp.log(2) * jnp.log(2) * jnp.exp2(hazards + log_times)

    survival_loss = (scale * factor).mean(dtype=jnp.float32)

    val = u.at[-1].set(0)

    return survival_loss + l2 * jnp.dot(val, val)


def train_logistic_regression(train_reprs, train_labels, val_reprs, val_labels):
    train_data = {"reprs": jnp.array(train_reprs), "labels": jnp.array(train_labels)}
    val_data = {"reprs": jnp.array(val_reprs), "labels": jnp.array(val_labels)}

    rng = jax.random.PRNGKey(42)

    beta = jnp.zeros(train_reprs.shape[-1], dtype=jnp.float16)

    best_train_loss = None
    best_val_loss = None
    best_beta = None
    best_l = None

    start_l, end_l = -5, 5
    for l_exp in np.linspace(end_l, start_l, num=20):
        if l_exp == start_l:
            l2 = 0
        else:
            l2 = 10 ** (l_exp)

        g = None
        u = None
        while True:
            beta, g, u = conjugate_gradient(
                beta, g, u, train_data, l2, compute_hessian=compute_logistic_hessian, compute_grad=compute_logistic_grad
            )
            grad_norm = jnp.linalg.norm(g, ord=2)

            if grad_norm < 0.0001:
                break

        train_hazards = jnp.dot(train_reprs, beta)
        val_hazards = jnp.dot(val_reprs, beta)

        train_loss = sklearn.metrics.roc_auc_score(train_labels, train_hazards)
        val_loss = sklearn.metrics.roc_auc_score(val_labels, val_hazards)

        print(l2, train_loss, val_loss)

        if best_val_loss is None or val_loss > best_val_loss:
            best_train_loss, best_val_loss, best_beta, best_l = train_loss, val_loss, np.array(beta), l2

    return best_beta
