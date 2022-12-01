from __future__ import annotations

import random as pyrandom

import jax.numpy as jnp
import numpy as np
from jax import device_put, devices, grad, jit, random, value_and_grad, vmap

from piton.jax import (
    embedding_dot,
    embedding_dot_fallback,
    exp_mean,
    exp_mean_fallback,
    get_shifts_and_mult,
    local_attention,
    local_attention_fallback,
)

jnp.set_printoptions(linewidth=200)


def get_result_and_grad(device, dtype):
    key = random.PRNGKey(12352)

    key, k1, k2, k3 = random.split(key, 4)

    embedding1 = device_put(random.normal(k1, (30, 70), dtype=dtype), device)
    embedding2 = device_put(random.normal(k2, (60, 70), dtype=dtype), device)

    indices = device_put(
        random.randint(k3, (50, 2), minval=0, maxval=50), device
    ).astype(jnp.uint32)

    def total_embedding_fallback(embedding1, embedding2, indices):
        return embedding_dot_fallback(embedding1, embedding2, indices).sum()

    def total_embedding(embedding1, embedding2, indices):
        return embedding_dot(embedding1, embedding2, indices).sum()

    fallback = jit(value_and_grad(total_embedding_fallback, argnums=(0, 1)))

    optimized = jit(value_and_grad(total_embedding, argnums=(0, 1)))

    val1, (da1, db1) = fallback(embedding1, embedding2, indices)

    val2, (da2, db2) = optimized(embedding1, embedding2, indices)

    def assert_equal(a, b):
        print(a, b)
        assert a.shape == b.shape
        assert a.dtype == b.dtype
        assert a.device() == b.device() == device
        assert jnp.allclose(a, b, rtol=1e-2, atol=1e-2)

    assert_equal(val1, val2)
    assert_equal(da1, da2)
    assert_equal(db1, db2)


def test_cpu_implementation():
    get_result_and_grad(devices("cpu")[0], np.float64)
    get_result_and_grad(devices("cpu")[0], np.float32)
    get_result_and_grad(devices("cpu")[0], np.float16)


def test_gpu_implementation():
    get_result_and_grad(devices("gpu")[0], np.float32)
    get_result_and_grad(devices("gpu")[0], np.float16)


def exp_mean_helper(dtype, is_zero, device):
    np.random.seed(341213)
    pyrandom.seed(12315)

    N = 32 * 8
    K = 32 * 32
    M = 32 * 16

    if is_zero:
        a = np.zeros((N, K))
        b = np.zeros((M, K))
    else:
        a = np.random.uniform(size=(N, K)) / 8
        b = np.random.uniform(size=(M, K)) / 16

    c = np.zeros((N, M))

    indptr = [0]
    defaults = []
    indices = []
    values = []
    for i in range(N):
        val = pyrandom.random()
        defaults.append(val)
        c[i, :] = val

        for j in range(M):
            if pyrandom.random() < 0.1:
                other_val = 1 + pyrandom.random()
                c[i, j] = other_val
                indices.append(j)
                values.append(other_val)
        indptr.append(len(indices))

    if False:
        print(indptr)

    a = device_put(jnp.array(a, dtype=dtype), device)
    b = device_put(jnp.array(b, dtype=dtype), device)
    c = device_put(jnp.array(c, dtype=dtype), device)

    if False:
        print(c)

    indptr = device_put(jnp.array(indptr, dtype=np.uint32), device)
    defaults = device_put(jnp.array(defaults, dtype=np.float32), device)
    indices = device_put(jnp.array(indices, dtype=np.uint32), device)
    values = device_put(jnp.array(values, dtype=np.float32), device)

    sparse_c = (indptr, defaults, indices, values)

    v1 = exp_mean_fallback(a, b, c)
    v1_true = exp_mean_fallback(
        a.astype(jnp.float32), b.astype(jnp.float32), c.astype(jnp.float32)
    )
    v2 = jit(exp_mean)(a, b, sparse_c)

    if False:
        print(v1, v1_true, v2)

    assert jnp.allclose(v1, v2, rtol=1e-2, atol=1e-2)

    da1, db1 = grad(exp_mean_fallback, argnums=(0, 1))(a, b, c)
    da1_true, db1_true = grad(exp_mean_fallback, argnums=(0, 1))(
        a.astype(jnp.float32), b.astype(jnp.float32), c.astype(jnp.float32)
    )
    da2, db2 = jit(grad(exp_mean, argnums=(0, 1)))(a, b, sparse_c)

    if False:
        print(da1_true)
        print(da1)
        print(da2)

        print("-------")

        print(db1_true)
        print(db1)
        print(db2)

    assert jnp.allclose(da1, da2, rtol=1e-2, atol=1e-2)
    assert jnp.allclose(db1, db2, rtol=1e-2, atol=1e-2)


def test_exp_mean_simple_cpu():
    cpu_device = devices("cpu")[0]
    exp_mean_helper(dtype=np.float32, is_zero=True, device=cpu_device)


def test_exp_mean_complex_cpu():
    cpu_device = devices("cpu")[0]
    exp_mean_helper(dtype=np.float32, is_zero=False, device=cpu_device)


def test_exp_mean_simple_gpu():
    gpu_device = devices("gpu")[0]
    exp_mean_helper(dtype=np.float16, is_zero=True, device=gpu_device)


def test_exp_mean_complex_gpu():
    gpu_device = devices("gpu")[0]
    exp_mean_helper(dtype=np.float16, is_zero=False, device=gpu_device)


def divide(x, val, shift, mult):
    if (shift, mult) == (0, 0):
        assert val == 1
        # Divide by 1
        return x
    q = (x * mult) >> 32
    t = ((x - q) >> 1) + q
    return t >> (shift - 1)


def modulus(x, val, shift, mult):
    divisor = divide(x, val, shift, mult)
    return x - divisor * val


def test_shift_and_inverse():
    pyrandom.seed(123567)
    for i_n in range(100):
        n = pyrandom.randint(1, 1000)
        if i_n == 0:
            n = 1
        shift, mult = get_shifts_and_mult(n)

        for i_m in range(100):
            m = pyrandom.randint(0, 10000)
            if i_m <= 1:
                m = i_m
            assert (m // n) == divide(m, n, shift, mult)
            assert (m % n) == modulus(m, n, shift, mult)


def test_layout_strategy():
    pyrandom.seed(123125)
    for n in range(1, 100):
        for i_m in range(10):
            m = pyrandom.randint(1, 100)
            if i_m == 0:
                m = 1

            shift, mult = get_shifts_and_mult(m)
            seen = set()

            for i in range(n * m):
                x = divide(i, m, shift, mult)
                y = modulus(x + i, m, shift, mult)
                assert 0 <= x < n
                assert 0 <= y < m
                seen.add((x, y))

            assert len(seen) == n * m


def local_attention_helper(dtype, device):
    B = 4
    N = 512
    K = 64
    W = 512

    k = random.PRNGKey(123534)
    k1, k2, k3, k4 = random.split(k, 4)

    queries = random.normal(k1, shape=(2, B * N, K))
    keys = random.normal(k2, shape=(2, B * N, K))
    values = random.normal(k3, shape=(2, B * N, K))

    queries, keys, values = [
        device_put(jnp.array(a, dtype=dtype), device)
        for a in (queries, keys, values)
    ]

    N_arg = ~jnp.array(N - 1).astype(jnp.uint32)
    print(N_arg)

    _, res_true = local_attention_fallback(queries, keys, values, N_arg, W)

    res_est_f = local_attention(queries, keys, values, N_arg, W)

    res_est = jit(local_attention, static_argnames={"attention_width"})(
        queries, keys, values, N_arg, W
    )

    print(res_true.shape)
    print(res_est_f.shape)
    print(res_est.shape)

    assert jnp.allclose(res_true, res_est_f, atol=1e-2, rtol=1e-3)
    assert jnp.allclose(res_true, res_est, atol=1e-2, rtol=1e-3)

    valid = random.normal(k4, shape=(2, B * N, K))

    def helper(func):
        def h(queries, keys, values, length, attention_width):
            return (
                func(queries, keys, values, length, attention_width) * valid
            ).sum()

        return grad(
            h,
            argnums=(
                0,
                1,
                2,
            ),
        )

    dq1, dk1, dv1 = helper(
        lambda *args, **kwargs: local_attention_fallback(*args, **kwargs)[1]
    )(queries, keys, values, N_arg, attention_width=W)

    dq2_f, dk2_f, dv2_f = helper(local_attention)(
        queries, keys, values, N_arg, attention_width=W
    )

    dq2, dk2, dv2 = jit(
        helper(local_attention),
        static_argnames={"attention_width"},
    )(queries, keys, values, N_arg, attention_width=W)

    print("True dv", dv1)
    print("Est b dv", dv2_f)
    print("Est dv", dv2)
    assert jnp.allclose(dv1, dv2_f, atol=1e-2, rtol=1e-3)
    assert jnp.allclose(dv1, dv2, atol=1e-2, rtol=1e-3)

    print("True dq", dq1)
    print("Est dq", dq2)
    assert jnp.allclose(dq1, dq2, atol=1e-2, rtol=1e-3)

    print("True dk", dk1)
    print("Est dk", dk2)
    assert jnp.allclose(dk1, dk2, atol=1e-2, rtol=1e-3)


def test_local_attention_simple_cpu():
    cpu_device = devices("cpu")[0]
    local_attention_helper(dtype=np.float32, device=cpu_device)


def test_local_attention_simple_gpu():
    gpu_device = devices("gpu")[0]
    local_attention_helper(dtype=np.float16, device=gpu_device)
