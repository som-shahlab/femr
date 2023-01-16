from __future__ import annotations

import ctypes
import logging
import struct
import warnings
from functools import partial
from typing import Any, Dict, Optional, Sequence, Tuple, cast

import jax
import numpy as np
from jax import core, custom_vjp, debug, grad, lax, nn
from jax import numpy as jnp
from jax import value_and_grad, vmap, xla_computation
from jax.core import ConcreteArray, ShapedArray
from jax.interpreters import ad, batching, xla
from jax.lib import xla_client

from piton.extension.jax import (
    get_kernels,
    get_local_attention_data,
    get_local_attention_shape,
)

for name, value, platform in get_kernels():
    xla_client.register_custom_call_target(name, value, platform=platform)


def embedding_dot_fallback(a, b, indices):
    a_indices = indices[:, 0]
    b_indices = indices[:, 1]

    a_vals = jnp.take(
        a,
        a_indices,
        axis=0,
        mode="fill",
        fill_value=0,
    )

    b_vals = jnp.take(
        b,
        b_indices,
        axis=0,
        mode="fill",
        fill_value=0,
    )

    return jnp.multiply(a_vals, b_vals).sum(axis=1)


@partial(custom_vjp)
def embedding_dot(a, b, indices):
    return embedding_dot_forward_p.bind(a, b, indices)


def embedding_dot_fwd(a, b, indices):
    r = embedding_dot(a, b, indices)
    return r, (a, b, indices)


def embedding_dot_bwd(res, g):
    a, b, indices = res
    da, db = embedding_dot_backward_p.bind(a, b, indices, g)
    return (da, db, None)


embedding_dot.defvjp(embedding_dot_fwd, embedding_dot_bwd)

embedding_dot_forward_p = core.Primitive("embedding_dot_forward")


def embedding_dot_forward_abstract_eval(a, b, indices):
    assert a.dtype == b.dtype
    assert indices.dtype == jnp.uint32
    return ShapedArray((indices.shape[0],), a.dtype)


def convert_to_xla_shape(
    a: ShapedArray, force_dtype: Optional[Any] = None
) -> xla_client.Shape:
    return xla_client.Shape.array_shape(
        force_dtype or a.dtype, a.shape, list(reversed(range(len(a.shape))))
    )


def embedding_dot_forward_xla_translation(
    ctx: xla.TranslationContext,
    avals_in: Sequence[core.AbstractValue],
    avals_out: Sequence[core.AbstractValue],
    a: xla.XlaOp,
    b: xla.XlaOp,
    indices: xla.XlaOp,
) -> Sequence[xla.XlaOp]:
    computation = xla_computation(embedding_dot_fallback)(*avals_in)

    if ctx.platform == "cpu":
        res = xla_client.ops.Call(ctx.builder, computation, [a, b, indices])
        return [
            xla_client.ops.GetTupleElement(res, i)
            for i, _ in enumerate(avals_out)
        ]
    elif ctx.platform == "cuda":
        a_shape, b_shape, indices_shape = avals_in
        (result_shape,) = avals_out
        assert isinstance(a_shape, ShapedArray)
        assert isinstance(b_shape, ShapedArray)
        assert isinstance(indices_shape, ShapedArray)
        assert isinstance(result_shape, ShapedArray)
        assert a_shape.dtype == b_shape.dtype

        if a_shape.dtype == np.float32:
            name = "float_embedding_dot_forward"
        elif a_shape.dtype == np.float16:
            name = "half_embedding_dot_forward"
        else:
            raise ValueError("Could not natively suport dtype " + a_shape.dtype)

        return [
            xla_client.ops.CustomCallWithLayout(
                ctx.builder,
                name.encode("utf8"),
                operands=[a, b, indices],
                shape_with_layout=convert_to_xla_shape(result_shape),
                operand_shapes_with_layout=[
                    convert_to_xla_shape(val)
                    for val in (a_shape, b_shape, indices_shape)
                ],
                opaque=struct.pack(
                    "IIII",
                    a_shape.shape[1],
                    indices_shape.shape[0],
                    a_shape.shape[0],
                    b_shape.shape[0],
                ),
            )
        ]

    raise ValueError(
        "Unsupported platform; this must be either 'cpu' or 'gpu', got "
        + str(ctx.platform)
    )


embedding_dot_forward_p.def_abstract_eval(embedding_dot_forward_abstract_eval)
embedding_dot_forward_p.def_impl(embedding_dot_fallback)
xla.register_translation(
    embedding_dot_forward_p, embedding_dot_forward_xla_translation
)

embedding_dot_backward_p = core.Primitive("embedding_dot_backward")


def embedding_dot_backward_abstract_eval(a, b, indices, g):
    assert a.dtype == b.dtype
    assert g.shape == indices.shape[:1]
    assert indices.shape == (g.shape[0], 2)
    assert indices.dtype == jnp.uint32
    return (ShapedArray(a.shape, a.dtype), ShapedArray(b.shape, b.dtype))


def embedding_dot_backward_xla_translation(
    ctx: xla.TranslationContext,
    avals_in: Sequence[core.AbstractValue],
    avals_out: Sequence[core.AbstractValue],
    a: xla.XlaOp,
    b: xla.XlaOp,
    indices: xla.XlaOp,
    g: xla.XlaOp,
) -> Sequence[xla.XlaOp]:
    def helper(a, b, indices, g):
        return jnp.dot(embedding_dot_fallback(a, b, indices), g)

    computation = xla_computation(grad(helper, argnums=(0, 1)))(*avals_in)

    if ctx.platform == "cpu":
        res = xla_client.ops.Call(ctx.builder, computation, [a, b, indices, g])
        return [
            xla_client.ops.GetTupleElement(res, i)
            for i, _ in enumerate(avals_out)
        ]
    elif ctx.platform == "cuda":
        a_shape, b_shape, indices_shape, g_shape = avals_in
        (
            da_shape,
            db_shape,
        ) = avals_out
        assert isinstance(a_shape, ShapedArray)
        assert isinstance(b_shape, ShapedArray)
        assert isinstance(indices_shape, ShapedArray)
        assert isinstance(g_shape, ShapedArray)
        assert isinstance(da_shape, ShapedArray)
        assert isinstance(db_shape, ShapedArray)

        if a_shape.dtype == np.float32:
            name = "float_embedding_dot_backward"

            def _convert_back(op):
                return op

        elif a_shape.dtype == np.float16:
            name = "half_embedding_dot_backward"

            def _convert_back(op):
                return xla_client.ops.ConvertElementType(
                    op, xla_client.dtype_to_etype(np.float16)
                )

        else:
            raise ValueError("Could not natively suport dtype " + a_shape.dtype)

        res = xla_client.ops.CustomCallWithLayout(
            ctx.builder,
            name.encode("utf8"),
            operands=[a, b, indices, g],
            shape_with_layout=xla_client.Shape.tuple_shape(
                [
                    convert_to_xla_shape(
                        val, force_dtype=xla_client.dtype_to_etype(np.float32)
                    )
                    for val in (da_shape, db_shape)
                ]
            ),
            operand_shapes_with_layout=[
                convert_to_xla_shape(val)
                for val in (a_shape, b_shape, indices_shape, g_shape)
            ],
            opaque=struct.pack(
                "IIII",
                a_shape.shape[1],
                indices_shape.shape[0],
                a_shape.shape[0],
                b_shape.shape[0],
            ),
        )

        return [
            _convert_back(xla_client.ops.GetTupleElement(res, i))
            for i, _ in enumerate(avals_out)
        ]

    raise ValueError(
        "Unsupported platform; this must be either 'cpu' or 'gpu', got "
        + str(ctx.platform)
    )


embedding_dot_backward_p.def_abstract_eval(embedding_dot_backward_abstract_eval)
embedding_dot_backward_p.multiple_results = True
embedding_dot_backward_p.def_impl(grad(embedding_dot_fallback, argnums=(0, 1)))
xla.register_translation(
    embedding_dot_backward_p, embedding_dot_backward_xla_translation
)


def exp_mean_fallback(a, b, c):
    matrix = jnp.matmul(a, b.T)
    return jnp.exp2(matrix + c).astype(jnp.float32).mean().astype(a.dtype)


@partial(custom_vjp, nondiff_argnums=(2,))
def exp_mean(a, b, sparse_c):
    return exp_mean_fwd(a, b, sparse_c)[0]


@jax.jit
def exp_mean_fwd(a, b, sparse_c):
    r, da, db = exp_mean_p.bind(a, b, *sparse_c)
    return r, (a, da, db)


@jax.jit
def exp_mean_bwd(_sparse_c, res, g):
    (
        a,
        da,
        db,
    ) = res
    # Make sure we can safely do the multiplication
    assert da.dtype == g.dtype

    return ((da * g).astype(a.dtype), (db * g).astype(a.dtype))


exp_mean.defvjp(exp_mean_fwd, exp_mean_bwd)

exp_mean_p = core.Primitive("exp_mean")


def exp_mean_abstract_eval(a, b, offsets, defaults, indices, values):
    assert a.dtype == b.dtype
    assert len(a.shape) == len(b.shape) == 2
    assert (
        len(offsets.shape)
        == len(defaults.shape)
        == len(indices.shape)
        == len(values.shape)
        == 1
    )

    assert a.shape[1] == b.shape[1]

    assert offsets.shape == (a.shape[0] + 1,)
    assert defaults.shape == (a.shape[0],)

    assert indices.shape == values.shape
    assert len(indices.shape) == 1

    assert a.dtype == b.dtype
    assert offsets.dtype == jnp.uint32
    assert defaults.dtype == values.dtype == jnp.float32
    assert indices.dtype == jnp.uint32

    assert a.shape[0] % 32 == 0
    assert b.shape[0] % 32 == 0

    assert a.shape[1] % 32 == 0

    return (
        ShapedArray((), jnp.float32),
        ShapedArray(a.shape, jnp.float32),
        ShapedArray(b.shape, jnp.float32),
    )


def exp_mean_xla_translation(
    ctx: xla.TranslationContext,
    avals_in: Sequence[core.AbstractValue],
    avals_out: Sequence[core.AbstractValue],
    a: xla.XlaOp,
    b: xla.XlaOp,
    offsets: xla.XlaOp,
    defaults: xla.XlaOp,
    indices: xla.XlaOp,
    values: xla.XlaOp,
) -> Sequence[xla.XlaOp]:
    assert (isinstance(val, ShapedArray) for val in avals_in)
    assert (isinstance(val, ShapedArray) for val in avals_out)
    avals_in = cast(Sequence[ShapedArray], avals_in)
    avals_out = cast(Sequence[ShapedArray], avals_out)

    (
        a_shape,
        b_shape,
        offsets_shape,
        defaults_shape,
        indices_shape,
        values_shape,
    ) = avals_in

    (
        result_shape,
        da_shape,
        db_shape,
    ) = avals_out

    if ctx.platform == "cpu" or a_shape.dtype == jnp.float32:
        if ctx.platform == "cuda":
            warnings.warn(
                "Using an inefficient exp_sum mechanism", RuntimeWarning
            )

        c_shape = ShapedArray([a_shape.shape[0], b_shape.shape[0]], jnp.float32)

        value_and_grad_computation = xla_computation(
            value_and_grad(exp_mean_fallback, argnums=(0, 1))
        )(
            a_shape,
            b_shape,
            c_shape,
        )

        dense = xla_client.ops.CustomCall(
            ctx.builder,
            "convert_to_dense".encode("utf8"),
            [
                xla_client.ops.ConstantLiteral(ctx.builder, a_shape.shape[0]),
                xla_client.ops.ConstantLiteral(ctx.builder, b_shape.shape[0]),
                offsets,
                defaults,
                indices,
                values,
            ],
            convert_to_xla_shape(c_shape),
        )
        value_and_grad_res = xla_client.ops.Call(
            ctx.builder, value_and_grad_computation, [a, b, dense]
        )

        def _convert_to_full(op):
            return xla_client.ops.ConvertElementType(
                op, xla_client.dtype_to_etype(jnp.float32)
            )

        return [
            _convert_to_full(
                xla_client.ops.GetTupleElement(value_and_grad_res, i)
            )
            for i, _ in enumerate(avals_out)
        ]
    elif ctx.platform == "cuda":

        if a_shape.dtype == jnp.float16:
            name = "half_exp_mean_with_grad"
        else:
            raise ValueError(
                "Could not natively suport dtype " + str(a_shape.dtype)
            )

        m_shift, m_mult = get_shifts_and_mult(
            (b_shape.shape[0] + 8 * 16 - 1) // (8 * 16)
        )

        res = xla_client.ops.CustomCallWithLayout(
            ctx.builder,
            name.encode("utf8"),
            operands=[a, b, offsets, defaults, indices, values],
            shape_with_layout=xla_client.Shape.tuple_shape(
                [
                    convert_to_xla_shape(val)
                    for val in (result_shape, da_shape, db_shape)
                ]
            ),
            operand_shapes_with_layout=[
                convert_to_xla_shape(val) for val in avals_in
            ],
            opaque=struct.pack(
                "IIIII",
                a_shape.shape[0],
                b_shape.shape[0],
                a_shape.shape[1],
                m_shift,
                m_mult,
            ),
        )
        return [
            xla_client.ops.GetTupleElement(res, i)
            for i, _ in enumerate(avals_out)
        ]

    raise ValueError(
        "Unsupported platform; this must be either 'cpu' or 'gpu', got "
        + str(ctx.platform)
    )


exp_mean_p.def_abstract_eval(exp_mean_abstract_eval)
exp_mean_p.multiple_results = True
xla.register_translation(exp_mean_p, exp_mean_xla_translation)


def get_shifts_and_mult(n):
    if n == 0:
        raise ValueError("No inverse for 0")

    if n == 1:
        return 0, 0

    shift = 0
    while n > (1 << shift):
        shift += 1

    mult = ((2 ** (32 + shift) + n - 1) // n) % (2**32)
    return shift, mult


@partial(vmap, in_axes=(0, 0, 0, None, None))
def local_attention_fallback_single(
    queries, keys, values, length, attention_width
):
    logits = queries @ keys.T

    causal_mask = jnp.tri(N=queries.shape[0], k=0, dtype=jnp.bool_)
    local_mask = jnp.tri(
        N=queries.shape[0], k=-(attention_width + 1), dtype=jnp.bool_
    )

    indices = jnp.arange(queries.shape[0])
    row_indices = jnp.zeros_like(local_mask) + indices.reshape(
        (1, queries.shape[0])
    )
    col_indices = jnp.zeros_like(local_mask) + indices.reshape(
        (queries.shape[0], 1)
    )

    col_indices = col_indices.astype(jnp.uint32)
    row_indices = row_indices.astype(jnp.uint32)

    row_indices = jnp.bitwise_and(row_indices, length)
    col_indices = jnp.bitwise_and(col_indices, length)

    length_mask = row_indices == col_indices

    full_mask = causal_mask & (~local_mask) & length_mask

    logits = jnp.where(full_mask, logits, float("-inf"))

    attention = nn.softmax(logits / jnp.sqrt(keys.shape[1]))

    result = attention @ values

    return result


def local_attention_fallback(queries, keys, values, length, attention_width):
    result = local_attention_fallback_single(
        queries, keys, values, length, attention_width
    )
    attention_shape = tuple(
        get_local_attention_shape(
            queries.shape[0],
            queries.shape[1],
            queries.shape[2],
            attention_width,
        )
    )

    dummy_attention = jnp.zeros(attention_shape, queries.dtype)

    return dummy_attention, result


@partial(grad, argnums=(0, 1, 2))
def local_attention_backward_fallback(
    queries, keys, values, length, _attention, g, attention_width
):
    result = local_attention_fallback_single(
        queries, keys, values, length, attention_width
    )
    return (result * g).sum()


# The following is "technically" a memory leak
# We just have to hope that piton users don't keep their program open forever ...
_local_attention_data_cache: Dict[Tuple[int, int, int, int], Any] = {}


def _get_cached_local_attention_data(b, n, k, w):
    key = (b, n, k, w)
    if key not in _local_attention_data_cache:
        _local_attention_data_cache[key] = get_local_attention_data(b, n, k, w)

    ctypes.pythonapi.PyCapsule_GetPointer.restype = ctypes.c_void_p
    ctypes.pythonapi.PyCapsule_GetPointer.argtypes = [
        ctypes.py_object,
        ctypes.c_char_p,
    ]
    pointer = ctypes.pythonapi.PyCapsule_GetPointer(
        _local_attention_data_cache[key], None
    )

    result = struct.pack("P", pointer)

    return result


def batch_if_necessary(data):
    if len(data.shape) == 2:
        return jnp.expand_dims(data, 0)
    else:
        return data


def local_attention_impl(queries, keys, values, length, attention_width):
    queries = batch_if_necessary(queries)
    keys = batch_if_necessary(keys)
    values = batch_if_necessary(values)

    attention, result = local_attention_forward_p.bind(
        queries, keys, values, length, attention_width=attention_width
    )
    return result


def local_attention_fwd(queries, keys, values, length, attention_width):
    queries = batch_if_necessary(queries)
    keys = batch_if_necessary(keys)
    values = batch_if_necessary(values)

    attention, result = local_attention_forward_p.bind(
        queries, keys, values, length, attention_width=attention_width
    )
    # print(attention[:32, :32])
    # debug.breakpoint()
    res = (queries, keys, values, length, attention)
    return result, res


def local_attention_bwd(attention_width, res, g):
    # debug.print("Expected for backward {a} {b}", a=g, b=g.shape)
    queries, keys, values, length, attention = res

    dq, dk, dv = local_attention_backward_p.bind(
        queries,
        keys,
        values,
        length,
        attention,
        g,
        attention_width=attention_width,
    )
    return (dq, dk, dv, None)


local_attention = custom_vjp(local_attention_impl, nondiff_argnums=(4,))
local_attention.defvjp(local_attention_fwd, local_attention_bwd)

local_attention_forward_p = core.Primitive("local_attention_forward")


def local_attention_forward_abstract_eval(
    queries: ShapedArray,
    keys: ShapedArray,
    values: ShapedArray,
    length: ShapedArray,
    *,
    attention_width: int,
):
    assert len(queries.shape) == 3

    assert queries.dtype == keys.dtype == values.dtype
    assert queries.shape == keys.shape == values.shape

    assert attention_width % 16 == 0
    assert queries.shape[1] % 16 == 0
    assert queries.shape[2] % 16 == 0

    assert length.shape == tuple()
    assert length.dtype == jnp.uint32

    attention_shape = tuple(
        get_local_attention_shape(
            queries.shape[0],
            queries.shape[1],
            queries.shape[2],
            attention_width,
        )
    )

    logging.info("Using temp shape %s", str(attention_shape))

    return (
        ShapedArray(attention_shape, queries.dtype),
        ShapedArray(queries.shape, queries.dtype),
    )


def local_attention_forward_xla_translation(
    ctx: xla.TranslationContext,
    avals_in: Sequence[core.AbstractValue],
    avals_out: Sequence[core.AbstractValue],
    queries: xla.XlaOp,
    keys: xla.XlaOp,
    values: xla.XlaOp,
    length: xla.XlaOp,
    *,
    attention_width: int,
) -> Sequence[xla.XlaOp]:
    assert all(isinstance(a, ShapedArray) for a in avals_in)
    avals_in = cast(Sequence[ShapedArray], avals_in)
    assert all(isinstance(a, ShapedArray) for a in avals_out)
    avals_out = cast(Sequence[ShapedArray], avals_out)

    queries_shape, keys_shape, values_shape, length_shape = avals_in
    attention_shape, result_shape = avals_out

    if ctx.platform == "cpu" or (
        ctx.platform == "cuda" and queries_shape.dtype != jnp.float16
    ):
        computation = xla_computation(
            local_attention_fallback,
            static_argnums=(4,),
        )(*avals_in, attention_width)

        res = xla_client.ops.Call(
            ctx.builder, computation, [queries, keys, values, length]
        )

        if ctx.platform == "cuda":
            warnings.warn(
                "Using an inefficient CUDA attention mechanism", RuntimeWarning
            )

        return [
            xla_client.ops.GetTupleElement(res, i)
            for i, _ in enumerate(avals_out)
        ]
    elif ctx.platform == "cuda":
        if queries_shape.dtype == jnp.float16:
            name = "half_local_attention_forward"
        else:
            raise ValueError(
                "Could not natively suport dtype " + str(queries_shape.dtype)
            )

        opaque = _get_cached_local_attention_data(
            queries_shape.shape[0],
            queries_shape.shape[1],
            queries_shape.shape[2],
            attention_width,
        )

        res = xla_client.ops.CustomCallWithLayout(
            ctx.builder,
            name.encode("utf8"),
            operands=[queries, keys, values, length],
            shape_with_layout=xla_client.Shape.tuple_shape(
                [convert_to_xla_shape(res) for res in avals_out]
            ),
            operand_shapes_with_layout=[
                convert_to_xla_shape(val) for val in avals_in
            ],
            opaque=opaque,
        )

        return [
            xla_client.ops.GetTupleElement(res, i)
            for i, _ in enumerate(avals_out)
        ]

    raise ValueError(
        "Unsupported platform; this must be either 'cpu' or 'gpu', got "
        + str(ctx.platform)
    )


local_attention_forward_p.def_abstract_eval(
    local_attention_forward_abstract_eval
)
local_attention_forward_p.def_impl(local_attention_fallback)
local_attention_forward_p.multiple_results = True
xla.register_translation(
    local_attention_forward_p, local_attention_forward_xla_translation
)

local_attention_backward_p = core.Primitive("local_attention_backward")


def local_attention_backward_abstract_eval(
    queries: ShapedArray,
    keys: ShapedArray,
    values: ShapedArray,
    length: ShapedArray,
    attention: ShapedArray,
    g: ShapedArray,
    attention_width: int,
):
    assert length.shape == tuple()
    assert length.dtype == jnp.uint32

    assert queries.dtype == keys.dtype == values.dtype == g.dtype
    assert queries.shape == keys.shape == values.shape == g.shape

    assert attention.dtype == queries.dtype

    assert queries.shape[2] == 64

    assert queries.shape[1] % 16 == 0
    assert queries.shape[2] % 16 == 0

    attention_shape = tuple(
        get_local_attention_shape(
            queries.shape[0],
            queries.shape[1],
            queries.shape[2],
            attention_width,
        )
    )
    assert attention_shape == attention.shape

    return (queries, keys, values)


def local_attention_backward_xla_translation(
    ctx: xla.TranslationContext,
    avals_in: Sequence[core.AbstractValue],
    avals_out: Sequence[core.AbstractValue],
    queries: xla.XlaOp,
    keys: xla.XlaOp,
    values: xla.XlaOp,
    length: xla.XlaOp,
    attention: xla.XlaOp,
    g: xla.XlaOp,
    *,
    attention_width: int,
) -> Sequence[xla.XlaOp]:
    assert all(isinstance(a, ShapedArray) for a in avals_in)
    avals_in = cast(Sequence[ShapedArray], avals_in)

    assert all(isinstance(a, ShapedArray) for a in avals_out)
    avals_out = cast(Sequence[ShapedArray], avals_out)

    (
        queries_shape,
        keys_shape,
        values_shape,
        length_shape,
        attention_shape,
        g_shape,
    ) = avals_in
    dq, dk, dv = avals_out

    if ctx.platform == "cpu":
        computation = xla_computation(
            local_attention_backward_fallback, static_argnums=(6,)
        )(*avals_in, attention_width)

        res = xla_client.ops.Call(
            ctx.builder,
            computation,
            [queries, keys, values, length, attention, g],
        )
        return [
            xla_client.ops.GetTupleElement(res, i)
            for i, _ in enumerate(avals_out)
        ]
    elif ctx.platform == "cuda":
        if queries_shape.dtype == np.float16:
            name = "half_local_attention_backward"
        else:
            raise ValueError(
                "Could not natively suport dtype " + avals_in[0].dtype
            )

        opaque = _get_cached_local_attention_data(
            queries_shape.shape[0],
            queries_shape.shape[1],
            queries_shape.shape[2],
            attention_width,
        )

        res = xla_client.ops.CustomCallWithLayout(
            ctx.builder,
            name.encode("utf8"),
            operands=[queries, keys, values, length, attention, g],
            shape_with_layout=xla_client.Shape.tuple_shape(
                [convert_to_xla_shape(avals_out[0])]
                + [
                    convert_to_xla_shape(
                        val, force_dtype=xla_client.dtype_to_etype(jnp.float32)
                    )
                    for val in avals_out[1:]
                ]
            ),
            operand_shapes_with_layout=[
                convert_to_xla_shape(val) for val in avals_in
            ],
            opaque=opaque,
        )

        def _convert_back(op):
            return xla_client.ops.ConvertElementType(
                op, xla_client.dtype_to_etype(jnp.float16)
            )

        return [
            _convert_back(xla_client.ops.GetTupleElement(res, i))
            if i > 0
            else xla_client.ops.GetTupleElement(res, i)
            for i, _ in enumerate(avals_out)
        ]

    raise ValueError(
        "Unsupported platform; this must be either 'cpu' or 'gpu', got "
        + str(ctx.platform)
    )


local_attention_backward_p.def_abstract_eval(
    local_attention_backward_abstract_eval
)
local_attention_backward_p.multiple_results = True
local_attention_backward_p.def_impl(local_attention_backward_fallback)
xla.register_translation(
    local_attention_backward_p, local_attention_backward_xla_translation
)
