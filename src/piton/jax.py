"""
A collection of custom jax primitives that are necessary for piton's neural network models to operate.
"""

from __future__ import annotations

import ctypes
import struct
import warnings
from functools import partial
from typing import Any, Dict, Optional, Sequence, Tuple, cast

import jax
import numpy as np
from jax import core, grad, nn
from jax import numpy as jnp
from jax import value_and_grad, vmap, xla_computation
from jax.interpreters import xla
from jax.lib import xla_client

import piton.extension.jax

# Globally register all of our custom operators
for name, value, platform in piton.extension.jax.get_kernels():
    jax.lib.xla_client.register_custom_call_target(name, value, platform=platform)

# Per the jax documentation, we currently don't have good typing for arrays
Array = Any


def _convert_to_xla_shape(a: jax.core.ShapedArray, force_dtype: Optional[Any] = None) -> xla_client.Shape:
    """A helper function for converting between a JAX jax.core.ShapedArray and an xla Shape."""
    return xla_client.Shape.array_shape(force_dtype or a.dtype, a.shape, list(reversed(range(len(a.shape)))))


def _assert_and_cast_to_shaped(
    shapes: Sequence[jax.core.AbstractValue],
) -> Sequence[jax.core.ShapedArray]:
    """Assert that a sequence of abstract tokens are actually shaped arrays."""
    assert all(isinstance(a, jax.core.ShapedArray) for a in shapes)
    return cast(Sequence[jax.core.ShapedArray], shapes)


@partial(
    jax.custom_vjp,
    nondiff_argnums=(2,),
)
def gather_scatter_add(a: Array, indices: Array, output_dim: int) -> Array:
    """Perform a combined gather / scatter add operation.

    a is the tensor that we perform reads from.

    indices is an array of shape (N, 2) for N indices.
    This operator reads from the index specified in indices[:, 0] and writes them to indices[:, 1].

    output_dim is the size of the ouptut array.

    Note that this operator requires that indices must be sorted in the order of the write indices (indices[:, 1])
    A secondary note: This operator is non-deterministic. The additionals will be performed in an arbitrary order.
    """
    return gather_scatter_add_fwd(a, indices, output_dim)[0]


def gather_scatter_add_fallback(a: Array, indices: Array, output_dim: int) -> Array:
    """The python fallback implementation of gather scatter add.

    See gather_scatter for the documentation.
    """
    a_indices = indices[:, 0]
    b_indices = indices[:, 1]

    a_vals = a.at[a_indices, :].get(mode="fill", fill_value=0)

    result = jnp.zeros((output_dim, a.shape[1]), dtype=a.dtype)

    return result.at[b_indices, :].add(a_vals, mode="drop")


def gather_scatter_add_fwd(a: Array, indices: Array, output_dim: int) -> Tuple[Array, Tuple[int, Array]]:
    """The forward pass for gather / scatter"""
    r = gather_scatter_add_p.bind(a, indices, output_dim=output_dim)
    return r, (a.shape[0], indices)


def gather_scatter_add_bwd(_output_dim: int, res: Tuple[int, Array], g: Array) -> Tuple[Array, None]:
    """The backward pass for gather / scatter

    The basic idea here is that we transpose the indices and apply it in reverse to get the backward pass.
    """

    a_dim, indices = res

    indices_a = indices[:, 0]
    indices_b = indices[:, 1]

    sort_indices = jnp.argsort(indices_a)

    # Note that these are transposed
    trans_indices = jnp.stack((indices_b, indices_a), axis=-1)
    trans_indices = trans_indices[sort_indices]

    r = gather_scatter_add_p.bind(g, trans_indices, output_dim=a_dim)

    return (r, None)


gather_scatter_add.defvjp(gather_scatter_add_fwd, gather_scatter_add_bwd)

# The actual primitive used to implement this operation
gather_scatter_add_p = jax.core.Primitive("gather_scatter_add_p")


def gather_scatter_add_p_abstract_eval(a: Array, indices: Array, output_dim: int) -> jax.core.ShapedArray:
    """The abstract shape computation for gather_scatter_p primitive."""
    # Our CUDA kernel currently assume that the inner size is a factor of 32.
    # TODO: Remove this requirement
    assert a.shape[1] % 32 == 0

    # Our CUDA kernel is currently only implemented for the following types
    # Note that this kernel is sorta incoherent for float16 anyways ...
    assert a.dtype == jnp.float32
    assert indices.dtype == jnp.uint32

    return jax.core.ShapedArray(
        (
            output_dim,
            a.shape[1],
        ),
        a.dtype,
    )


def gather_scatter_add_p_xla_translation(
    ctx: jax.interpreters.xla.TranslationContext,
    avals_in: Sequence[jax.core.AbstractValue],
    avals_out: Sequence[jax.core.AbstractValue],
    a: jax.interpreters.xla.XlaOp,
    indices: jax.interpreters.xla.XlaOp,
    *,
    output_dim: int,
) -> Sequence[jax.interpreters.xla.XlaOp]:
    """Actually compute gather / scatter in hardware."""
    avals_in = _assert_and_cast_to_shaped(avals_in)
    avals_out = _assert_and_cast_to_shaped(avals_out)

    (
        input_shape,
        indices_shape,
    ) = avals_in

    (result_shape,) = avals_out

    if ctx.platform == "cpu":
        # For our CPU implementation, we simply compile our fallback
        computation = jax.xla_computation(gather_scatter_add_fallback, static_argnums=(2,))(*avals_in, output_dim)

        res = xla_client.ops.Call(ctx.builder, computation, [a, indices])
        return [xla_client.ops.GetTupleElement(res, i) for i, _ in enumerate(avals_out)]

    elif ctx.platform == "cuda":
        # For our GPU implementation, we actually call out to CUDA code
        assert result_shape.shape[0] == output_dim
        assert result_shape.shape[1] == input_shape.shape[1]

        assert input_shape.dtype == result_shape.dtype

        if input_shape.dtype == np.float32:
            name = "float_gather_scatter"
        else:
            raise ValueError("Could not natively suport dtype " + str(input_shape.dtype))

        return [
            xla_client.ops.CustomCallWithLayout(
                ctx.builder,
                name.encode("utf8"),
                operands=[a, indices],
                shape_with_layout=_convert_to_xla_shape(result_shape),
                operand_shapes_with_layout=[_convert_to_xla_shape(val) for val in (input_shape, indices_shape)],
                opaque=struct.pack(
                    "IIII",
                    indices_shape.shape[0],
                    input_shape.shape[1],
                    input_shape.shape[0],
                    result_shape.shape[0],
                ),
            )
        ]

    raise ValueError("Unsupported platform; this must be either 'cpu' or 'gpu', got " + str(ctx.platform))


gather_scatter_add_p.def_abstract_eval(gather_scatter_add_p_abstract_eval)
gather_scatter_add_p.def_impl(gather_scatter_add_fallback)
xla.register_translation(gather_scatter_add_p, gather_scatter_add_p_xla_translation)


@partial(jax.custom_vjp)
def embedding_dot(a: Array, b: Array, indices: Array) -> Array:
    """Perform a dot product between two embedding layers over the provided indices.

    a is a 2-d tensor of shape (A, K).
    b is a 2-d tensor of shape (B, K).

    K is the length of the embedding dimension.

    indices is a 2-d tensor of shape (N, 2), such that indices[:, 0] are
    indices into A and indices[:, 1] are indices into B.

    This returns a (N,) shaped tensor that contains the dot products for each row in indices.

    Indices out of range are dropped and the ouput is padded with zeros.
    """
    return embedding_dot_fwd(a, b, indices)[0]


def embedding_dot_fallback(a: Array, b: Array, indices: Array):
    """The fallback implementation for embedding_dot."""
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


def embedding_dot_fwd(a: Array, b: Array, indices: Array) -> Tuple[Array, Tuple[Array, Array, Array]]:
    """The forward pass for embedding dot."""
    r = embedding_dot_forward_p.bind(a, b, indices)
    return r, (a, b, indices)


def embedding_dot_bwd(res: Tuple[Array, Array, Array], g: Array) -> Tuple[Array, Array, None]:
    """The backward pass for embedding dot."""
    a, b, indices = res
    da, db = embedding_dot_backward_p.bind(a, b, indices, g)
    return (da, db, None)


embedding_dot.defvjp(embedding_dot_fwd, embedding_dot_bwd)

embedding_dot_forward_p = jax.core.Primitive("embedding_dot_forward")


def embedding_dot_forward_p_abstract_eval(
    a: jax.core.ShapedArray,
    b: jax.core.ShapedArray,
    indices: jax.core.ShapedArray,
):
    """The abstract shape for the forward primitive."""
    assert a.dtype == b.dtype
    assert indices.dtype == jnp.uint32
    return jax.core.ShapedArray((indices.shape[0],), a.dtype)


def embedding_dot_forward_p_xla_translation(
    ctx: jax.interpreters.xla.TranslationContext,
    avals_in: Sequence[jax.core.AbstractValue],
    avals_out: Sequence[jax.core.AbstractValue],
    a: jax.interpreters.xla.XlaOp,
    b: jax.interpreters.xla.XlaOp,
    indices: jax.interpreters.xla.XlaOp,
) -> Sequence[jax.interpreters.xla.XlaOp]:
    """The actual hardware implementation of embedding_dot_forward_p."""
    avals_in = _assert_and_cast_to_shaped(avals_in)
    avals_out = _assert_and_cast_to_shaped(avals_out)

    a_shape, b_shape, indices_shape = avals_in
    (result_shape,) = avals_out

    if ctx.platform == "cpu":
        computation = jax.xla_computation(embedding_dot_fallback)(*avals_in)
        res = xla_client.ops.Call(ctx.builder, computation, [a, b, indices])
        return [xla_client.ops.GetTupleElement(res, i) for i, _ in enumerate(avals_out)]
    elif ctx.platform == "cuda":
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
                shape_with_layout=_convert_to_xla_shape(result_shape),
                operand_shapes_with_layout=[_convert_to_xla_shape(val) for val in (a_shape, b_shape, indices_shape)],
                opaque=struct.pack(
                    "IIII",
                    a_shape.shape[1],
                    indices_shape.shape[0],
                    a_shape.shape[0],
                    b_shape.shape[0],
                ),
            )
        ]

    raise ValueError("Unsupported platform; this must be either 'cpu' or 'gpu', got " + str(ctx.platform))


embedding_dot_forward_p.def_abstract_eval(embedding_dot_forward_p_abstract_eval)
embedding_dot_forward_p.def_impl(embedding_dot_fallback)
xla.register_translation(embedding_dot_forward_p, embedding_dot_forward_p_xla_translation)

embedding_dot_backward_p = core.Primitive("embedding_dot_backward")


def embedding_dot_backward_p_abstract_eval(
    a: jax.core.ShapedArray,
    b: jax.core.ShapedArray,
    indices: jax.core.ShapedArray,
    g: jax.core.ShapedArray,
):
    """The abstract shape for embedding_dot_backward_p."""
    assert a.dtype == b.dtype
    assert g.shape == indices.shape[:1]
    assert indices.shape == (g.shape[0], 2)
    assert indices.dtype == jnp.uint32
    return (
        jax.core.ShapedArray(a.shape, a.dtype),
        jax.core.ShapedArray(b.shape, b.dtype),
    )


def embedding_dot_backward_p_xla_translation(
    ctx: xla.TranslationContext,
    avals_in: Sequence[core.AbstractValue],
    avals_out: Sequence[core.AbstractValue],
    a: xla.XlaOp,
    b: xla.XlaOp,
    indices: xla.XlaOp,
    g: xla.XlaOp,
) -> Sequence[xla.XlaOp]:
    """The actual hardware definion of the embedding dot backward pass."""
    avals_in = _assert_and_cast_to_shaped(avals_in)
    avals_out = _assert_and_cast_to_shaped(avals_out)

    a_shape, b_shape, indices_shape, g_shape = avals_in
    (
        da_shape,
        db_shape,
    ) = avals_out

    if ctx.platform == "cpu":

        def helper(a, b, indices, g):
            return jnp.dot(embedding_dot_fallback(a, b, indices), g)

        computation = xla_computation(grad(helper, argnums=(0, 1)))(*avals_in)
        res = xla_client.ops.Call(ctx.builder, computation, [a, b, indices, g])
        return [xla_client.ops.GetTupleElement(res, i) for i, _ in enumerate(avals_out)]
    elif ctx.platform == "cuda":
        if a_shape.dtype == np.float32:
            name = "float_embedding_dot_backward"

            def _convert_back(op):
                return op

        elif a_shape.dtype == np.float16:
            name = "half_embedding_dot_backward"

            def _convert_back(op):
                return xla_client.ops.ConvertElementType(op, xla_client.dtype_to_etype(np.float16))

        else:
            raise ValueError("Could not natively suport dtype " + a_shape.dtype)

        res = xla_client.ops.CustomCallWithLayout(
            ctx.builder,
            name.encode("utf8"),
            operands=[a, b, indices, g],
            shape_with_layout=xla_client.Shape.tuple_shape(
                [
                    _convert_to_xla_shape(val, force_dtype=xla_client.dtype_to_etype(np.float32))
                    for val in (da_shape, db_shape)
                ]
            ),
            operand_shapes_with_layout=[
                _convert_to_xla_shape(val) for val in (a_shape, b_shape, indices_shape, g_shape)
            ],
            opaque=struct.pack(
                "IIII",
                a_shape.shape[1],
                indices_shape.shape[0],
                a_shape.shape[0],
                b_shape.shape[0],
            ),
        )

        return [_convert_back(xla_client.ops.GetTupleElement(res, i)) for i, _ in enumerate(avals_out)]

    raise ValueError("Unsupported platform; this must be either 'cpu' or 'gpu', got " + str(ctx.platform))


embedding_dot_backward_p.def_abstract_eval(embedding_dot_backward_p_abstract_eval)
embedding_dot_backward_p.multiple_results = True
embedding_dot_backward_p.def_impl(grad(embedding_dot_fallback, argnums=(0, 1)))
xla.register_translation(embedding_dot_backward_p, embedding_dot_backward_p_xla_translation)


@partial(jax.custom_vjp, nondiff_argnums=(2,))
def exp_mean(a: Array, b: Array, sparse_c: Array):
    """exp_mean computes the mean value of 2^(a @ b + c), where c is a sparse matrix.

    sparse_c is in CSR form with a tuple of the following elements:
        offsets: The offsets for each row, indexing into indices / values
        defaults: The default value for each row
        indices: All of the indices
        values: All of the values

    The columns for each row j are in indices[offsets[j] : offsets[j+1]].
    The values for each row are either defaults[j] or values[offsets[j] : offsets[j+1]]
    """
    return exp_mean_fwd(a, b, sparse_c)[0]


def exp_mean_fallback(a: Array, b: Array, c: Array) -> Array:
    """A fallback implementation of exp_mean, where c is dense."""
    matrix = jnp.matmul(a, b.T)
    return jnp.exp2(matrix + c).astype(jnp.float32).mean().astype(a.dtype)


@jax.jit
def exp_mean_fwd(a: Array, b: Array, sparse_c: Array) -> Tuple[Array, Tuple[Array, Array, Array]]:
    """The forward pass for exp_mean"""
    r, da, db = exp_mean_p.bind(a, b, *sparse_c)
    return r, (a, da, db)


@jax.jit
def exp_mean_bwd(_sparse_c: Array, res: Tuple[Array, Array, Array], g: Array) -> Tuple[Array, Array]:
    """The backward pass for exp_mean"""
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


def exp_mean_p_abstract_eval(
    a: Array,
    b: Array,
    offsets: Array,
    defaults: Array,
    indices: Array,
    values: Array,
) -> Tuple[jax.core.ShapedArray, jax.core.ShapedArray, jax.core.ShapedArray]:
    """Abstract shapes for exp_mean_p."""
    assert a.dtype == b.dtype
    assert len(a.shape) == len(b.shape) == 2
    assert len(offsets.shape) == len(defaults.shape) == len(indices.shape) == len(values.shape) == 1

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
        jax.core.ShapedArray((), jnp.float32),
        jax.core.ShapedArray(a.shape, jnp.float32),
        jax.core.ShapedArray(b.shape, jnp.float32),
    )


def exp_mean_p_xla_translation(
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
    assert (isinstance(val, jax.core.ShapedArray) for val in avals_in)
    assert (isinstance(val, jax.core.ShapedArray) for val in avals_out)
    avals_in = cast(Sequence[jax.core.ShapedArray], avals_in)
    avals_out = cast(Sequence[jax.core.ShapedArray], avals_out)

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
            warnings.warn("Using an inefficient exp_sum mechanism", RuntimeWarning)

        c_shape = jax.core.ShapedArray([a_shape.shape[0], b_shape.shape[0]], jnp.float32)

        value_and_grad_computation = xla_computation(value_and_grad(exp_mean_fallback, argnums=(0, 1)))(
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
            _convert_to_xla_shape(c_shape),
        )
        value_and_grad_res = xla_client.ops.Call(ctx.builder, value_and_grad_computation, [a, b, dense])

        def _convert_to_full(op):
            return xla_client.ops.ConvertElementType(op, xla_client.dtype_to_etype(jnp.float32))

        return [
            _convert_to_full(xla_client.ops.GetTupleElement(value_and_grad_res, i)) for i, _ in enumerate(avals_out)
        ]
    elif ctx.platform == "cuda":

        if a_shape.dtype == jnp.float16:
            name = "half_exp_mean_with_grad"
        else:
            raise ValueError("Could not natively suport dtype " + str(a_shape.dtype))

        factor_to_divide_by = (b_shape.shape[0] + 8 * 16 - 1) // (8 * 16)
        m_shift, m_mult = get_shifts_and_mult(factor_to_divide_by)

        res = xla_client.ops.CustomCallWithLayout(
            ctx.builder,
            name.encode("utf8"),
            operands=[a, b, offsets, defaults, indices, values],
            shape_with_layout=xla_client.Shape.tuple_shape(
                [_convert_to_xla_shape(val) for val in (result_shape, da_shape, db_shape)]
            ),
            operand_shapes_with_layout=[_convert_to_xla_shape(val) for val in avals_in],
            opaque=struct.pack(
                "IIIII",
                a_shape.shape[0],
                b_shape.shape[0],
                a_shape.shape[1],
                m_shift,
                m_mult,
            ),
        )
        return [xla_client.ops.GetTupleElement(res, i) for i, _ in enumerate(avals_out)]

    raise ValueError("Unsupported platform; this must be either 'cpu' or 'gpu', got " + str(ctx.platform))


exp_mean_p.def_abstract_eval(exp_mean_p_abstract_eval)
exp_mean_p.multiple_results = True
xla.register_translation(exp_mean_p, exp_mean_p_xla_translation)


def get_shifts_and_mult(n):
    """Get the shifts and multiplication factors for the standard constant time division algorithm.
    See https://gmplib.org/~tege/divcnst-pldi94.pdf
    """
    if n == 0:
        raise ValueError("No inverse for 0")

    if n == 1:
        return 0, 0

    shift = 0
    while n > (1 << shift):
        shift += 1

    mult = ((2 ** (32 + shift) + n - 1) // n) % (2**32)
    return shift, mult


# The following is "technically" a memory leak
# We just have to hope that piton users don't keep their program open forever ...
_local_attention_data_cache: Dict[Tuple[int, int, int, int], Any] = {}


def _get_cached_local_attention_data(b: int, n: int, k: int, w: int) -> bytes:
    """The local attention data is the C++ data necessary to run the kernel.

    This data only depends on the sizes passed in, so can be precomputed and shared across invocations of the kernel.

    Note that we have to return a byte string of the pointer to the data.
    """
    key = (b, n, k, w)
    if key not in _local_attention_data_cache:
        _local_attention_data_cache[key] = piton.extension.jax.get_local_attention_data(b, n, k, w)

    # We need to pull out the actual pointer and pass that directly.

    ctypes.pythonapi.PyCapsule_GetPointer.restype = ctypes.c_void_p
    ctypes.pythonapi.PyCapsule_GetPointer.argtypes = [
        ctypes.py_object,
        ctypes.c_char_p,
    ]
    pointer = ctypes.pythonapi.PyCapsule_GetPointer(_local_attention_data_cache[key], None)

    result = struct.pack("P", pointer)

    return result


def add_batch_if_necessary(data: Array) -> Array:
    """Add a batch dimension if one is missing."""
    if len(data.shape) == 2:
        return jnp.expand_dims(data, 0)
    else:
        return data


@partial(jax.custom_vjp, nondiff_argnums=(4,))
def local_attention(
    queries: Array,
    keys: Array,
    values: Array,
    length_mask: Array,
    attention_width: int,
) -> Array:
    """
    local_attention is an operation that implements multi-headed attention with a causal local attention pattern.

    A causal local attention pattern means that each token attends to a fixed number of tokens behind it in the
    sequence.

    queries, keys, values and attention_width have the obvious meaning.

    length_mask is special as local_attention supports additional length masking.
    All indices are masked with the length mask before attention is applied.
    This allows you to pass multiple sequences in one call to local_attention and know they won't have cross
    attention.
    """
    return local_attention_fwd(queries, keys, values, length_mask, attention_width)[0]


@partial(vmap, in_axes=(0, 0, 0, None, None))
def local_attention_fallback_single(
    queries: Array,
    keys: Array,
    values: Array,
    length_mask: Array,
    attention_width: int,
) -> Array:
    """A local attention fallback for a single sequence."""
    logits = queries @ keys.T

    causal_mask = jnp.tri(N=queries.shape[0], k=0, dtype=jnp.bool_)
    local_mask = jnp.tri(N=queries.shape[0], k=-(attention_width + 1), dtype=jnp.bool_)

    indices = jnp.arange(queries.shape[0])
    row_indices = jnp.zeros_like(local_mask) + indices.reshape((1, queries.shape[0]))
    col_indices = jnp.zeros_like(local_mask) + indices.reshape((queries.shape[0], 1))

    col_indices = col_indices.astype(jnp.uint32)
    row_indices = row_indices.astype(jnp.uint32)

    row_indices = jnp.bitwise_and(row_indices, length_mask)
    col_indices = jnp.bitwise_and(col_indices, length_mask)

    length_mask = row_indices == col_indices

    full_mask = causal_mask & (~local_mask) & length_mask

    logits = jnp.where(full_mask, logits, float("-inf"))

    attention = nn.softmax(logits / jnp.sqrt(keys.shape[1]))

    result = attention @ values

    return result


def local_attention_fallback(
    queries: Array,
    keys: Array,
    values: Array,
    length: Array,
    attention_width: int,
) -> Tuple[Array, Array]:
    """The full fallback, that supports batching and the dummy attention values"""
    result = local_attention_fallback_single(queries, keys, values, length, attention_width)
    attention_shape = tuple(
        piton.extension.jax.get_local_attention_shape(
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
    queries: Array,
    keys: Array,
    values: Array,
    length: Array,
    _attention: Array,
    g: Array,
    attention_width: int,
) -> Array:
    """Fallback for the gradient. Note that we nede to discard the attention."""
    result = local_attention_fallback_single(queries, keys, values, length, attention_width)
    return (result * g).sum()


def local_attention_fwd(
    queries: Array,
    keys: Array,
    values: Array,
    length: Array,
    attention_width: int,
) -> Tuple[Array, Tuple[Array, Array, Array, Array, Array]]:
    """The forward pass for local_attention."""
    queries = add_batch_if_necessary(queries)
    keys = add_batch_if_necessary(keys)
    values = add_batch_if_necessary(values)

    attention, result = local_attention_forward_p.bind(queries, keys, values, length, attention_width=attention_width)
    res = (queries, keys, values, length, attention)
    return result, res


def local_attention_bwd(
    attention_width: int,
    res: Tuple[Array, Array, Array, Array, Array],
    g: Array,
) -> Tuple[Array, Array, Array, None]:
    """The backward pass for local_attention."""
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


local_attention.defvjp(local_attention_fwd, local_attention_bwd)

local_attention_forward_p = core.Primitive("local_attention_forward")


def local_attention_forward_abstract_eval(
    queries: jax.core.ShapedArray,
    keys: jax.core.ShapedArray,
    values: jax.core.ShapedArray,
    length: jax.core.ShapedArray,
    *,
    attention_width: int,
) -> Tuple[jax.core.ShapedArray, jax.core.ShapedArray]:
    """Forward shapes for local_attention."""
    assert len(queries.shape) == 3

    assert queries.dtype == keys.dtype == values.dtype
    assert queries.shape == keys.shape == values.shape

    assert attention_width % 16 == 0
    assert queries.shape[1] % 16 == 0
    assert queries.shape[2] % 16 == 0

    assert length.shape == tuple()
    assert length.dtype == jnp.uint32

    attention_shape = tuple(
        piton.extension.jax.get_local_attention_shape(
            queries.shape[0],
            queries.shape[1],
            queries.shape[2],
            attention_width,
        )
    )

    return (
        jax.core.ShapedArray(attention_shape, queries.dtype),
        jax.core.ShapedArray(queries.shape, queries.dtype),
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
    """Forward op for local_attention."""
    assert all(isinstance(a, jax.core.ShapedArray) for a in avals_in)
    avals_in = cast(Sequence[jax.core.ShapedArray], avals_in)
    assert all(isinstance(a, jax.core.ShapedArray) for a in avals_out)
    avals_out = cast(Sequence[jax.core.ShapedArray], avals_out)

    queries_shape, keys_shape, values_shape, length_shape = avals_in
    attention_shape, result_shape = avals_out

    if ctx.platform == "cpu" or (ctx.platform == "cuda" and queries_shape.dtype != jnp.float16):
        computation = xla_computation(
            local_attention_fallback,
            static_argnums=(4,),
        )(*avals_in, attention_width)

        res = xla_client.ops.Call(ctx.builder, computation, [queries, keys, values, length])

        if ctx.platform == "cuda":
            warnings.warn("Using an inefficient CUDA attention mechanism", RuntimeWarning)

        return [xla_client.ops.GetTupleElement(res, i) for i, _ in enumerate(avals_out)]
    elif ctx.platform == "cuda":
        if queries_shape.dtype == jnp.float16:
            name = "half_local_attention_forward"
        else:
            raise ValueError("Could not natively suport dtype " + str(queries_shape.dtype))

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
            shape_with_layout=xla_client.Shape.tuple_shape([_convert_to_xla_shape(res) for res in avals_out]),
            operand_shapes_with_layout=[_convert_to_xla_shape(val) for val in avals_in],
            opaque=opaque,
        )

        return [xla_client.ops.GetTupleElement(res, i) for i, _ in enumerate(avals_out)]

    raise ValueError("Unsupported platform; this must be either 'cpu' or 'gpu', got " + str(ctx.platform))


local_attention_forward_p.def_abstract_eval(local_attention_forward_abstract_eval)
local_attention_forward_p.def_impl(local_attention_fallback)
local_attention_forward_p.multiple_results = True
xla.register_translation(local_attention_forward_p, local_attention_forward_xla_translation)

local_attention_backward_p = core.Primitive("local_attention_backward")


def local_attention_backward_abstract_eval(
    queries: jax.core.ShapedArray,
    keys: jax.core.ShapedArray,
    values: jax.core.ShapedArray,
    length: jax.core.ShapedArray,
    attention: jax.core.ShapedArray,
    g: jax.core.ShapedArray,
    attention_width: int,
) -> Tuple[jax.core.ShapedArray, jax.core.ShapedArray, jax.core.ShapedArray]:
    """Abstract shapes for local_attention."""
    assert length.shape == tuple()
    assert length.dtype == jnp.uint32

    assert queries.dtype == keys.dtype == values.dtype == g.dtype
    assert queries.shape == keys.shape == values.shape == g.shape

    assert attention.dtype == queries.dtype

    assert queries.shape[2] == 64

    assert queries.shape[1] % 16 == 0
    assert queries.shape[2] % 16 == 0

    attention_shape = tuple(
        piton.extension.jax.get_local_attention_shape(
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
    """Backward op for local_attention."""
    assert all(isinstance(a, jax.core.ShapedArray) for a in avals_in)
    avals_in = cast(Sequence[jax.core.ShapedArray], avals_in)

    assert all(isinstance(a, jax.core.ShapedArray) for a in avals_out)
    avals_out = cast(Sequence[jax.core.ShapedArray], avals_out)

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
        computation = xla_computation(local_attention_backward_fallback, static_argnums=(6,))(
            *avals_in, attention_width
        )

        res = xla_client.ops.Call(
            ctx.builder,
            computation,
            [queries, keys, values, length, attention, g],
        )
        return [xla_client.ops.GetTupleElement(res, i) for i, _ in enumerate(avals_out)]
    elif ctx.platform == "cuda":
        if queries_shape.dtype == np.float16:
            name = "half_local_attention_backward"
        else:
            raise ValueError("Could not natively suport dtype " + avals_in[0].dtype)

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
                [_convert_to_xla_shape(avals_out[0])]
                + [
                    _convert_to_xla_shape(val, force_dtype=xla_client.dtype_to_etype(jnp.float32))
                    for val in avals_out[1:]
                ]
            ),
            operand_shapes_with_layout=[_convert_to_xla_shape(val) for val in avals_in],
            opaque=opaque,
        )

        def _convert_back(op):
            return xla_client.ops.ConvertElementType(op, xla_client.dtype_to_etype(jnp.float16))

        return [
            _convert_back(xla_client.ops.GetTupleElement(res, i)) if i > 0 else xla_client.ops.GetTupleElement(res, i)
            for i, _ in enumerate(avals_out)
        ]

    raise ValueError("Unsupported platform; this must be either 'cpu' or 'gpu', got " + str(ctx.platform))


local_attention_backward_p.def_abstract_eval(local_attention_backward_abstract_eval)
local_attention_backward_p.multiple_results = True
local_attention_backward_p.def_impl(local_attention_backward_fallback)
xla.register_translation(local_attention_backward_p, local_attention_backward_xla_translation)
