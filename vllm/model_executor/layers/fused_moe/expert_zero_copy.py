# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""View page-locked host memory as CUDA tensors, for unified-memory boxes.

On a discrete GPU an expert has to be copied into device memory before a
kernel can read it. On GB10-class unified memory the two "memories" are one
physical LPDDR5x pool, so a ``cudaHostRegister``-ed row already carries a
valid device address and the copy is pure overhead -- measured at 1.35 TB of
host-to-device traffic over a single 256-token run.

This module builds the CUDA tensor view. The MoE kernel takes the expert
dimension's stride explicitly (``B.stride(0)``) and only requires
``stride(-1) == 1``, so a view straight over the disk tier's record pool
works: records sit ``record_stride`` bytes apart with each field at a fixed
offset inside them.

Measured on GB10 (bench/zero_copy_probe.py, Qwen3-30B decode shapes): the
kernel reads registered host memory at ~150 GB/s against ~210 GB/s from
device memory, and produces bit-identical output. Slower per GEMM, but 3.5x
to 5x faster than the fill-then-compute path it replaces.
"""

import ctypes

import torch

_kDLCUDA = 2

# DLPack's C ABI. torch.utils.dlpack.from_dlpack consumes a "dltensor"
# capsule; building one by hand is what lets an arbitrary (device-addressable)
# pointer become a tensor without a copy.


class _DLDevice(ctypes.Structure):
    _fields_ = [("device_type", ctypes.c_int32), ("device_id", ctypes.c_int32)]


class _DLDataType(ctypes.Structure):
    _fields_ = [
        ("code", ctypes.c_uint8),
        ("bits", ctypes.c_uint8),
        ("lanes", ctypes.c_uint16),
    ]


class _DLTensor(ctypes.Structure):
    _fields_ = [
        ("data", ctypes.c_void_p),
        ("device", _DLDevice),
        ("ndim", ctypes.c_int32),
        ("dtype", _DLDataType),
        ("shape", ctypes.POINTER(ctypes.c_int64)),
        ("strides", ctypes.POINTER(ctypes.c_int64)),
        ("byte_offset", ctypes.c_uint64),
    ]


class _DLManagedTensor(ctypes.Structure):
    pass


_DELETER = ctypes.CFUNCTYPE(None, ctypes.POINTER(_DLManagedTensor))
_DLManagedTensor._fields_ = [
    ("dl_tensor", _DLTensor),
    ("manager_ctx", ctypes.c_void_p),
    ("deleter", _DELETER),
]

# DLPack type codes, read back from torch.utils.dlpack.to_dlpack rather than
# assumed: the fp8 formats have their own codes, not kDLFloat with 8 bits,
# and PyTorch rejects the latter with "Unsupported kFloat bits 8".
_DTYPE_CODE: dict[torch.dtype, tuple[int, int]] = {
    torch.bfloat16: (4, 16),
    torch.float16: (2, 16),
    torch.float32: (2, 32),
    torch.float8_e4m3fn: (10, 8),
}

ctypes.pythonapi.PyCapsule_New.restype = ctypes.py_object
ctypes.pythonapi.PyCapsule_New.argtypes = [
    ctypes.c_void_p,
    ctypes.c_char_p,
    ctypes.c_void_p,
]


_libc = ctypes.CDLL(None)
_libc.malloc.restype = ctypes.c_void_p
_libc.malloc.argtypes = [ctypes.c_size_t]


def _malloc(nbytes: int) -> int:
    """Allocate memory Python will never free."""
    ptr = _libc.malloc(nbytes)
    if not ptr:
        raise MemoryError(f"malloc({nbytes}) failed building a zero-copy view")
    ctypes.memset(ptr, 0, nbytes)
    return ptr


# The consumer keeps the DLManagedTensor pointer for the tensor's whole life
# and dereferences it on destruction, so nothing the capsule points at may be
# Python-owned. Two ways that bites, both observed on GB10:
#
#   * keepalive on the returned tensor -> its attributes are cleared before the
#     destructor runs, so the deleter callback is already freed when called;
#     the crash lands on whatever unrelated line dropped the last reference.
#   * keepalive in a module global -> interpreter shutdown clears the module
#     before the tensors die, and the destructor reads a freed struct.
#
# So the structs come from raw malloc and the deleter is NULL (PyTorch checks
# it before calling). Deliberately leaked: a few hundred bytes per view, and
# views are built once per layer at startup.
#
# ``owner`` is different -- a plain Python reference, no C indirection -- and
# only has to outlive the views' *use*, which the provider already guarantees
# by holding the pool.
_LIVE_OWNERS: list[torch.Tensor] = []


def cuda_view(
    ptr: int,
    shape: tuple[int, ...],
    strides: tuple[int, ...],
    dtype: torch.dtype,
    owner: torch.Tensor,
) -> torch.Tensor:
    """View memory at *ptr* as a CUDA tensor. No copy, no ownership.

    Args:
        ptr: device-addressable address; for registered host memory the host
            pointer is that address (unified virtual addressing).
        shape: tensor shape.
        strides: strides in *elements*, matching shape.
        dtype: element type; must be in ``_DTYPE_CODE``.
        owner: tensor that actually owns the memory. Retained module-side so
            the backing allocation cannot be freed while a view aliases it.

    Returns:
        A CUDA tensor aliasing that memory.
    """
    if dtype not in _DTYPE_CODE:
        raise ValueError(f"zero-copy view unsupported for dtype {dtype}")
    code, bits = _DTYPE_CODE[dtype]
    ndim = len(shape)
    if len(strides) != ndim:
        raise ValueError(f"shape {shape} and strides {strides} disagree in rank")
    dims = ctypes.c_int64 * ndim
    shape_arr = dims.from_address(_malloc(ctypes.sizeof(dims)))
    stride_arr = dims.from_address(_malloc(ctypes.sizeof(dims)))
    shape_arr[:] = shape
    stride_arr[:] = strides

    mt_addr = _malloc(ctypes.sizeof(_DLManagedTensor))
    mt = _DLManagedTensor.from_address(mt_addr)
    mt.dl_tensor.data = ctypes.c_void_p(ptr)
    mt.dl_tensor.device = _DLDevice(_kDLCUDA, torch.cuda.current_device())
    mt.dl_tensor.ndim = ndim
    mt.dl_tensor.dtype = _DLDataType(code, bits, 1)
    mt.dl_tensor.shape = shape_arr
    mt.dl_tensor.strides = stride_arr
    mt.dl_tensor.byte_offset = 0
    mt.manager_ctx = None
    mt.deleter = ctypes.cast(None, _DELETER)  # nothing to free; never called
    _LIVE_OWNERS.append(owner)

    capsule = ctypes.pythonapi.PyCapsule_New(
        ctypes.c_void_p(mt_addr), b"dltensor", None
    )
    return torch.utils.dlpack.from_dlpack(capsule)


def pool_field_view(
    pool: torch.Tensor,
    record_stride: int,
    offset: int,
    shape: tuple[int, ...],
    dtype: torch.dtype,
    slots: int,
) -> torch.Tensor:
    """Expose one record field across every pool slot as ``[slots, *shape]``.

    The record pool is ``[slots, record_stride]`` bytes; this gathers the
    field living at ``offset`` inside each record into a single strided
    tensor whose first dimension indexes the slot -- exactly the layout the
    MoE kernel wants for ``w1``/``w2``.

    Raises:
        ValueError: if the field's address or the record stride is not a
            whole number of elements, which would make the stride
            unrepresentable.
    """
    itemsize = torch.empty(0, dtype=dtype).element_size()
    base = pool.data_ptr() + offset
    if base % itemsize or record_stride % itemsize:
        raise ValueError(
            f"record layout not {itemsize}-byte aligned "
            f"(offset={offset}, stride={record_stride})"
        )
    inner: list[int] = []
    acc = 1
    for d in reversed(shape):
        inner.append(acc)
        acc *= d
    inner.reverse()
    return cuda_view(
        base,
        (slots, *shape),
        (record_stride // itemsize, *inner),
        dtype,
        pool,
    )
