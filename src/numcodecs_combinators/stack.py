"""
This module defines the [`CodecStack`][numcodecs_combinators.stack.CodecStack] class, which exposes a stack of codecs as a combined codec.
"""

__all__ = ["CodecStack"]

from typing import TYPE_CHECKING, Callable, Optional

import numcodecs
import numcodecs.compat
import numcodecs.registry
import numpy as np
from typing_extensions import Buffer, Self  # MSPV 3.12

if TYPE_CHECKING:
    try:
        import xarray as xr
    except ImportError:
        pass

from numcodecs.abc import Codec

from ._chunked import ChunkedNdArray
from .abc import CodecCombinatorMixin


def _is_mask_aware(codec: Codec) -> bool:
    """Check for the mask-aware codec protocol of
    [`numcodecs-mask`](https://numcodecs-mask.readthedocs.io) structurally, so
    that `numcodecs-combinators` does not need to depend on it."""
    return callable(getattr(codec, "encode_masked", None)) and callable(
        getattr(codec, "decode_masked", None)
    )


class CodecStack(Codec, CodecCombinatorMixin, tuple[Codec]):
    """
    A stack of codecs, which makes up a combined codec.

    On encoding, the codecs are applied to encode from left to right, i.e.
    ```python
    CodecStack(a, b, c).encode(buf)
    ```
    computes
    ```python
    c.encode(b.encode(a.encode(buf)))
    ```

    On decoding, the codecs are applied to decode from right to left, i.e.
    ```python
    CodecStack(a, b, c).decode(buf)
    ```
    computes
    ```python
    a.decode(b.decode(c.decode(buf)))
    ```

    The [`CodecStack`][numcodecs_combinators.stack.CodecStack] provides the
    additional
    [`encode_decode(buf)`][numcodecs_combinators.stack.CodecStack.encode_decode]
    method that computes
    ```python
    stack.decode(stack.encode(buf))
    ```
    but makes use of knowing the shapes and dtypes of all intermediary encoding
    stages.
    """

    __slots__ = ()

    codec_id: str = "combinators.stack"  # type: ignore

    def __init__(self, *args: dict | Codec):
        pass

    def __new__(cls, *args: dict | Codec) -> Self:
        return super(CodecStack, cls).__new__(
            cls,
            tuple(
                codec
                if isinstance(codec, Codec)
                else numcodecs.registry.get_codec(codec)
                for codec in args
            ),
        )

    def encode(self, buf: Buffer) -> Buffer:
        """Encode the data in `buf`.

        Parameters
        ----------
        buf : Buffer
            Data to be encoded. May be any object supporting the new-style
            buffer protocol.

        Returns
        -------
        enc : Buffer
            Encoded data. May be any object supporting the new-style buffer
            protocol.
        """

        if len(self) == 0:
            return buf

        chunked = getattr(buf, "chunked", False)

        encoded = buf
        for codec in self:
            encoded_ndarray = np.asarray(
                numcodecs.compat.ensure_contiguous_ndarray_like(encoded, flatten=False)
            )
            encoded = codec.encode(
                ChunkedNdArray(encoded_ndarray) if chunked else encoded_ndarray
            )

        if getattr(encoded, "chunked", False):
            return np.array(encoded).view(np.ndarray)  # type: ignore
        return encoded

    def decode(self, buf: Buffer, out: Optional[Buffer] = None) -> Buffer:
        """Decode the data in `buf`.

        Parameters
        ----------
        buf : Buffer
            Encoded data. May be any object supporting the new-style buffer
            protocol.
        out : Buffer, optional
            Writeable buffer to store decoded data. N.B. if provided, this buffer must
            be exactly the right size to store the decoded data.

        Returns
        -------
        dec : Buffer
            Decoded data. May be any object supporting the new-style
            buffer protocol.
        """

        decoded = buf
        for codec in reversed(self):
            decoded = codec.decode(
                numcodecs.compat.ensure_contiguous_ndarray_like(decoded, flatten=False),
                out=None,
            )
        return numcodecs.compat.ndarray_copy(decoded, out)  # type: ignore

    def encode_masked(
        self, buf: Buffer, mask: np.ndarray[tuple[int, ...], np.dtype[np.bool]]
    ) -> Buffer:
        """Encode the data in `buf`, ignoring the values where `mask` is
        [`True`][True].

        The stack forwards the mask to every codec in the stack that
        implements the mask-aware codec protocol of
        [`numcodecs-mask`](https://numcodecs-mask.readthedocs.io), i.e. that
        provides `encode_masked` and `decode_masked` methods; all other codecs
        encode as usual. All mask-aware codecs in the stack must encode data
        of the same shape as the mask, e.g. shape-preserving filters may
        precede them but reshaping codecs must not.

        Parameters
        ----------
        buf : Buffer
            Data to be encoded. May be any object supporting the new-style
            buffer protocol. The values at masked positions are unspecified.
        mask : np.ndarray[tuple[int, ...], np.dtype[np.bool]]
            The [boolean][numpy.bool] mask, of the same shape as the data, of
            the values that do not need to be preserved.

        Returns
        -------
        enc : Buffer
            Encoded data. May be any object supporting the new-style buffer
            protocol.
        """

        if len(self) == 0:
            return buf

        encoded = buf
        for codec in self:
            encoded_ndarray = np.asarray(
                numcodecs.compat.ensure_contiguous_ndarray_like(encoded, flatten=False)
            )
            if _is_mask_aware(codec):
                encoded = codec.encode_masked(encoded_ndarray, mask)  # type: ignore
            else:
                encoded = codec.encode(encoded_ndarray)
        return encoded

    def decode_masked(
        self,
        buf: Buffer,
        mask: np.ndarray[tuple[int, ...], np.dtype[np.bool]],
        out: Optional[Buffer] = None,
    ) -> Buffer:
        """Decode the data in `buf`, which was encoded with the same `mask`.

        The mask is forwarded to every codec in the stack that implements the
        mask-aware codec protocol of
        [`numcodecs-mask`](https://numcodecs-mask.readthedocs.io), matching
        [`encode_masked`][numcodecs_combinators.stack.CodecStack.encode_masked].

        Parameters
        ----------
        buf : Buffer
            Encoded data. May be any object supporting the new-style buffer
            protocol.
        mask : np.ndarray[tuple[int, ...], np.dtype[np.bool]]
            The [boolean][numpy.bool] mask, of the same shape as the decoded
            data, that was passed to
            [`encode_masked`][numcodecs_combinators.stack.CodecStack.encode_masked].
        out : Buffer, optional
            Writeable buffer to store decoded data. N.B. if provided, this
            buffer must be exactly the right size to store the decoded data.

        Returns
        -------
        dec : Buffer
            Decoded data. May be any object supporting the new-style buffer
            protocol. The values at masked positions are unspecified.
        """

        decoded = buf
        for codec in reversed(self):
            decoded_ndarray = numcodecs.compat.ensure_contiguous_ndarray_like(
                decoded, flatten=False
            )
            if _is_mask_aware(codec):
                decoded = codec.decode_masked(decoded_ndarray, mask, out=None)  # type: ignore
            else:
                decoded = codec.decode(decoded_ndarray, out=None)
        return numcodecs.compat.ndarray_copy(decoded, out)  # type: ignore

    def encode_decode(self, buf: Buffer) -> Buffer:
        """
        Encode, then decode the data in `buf`.

        Parameters
        ----------
        buf : Buffer
            Data to be encoded. May be any object supporting the new-style
            buffer protocol.

        Returns
        -------
        dec : Buffer
            Decoded data. May be any object supporting the new-style
            buffer protocol.
        """

        chunked = getattr(buf, "chunked", False)

        encoded = np.asarray(
            numcodecs.compat.ensure_contiguous_ndarray_like(buf, flatten=False)
        )
        silhouettes = []

        for codec in self:
            silhouettes.append((encoded.shape, encoded.dtype))
            encoded = np.asarray(
                numcodecs.compat.ensure_contiguous_ndarray_like(
                    codec.encode(ChunkedNdArray(encoded) if chunked else encoded),
                    flatten=False,
                )
            )

        decoded = encoded.view(np.ndarray)

        for codec in reversed(self):
            shape, dtype = silhouettes.pop()
            out = np.empty(shape=shape, dtype=dtype)
            decoded = (
                codec.decode(decoded, ChunkedNdArray(out) if chunked else out)
                .view(dtype)
                .reshape(shape)
            )

        if getattr(decoded, "chunked", False):
            decoded = decoded.view(np.ndarray)

        if isinstance(decoded, type(buf)):
            return decoded

        return type(buf)(decoded)  # type: ignore

    def encode_decode_data_array(self, da: "xr.DataArray") -> "xr.DataArray":
        """
        Encode, then decode the data array `da`. If `da` is chunked, each chunk
        is encoded and decoded *independently*.

        Since each chunk is encoded *independently*, this method may cause
        chunk boundary artifacts. Do *not* use this method if the codec
        requires access to the entire data at once or if it needs to access
        a neighbourhood of points across the chunk boundary. In these cases,
        it is preferable to use
        `da.copy(data=stack.encode_decode(da.values))` instead.

        The encode-decode computation may be deferred until the
        [`compute`][xarray.DataArray.compute] method is called on the result.

        This method requires the optional [`xarray`][xarray] dependency to be
        installed.

        Parameters
        ----------
        da : xr.DataArray
            Data to be encoded.

        Returns
        -------
        dec : xr.DataArray
            Decoded data.
        """

        import xarray as xr

        if da.chunks is None:
            return da.copy(data=self.encode_decode(da.values))  # type: ignore

        def encode_decode_data_array_single_chunk(
            da: xr.DataArray,
        ) -> xr.DataArray:
            single_chunk = {dim: -1 for dim in da.dims}

            # return early for zero-sized arrays
            if da.size == 0:
                return da.copy(deep=False).chunk(single_chunk)

            # eagerly compute the input chunk and encode and decode it
            decoded = self.encode_decode(
                ChunkedNdArray(  # type: ignore
                    # ensure that the array is contiguous, copying if necessary
                    np.ascontiguousarray(da.values).reshape(da.shape)
                )
            )

            return da.copy(deep=False, data=np.array(decoded).view(np.ndarray)).chunk(
                single_chunk
            )

        return xr.map_blocks(encode_decode_data_array_single_chunk, da)

    def get_config(self) -> dict:
        """
        Returns the configuration of the codec stack.

        [`numcodecs.registry.get_codec(config)`][numcodecs.registry.get_codec]
        can be used to reconstruct this stack from the returned config.

        Returns
        -------
        config : dict
            Configuration of the codec stack.
        """

        return dict(
            id=type(self).codec_id,
            codecs=tuple(codec.get_config() for codec in self),
        )

    @classmethod
    def from_config(cls, config: dict) -> Self:
        """
        Instantiate the codec stack from a configuration [`dict`][dict].

        Parameters
        ----------
        config : dict
            Configuration of the codec stack.

        Returns
        -------
        stack : CodecStack
            Instantiated codec stack.
        """

        return cls(*config["codecs"])

    def __repr__(self) -> str:
        repr = ", ".join(f"{codec!r}" for codec in self)

        return f"{type(self).__name__}({repr})"

    def map(self, mapper: Callable[[Codec], Codec]) -> "CodecStack":
        """
        Apply the `mapper` to all codecs that are in this stack.
        In the returned stack, each codec is replaced by its mapped codec.

        The `mapper` should recursively apply itself to any inner codecs that
        also implement the [`CodecCombinatorMixin`][numcodecs_combinators.abc.CodecCombinatorMixin]
        mixin.

        To automatically handle the recursive application as a caller, you can
        use
        ```python
        numcodecs_combinators.map_codec(stack, mapper)
        ```
        instead.

        Parameters
        ----------
        mapper : Callable[[Codec], Codec]
            The callable that should be applied to each codec to map over this
            codec stack.

        Returns
        -------
        mapped : CodecStack
            The mapped codec stack.
        """

        return CodecStack(*map(mapper, self))

    def __add__(self, other) -> "CodecStack":
        return CodecStack(*tuple.__add__(self, other))

    def __mul__(self, other) -> "CodecStack":
        return CodecStack(*tuple.__mul__(self, other))

    def __rmul__(self, other) -> "CodecStack":
        return CodecStack(*tuple.__rmul__(self, other))


numcodecs.registry.register_codec(CodecStack)
