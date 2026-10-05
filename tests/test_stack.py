import numcodecs
import numpy as np
import xarray as xr
from numcodecs.abc import Codec

import numcodecs_combinators
from numcodecs_combinators.stack import CodecStack


def assert_config_roundtrip(codec: numcodecs.abc.Codec):
    config = codec.get_config()
    codec2 = numcodecs.get_codec(config)
    assert codec2 == codec


def test_init_config():
    stack = CodecStack()
    assert len(stack) == 0
    assert_config_roundtrip(stack)

    stack = CodecStack(dict(id="zlib", level=9))
    assert len(stack) == 1
    assert_config_roundtrip(stack)

    stack = CodecStack(dict(id="zlib", level=9), numcodecs.CRC32())
    assert len(stack) == 2
    assert_config_roundtrip(stack)


def test_encode_decode():
    stack = CodecStack(numcodecs.Zlib(level=9), numcodecs.CRC32())

    stack_encoded = stack.encode(b"abc")
    encoded = numcodecs.CRC32().encode(numcodecs.Zlib(level=9).encode(b"abc"))
    assert type(stack_encoded) is type(encoded)
    assert (np.array(stack_encoded) == np.array(encoded)).all()

    stack_decoded = stack.decode(stack_encoded)
    decoded = numcodecs.Zlib(level=9).decode(numcodecs.CRC32().decode(encoded))
    assert stack_decoded == decoded
    assert stack_decoded == b"abc"

    encoded_decoded = stack.encode_decode(b"abc")
    assert encoded_decoded == b"abc"

    encoded_decoded = stack.encode_decode_data_array(xr.DataArray([1.0, 2.0, 3.0]))
    assert encoded_decoded.equals(xr.DataArray([1.0, 2.0, 3.0]))

    encoded_decoded = stack.encode_decode_data_array(
        xr.DataArray([1.0, 2.0, 3.0]).chunk(1)
    )
    assert encoded_decoded.equals(xr.DataArray([1.0, 2.0, 3.0]))


def test_chunked_encode_decode():
    class CheckChunkedCodec(Codec):
        __slots__ = ("is_chunked",)
        is_chunked: bool

        def __init__(self, is_chunked: bool):
            self.is_chunked = is_chunked

        def encode(self, buf):
            assert getattr(buf, "chunked", False) == self.is_chunked
            return buf

        def decode(self, buf, out=None):
            assert getattr(buf, "chunked", False) is False
            assert getattr(out, "chunked", False) == self.is_chunked
            return numcodecs.compat.ndarray_copy(buf, out)

    stack = CodecStack(CheckChunkedCodec(False))

    encoded_decoded = stack.encode_decode(np.array([1.0, 2.0, 3.0]))
    assert np.all(encoded_decoded == np.array([1.0, 2.0, 3.0]))

    encoded_decoded = stack.encode_decode_data_array(xr.DataArray([1.0, 2.0, 3.0]))
    assert encoded_decoded.equals(xr.DataArray([1.0, 2.0, 3.0]))

    stack = CodecStack(CheckChunkedCodec(True))

    encoded_decoded = stack.encode_decode_data_array(
        xr.DataArray([1.0, 2.0, 3.0]).chunk(1)
    )
    assert encoded_decoded.equals(xr.DataArray([1.0, 2.0, 3.0]))


def test_chunked_encode_decode_non_contiguous():
    da = xr.DataArray(np.zeros((100, 100)), dims=["a", "b"])
    da_chunked = da.chunk(a=7, b=7)

    CodecStack().encode_decode_data_array(da_chunked).compute()


def test_map():
    stack = CodecStack(numcodecs.Zlib(level=9), numcodecs.CRC32())

    mapped = numcodecs_combinators.map_codec(stack, lambda c: c)
    assert mapped == stack

    mapped = numcodecs_combinators.map_codec(stack, lambda c: CodecStack(c))
    assert mapped == CodecStack(
        CodecStack(CodecStack(numcodecs.Zlib(level=9)), CodecStack(numcodecs.CRC32()))
    )

    mapped = numcodecs_combinators.map_codec(mapped, lambda c: CodecStack(c))
    assert mapped == CodecStack(
        CodecStack(
            CodecStack(
                CodecStack(
                    CodecStack(CodecStack(CodecStack(numcodecs.Zlib(level=9)))),
                    CodecStack(CodecStack(CodecStack(numcodecs.CRC32()))),
                )
            )
        )
    )


class SparseCodec(Codec):
    """Test codec implementing the mask-aware protocol of numcodecs-mask:
    it only stores the unmasked values and records the masks it was given."""

    codec_id = "sparse-test"

    def __init__(self):
        self.masks: list[np.ndarray] = []

    def encode(self, buf):
        return self.encode_masked(buf, np.zeros(np.shape(buf), dtype=np.bool))

    def decode(self, buf, out=None):
        return self.decode_masked(buf, np.zeros(np.shape(out), dtype=np.bool), out)

    def encode_masked(self, buf, mask):
        a = numcodecs.compat.ensure_ndarray(buf)
        assert mask.shape == a.shape and mask.dtype == np.bool
        self.masks.append(np.copy(mask))
        return a[~mask].tobytes()

    def decode_masked(self, buf, mask, out=None):
        assert mask.dtype == np.bool
        decoded = np.zeros(mask.shape, dtype=np.float64)
        decoded[~mask] = np.frombuffer(
            numcodecs.compat.ensure_bytes(buf), dtype=np.float64
        )
        return numcodecs.compat.ndarray_copy(decoded, out)

    def get_config(self) -> dict:
        return dict(id=type(self).codec_id)


class ClampFilterCodec(Codec):
    """Test filter codec that clamps on encode and passes through on decode,
    preserving the shape of the data."""

    codec_id = "clamp-test"

    def encode(self, buf):
        return np.minimum(numcodecs.compat.ensure_ndarray(buf), 10.0)

    def decode(self, buf, out=None):
        return numcodecs.compat.ndarray_copy(numcodecs.compat.ensure_ndarray(buf), out)

    def get_config(self) -> dict:
        return dict(id=type(self).codec_id)


def test_encode_decode_masked():
    data = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    mask = np.array([[False, True, False], [True, False, False]])

    # the mask is forwarded through mask-unaware, shape-preserving filters to
    # mask-aware codecs
    inner = SparseCodec()
    stack = CodecStack(ClampFilterCodec(), inner)

    encoded = stack.encode_masked(data, mask)
    assert len(inner.masks) == 1
    np.testing.assert_array_equal(inner.masks[0], mask)
    # the mask-aware codec only stored the four unmasked values
    assert len(numcodecs.compat.ensure_bytes(encoded)) == 4 * 8

    decoded = np.asarray(stack.decode_masked(encoded, mask)).reshape(data.shape)
    np.testing.assert_array_equal(decoded[~mask], data[~mask])

    # an empty stack and a stack of only mask-unaware codecs still roundtrip
    np.testing.assert_array_equal(CodecStack().encode_masked(data, mask), data)
    stack = CodecStack(dict(id="zlib", level=1))
    decoded = np.frombuffer(
        numcodecs.compat.ensure_bytes(
            stack.decode_masked(stack.encode_masked(data, mask), mask)
        )
    ).reshape(data.shape)
    np.testing.assert_array_equal(decoded, data)
