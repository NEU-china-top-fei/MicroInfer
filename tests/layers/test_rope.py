import pytest
import torch

from microinfer.layer.rope import RotaryCache, apply, comp_freq_cis, rotate


def test_frequency_table_shape_and_zero_position():
    freq = comp_freq_cis(7, 10000.0, 8)
    assert freq.shape == (7, 4)
    assert freq.is_complex()
    torch.testing.assert_close(freq[0], torch.ones(4, dtype=freq.dtype))
    torch.testing.assert_close(freq.abs(), torch.ones(7, 4))


def test_frequencies_match_independent_formula():
    dim, length, base = 8, 7, 10000.0
    freq = comp_freq_cis(length, base, dim)
    for position in [1, 3, 6]:
        for pair in range(dim // 2):
            angle = torch.tensor(position / base ** (2 * pair / dim))
            expected = torch.complex(angle.cos(), angle.sin())
            torch.testing.assert_close(freq[position, pair], expected)


def test_frequency_prefix_is_independent_of_table_length():
    short = comp_freq_cis(4, 10000.0, 8)
    long = comp_freq_cis(16, 10000.0, 8)
    torch.testing.assert_close(short, long[:4])


@pytest.mark.parametrize("interleaved", [False, True])
def test_rotary_cache_matches_rotate_and_builds_lazily(
    seeded_torch, interleaved
):
    x = torch.randn(2, 3, 2, 8)
    positions = torch.tensor([5, 1, 3])
    cache = RotaryCache(8, 8, 10000.0, interleaved=interleaved)
    assert cache._cos is None

    actual = cache.apply(x, positions)
    expected = rotate(x, positions, 10000.0, interleaved=interleaved)

    torch.testing.assert_close(actual, expected)
    assert cache._cos is not None and cache._sin is not None
    # Cached table is reused, not rebuilt on the next layer.
    again = cache.apply(x, positions)
    torch.testing.assert_close(again, actual)


def test_rotary_cache_supports_packed_and_decode_shapes(seeded_torch):
    cache = RotaryCache(16, 8, 10000.0)
    packed = torch.randn(5, 2, 8)
    positions = torch.arange(5)
    torch.testing.assert_close(
        cache.apply(packed, positions),
        rotate(packed, positions, 10000.0),
    )

    decode = torch.randn(3, 2, 8)
    decode_positions = torch.tensor([9, 4, 0])
    torch.testing.assert_close(
        cache.apply(decode, decode_positions),
        rotate(decode, decode_positions, 10000.0),
    )


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_rotation_matches_real_pair_formula(seeded_torch, device, dtype):
    # Current public apply contract uses adjacent real/imaginary pairs.
    # This is not HF Qwen's split-half checkpoint layout; no weight parity claimed.
    x = torch.randn(2, 3, 2, 8).to(device=device, dtype=dtype)
    angles = torch.arange(12, device=device, dtype=torch.float32).reshape(3, 4) / 7
    freq = torch.polar(torch.ones_like(angles), angles)
    before = x.clone()
    actual = apply(x, freq)
    cosine, sine = angles.cos()[None, :, None, :], angles.sin()[None, :, None, :]
    even, odd = x.float()[..., 0::2], x.float()[..., 1::2]
    expected = (
        torch.stack((even * cosine - odd * sine, even * sine + odd * cosine), dim=-1)
        .flatten(-2)
        .to(dtype)
    )
    assert actual.shape == x.shape
    assert actual.dtype == dtype
    assert actual.device == device
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(x, before)
    torch.testing.assert_close(
        actual.float().square().sum(-1), x.float().square().sum(-1), atol=0.1, rtol=0.01
    )
