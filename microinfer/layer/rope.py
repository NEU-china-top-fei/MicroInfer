import torch


def comp_freq_cis(max_len: int, base: float, dim: int) -> torch.Tensor:
    cis = base ** (torch.arange(0, dim, 2, dtype=torch.float32) / max_len)
    angle = torch.outer(torch.arange(0, max_len, dtype=torch.float32), cis)
    return torch.polar(torch.ones_like(angle), angle)


def apply(x: torch.Tensor, freq_cis: torch.Tensor) -> torch.Tensor:
    """
    [batchsize,seqlen,numhead,hiddendim]->complex->[batchsize,seqlen,numhead,hiddendim]
    """
    x_complex = torch.view_as_complex(x.view(*x.shape[:-1], -1, 2))
    freq_cis_complex = freq_cis.view(1, freq_cis.shape(0), 1, freq_cis.shape(1))

    product = x_complex * freq_cis_complex

    return torch.view_as_real(product).flatten(2).to(x.dtype)
