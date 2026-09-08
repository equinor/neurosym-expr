from expr07_tria.batched_diagonal_spline_transport import (
    BatchedDiagonalSplineTransport,
)
from expr07_tria.boosted_wavelet_transport import BoostedWaveletSplineTransport
from expr07_tria.torch_pspline_transport import AdaptiveSplineTransport

__all__ = [
    "AdaptiveSplineTransport",
    "BatchedDiagonalSplineTransport",
    "BoostedWaveletSplineTransport",
]


def main() -> None:
    print("PyTorch adaptive P-spline triangular transport")
