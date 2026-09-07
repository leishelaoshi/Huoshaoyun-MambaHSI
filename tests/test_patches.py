import numpy as np

from huoshaoyun_mambahsi.sampling import extract_patch
from huoshaoyun_mambahsi.splits import chebyshev_distance


def test_patch_shapes():
    img = np.random.default_rng(0).normal(size=(9, 9, 5)).astype(np.float32)
    for p in (1, 3, 5, 7):
        x = extract_patch(img, 4, 4, p)
        assert x.shape == (5, p, p)


def test_border_patch_shape():
    img = np.ones((4, 4, 3), dtype=np.float32)
    assert extract_patch(img, 0, 0, 7).shape == (3, 7, 7)


def test_chebyshev_distance():
    assert chebyshev_distance((0, 0), (6, 4)) == 6
