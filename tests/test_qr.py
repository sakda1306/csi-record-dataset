import numpy as np
import pytest

from csirec.qr import qr_matrix, qr_svg


def test_qr_size_grows_with_version():
    assert len(qr_matrix("x" * 10)) == 21  # version 1-M holds 14 bytes
    assert len(qr_matrix("x" * 60)) == 33  # version 4-M holds 62 bytes
    m = qr_matrix("x" * 60)
    assert all(len(row) == 33 for row in m)


def test_qr_finder_patterns():
    m = qr_matrix("hello")
    n = len(m)
    for r0, c0 in ((0, 0), (0, n - 7), (n - 7, 0)):
        for r in range(7):
            for c in range(7):
                ring = max(abs(r - 3), abs(c - 3))
                assert m[r0 + r][c0 + c] == (ring != 2), (r0, c0, r, c)


def test_qr_too_long_raises():
    with pytest.raises(ValueError):
        qr_matrix("x" * 300)


def test_qr_svg_is_svg_with_quiet_zone():
    s = qr_svg("hello")
    assert s.startswith("<svg")
    assert 'viewBox="0 0 29 29"' in s  # 21 modules + 2 x 4 border
    assert "crispEdges" in s


@pytest.mark.parametrize(
    "data",
    [
        "http://192.168.1.45:8765/m#pin=1234",
        "A",
        "http://192.168.100.200:8765/m#pin=" + "9" * 24,  # 58 chars, version 4
        "http://10.0.0.2:8765/m#pin=a%23b%26c",
        "x" * 150,  # version 8, has version-info blocks
    ],
)
def test_qr_decodes_with_opencv(data):
    cv2 = pytest.importorskip("cv2")
    m = np.array(qr_matrix(data), dtype=bool)
    img = np.where(m, 0, 255).astype(np.uint8)
    img = np.pad(img, 4, constant_values=255)
    img = np.kron(img, np.ones((8, 8), dtype=np.uint8))
    text, _, _ = cv2.QRCodeDetector().detectAndDecode(img)
    assert text == data
