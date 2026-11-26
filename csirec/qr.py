"""QR code encoder (byte mode, error correction level M, versions 1-10), standard library only.

The recorder must work offline with nothing extra installed, so it draws its own QR for the
phone address instead of using a QR package or web service.

    qr_matrix("http://192.168.1.45:8765/m")  -> rows of booleans (True = dark)
    qr_svg("http://192.168.1.45:8765/m")     -> SVG text with a 4-module quiet zone
"""

from __future__ import annotations

# Level M per version: (EC codewords per block, [(block count, data codewords per block), ...])
_BLOCKS = {
    1: (10, [(1, 16)]),
    2: (16, [(1, 28)]),
    3: (26, [(1, 44)]),
    4: (18, [(2, 32)]),
    5: (24, [(2, 43)]),
    6: (16, [(4, 27)]),
    7: (18, [(4, 31)]),
    8: (22, [(2, 38), (2, 39)]),
    9: (22, [(3, 36), (2, 37)]),
    10: (26, [(4, 43), (1, 44)]),
}
_ALIGN = {
    1: [],
    2: [6, 18],
    3: [6, 22],
    4: [6, 26],
    5: [6, 30],
    6: [6, 34],
    7: [6, 22, 38],
    8: [6, 24, 42],
    9: [6, 26, 46],
    10: [6, 28, 50],
}
_EC_M = 0b00  # the two format bits for level M

_MASKS = [
    lambda x, y: (x + y) % 2 == 0,
    lambda x, y: y % 2 == 0,
    lambda x, y: x % 3 == 0,
    lambda x, y: (x + y) % 3 == 0,
    lambda x, y: (x // 3 + y // 2) % 2 == 0,
    lambda x, y: x * y % 2 + x * y % 3 == 0,
    lambda x, y: (x * y % 2 + x * y % 3) % 2 == 0,
    lambda x, y: ((x + y) % 2 + x * y % 3) % 2 == 0,
]


def _data_capacity(version: int) -> int:
    _, groups = _BLOCKS[version]
    return sum(n * k for n, k in groups)


def _gf_mul(x: int, y: int) -> int:
    z = 0
    for i in range(7, -1, -1):
        z = (z << 1) ^ ((z >> 7) * 0x11D)
        z ^= ((y >> i) & 1) * x
    return z


def _rs_divisor(degree: int) -> list[int]:
    result = [0] * (degree - 1) + [1]
    root = 1
    for _ in range(degree):
        for j in range(degree):
            result[j] = _gf_mul(result[j], root)
            if j + 1 < degree:
                result[j] ^= result[j + 1]
        root = _gf_mul(root, 0x02)
    return result


def _rs_remainder(data: list[int], divisor: list[int]) -> list[int]:
    result = [0] * len(divisor)
    for b in data:
        factor = b ^ result.pop(0)
        result.append(0)
        for i, coef in enumerate(divisor):
            result[i] ^= _gf_mul(coef, factor)
    return result


def _codewords(payload: bytes) -> tuple[int, list[int]]:
    """Picks the smallest version that fits and returns it with the interleaved codewords."""
    for version in range(1, 11):
        count_bits = 8 if version < 10 else 16
        cap_bits = _data_capacity(version) * 8
        if 4 + count_bits + 8 * len(payload) <= cap_bits:
            break
    else:
        raise ValueError(f"text too long for a QR code here ({len(payload)} bytes)")

    bits: list[int] = []

    def put(value: int, n: int) -> None:
        bits.extend((value >> i) & 1 for i in range(n - 1, -1, -1))

    put(0b0100, 4)  # byte mode
    put(len(payload), count_bits)
    for b in payload:
        put(b, 8)
    put(0, min(4, cap_bits - len(bits)))  # terminator
    put(0, -len(bits) % 8)
    data = [int("".join(map(str, bits[i : i + 8])), 2) for i in range(0, len(bits), 8)]
    pad = 0xEC
    while len(data) < cap_bits // 8:
        data.append(pad)
        pad ^= 0xEC ^ 0x11

    ec_len, groups = _BLOCKS[version]
    divisor = _rs_divisor(ec_len)
    blocks: list[list[int]] = []
    pos = 0
    for n, k in groups:
        for _ in range(n):
            blocks.append(data[pos : pos + k])
            pos += k
    ecs = [_rs_remainder(b, divisor) for b in blocks]
    out = [b[i] for i in range(max(len(b) for b in blocks)) for b in blocks if i < len(b)]
    out += [e[i] for i in range(ec_len) for e in ecs]
    return version, out


def _bch(value: int, poly: int, poly_bits: int) -> int:
    rem = value << (poly_bits - 1)
    for i in range(rem.bit_length() - 1, poly_bits - 2, -1):
        if (rem >> i) & 1:
            rem ^= poly << (i - poly_bits + 1)
    return (value << (poly_bits - 1)) | rem


class _Grid:
    def __init__(self, version: int):
        self.version = version
        self.size = 17 + 4 * version
        self.dark = [[False] * self.size for _ in range(self.size)]
        self.func = [[False] * self.size for _ in range(self.size)]

    def set_func(self, x: int, y: int, dark: bool) -> None:
        self.dark[y][x] = dark
        self.func[y][x] = True

    def draw_function_patterns(self) -> None:
        n = self.size
        for i in range(n):
            self.set_func(6, i, i % 2 == 0)
            self.set_func(i, 6, i % 2 == 0)
        for cx, cy in ((3, 3), (n - 4, 3), (3, n - 4)):
            for dy in range(-4, 5):
                for dx in range(-4, 5):
                    x, y = cx + dx, cy + dy
                    if 0 <= x < n and 0 <= y < n:
                        self.set_func(x, y, max(abs(dx), abs(dy)) not in (2, 4))
        align = _ALIGN[self.version]
        last = len(align) - 1
        for i, ay in enumerate(align):
            for j, ax in enumerate(align):
                if (i, j) in ((0, 0), (0, last), (last, 0)):
                    continue
                for dy in range(-2, 3):
                    for dx in range(-2, 3):
                        self.set_func(ax + dx, ay + dy, max(abs(dx), abs(dy)) != 1)
        self.draw_format(0)  # reserves the area; drawn again once the mask is chosen
        if self.version >= 7:
            bits = _bch(self.version, 0x1F25, 13)
            for i in range(18):
                a, b = n - 11 + i % 3, i // 3
                bit = bool((bits >> i) & 1)
                self.set_func(a, b, bit)
                self.set_func(b, a, bit)

    def draw_format(self, mask: int) -> None:
        n = self.size
        bits = _bch((_EC_M << 3) | mask, 0x537, 11) ^ 0x5412
        bit = [bool((bits >> i) & 1) for i in range(15)]
        for i in range(6):
            self.set_func(8, i, bit[i])
        self.set_func(8, 7, bit[6])
        self.set_func(8, 8, bit[7])
        self.set_func(7, 8, bit[8])
        for i in range(9, 15):
            self.set_func(14 - i, 8, bit[i])
        for i in range(8):
            self.set_func(n - 1 - i, 8, bit[i])
        for i in range(8, 15):
            self.set_func(8, n - 15 + i, bit[i])
        self.set_func(8, n - 8, True)  # the dark module

    def draw_data(self, codewords: list[int]) -> None:
        n = self.size
        total = len(codewords) * 8
        i = 0
        right = n - 1
        while right >= 1:
            if right == 6:
                right = 5
            upward = ((right + 1) & 2) == 0
            for vert in range(n):
                y = n - 1 - vert if upward else vert
                for x in (right, right - 1):
                    if not self.func[y][x] and i < total:
                        self.dark[y][x] = bool((codewords[i >> 3] >> (7 - (i & 7))) & 1)
                        i += 1
            right -= 2

    def apply_mask(self, mask: int) -> None:
        fn = _MASKS[mask]
        for y in range(self.size):
            for x in range(self.size):
                if not self.func[y][x] and fn(x, y):
                    self.dark[y][x] = not self.dark[y][x]

    def penalty(self) -> int:
        n = self.size
        m = self.dark
        score = 0
        lines = [m[y] for y in range(n)] + [[m[y][x] for y in range(n)] for x in range(n)]
        finder_a = [True, False, True, True, True, False, True, False, False, False, False]
        finder_b = finder_a[::-1]
        for line in lines:
            run = 1
            for k in range(1, n + 1):
                if k < n and line[k] == line[k - 1]:
                    run += 1
                    continue
                if run >= 5:
                    score += 3 + run - 5
                run = 1
            for k in range(n - 10):
                window = line[k : k + 11]
                if window == finder_a or window == finder_b:
                    score += 40
        for y in range(n - 1):
            for x in range(n - 1):
                if m[y][x] == m[y][x + 1] == m[y + 1][x] == m[y + 1][x + 1]:
                    score += 3
        dark = sum(map(sum, m))
        total = n * n
        score += ((abs(dark * 20 - total * 10) + total - 1) // total - 1) * 10
        return score


def qr_matrix(data: str) -> list[list[bool]]:
    """Rows of modules, True = dark, without the quiet zone. Raises ValueError if too long."""
    version, codewords = _codewords(data.encode("utf-8"))
    best: _Grid | None = None
    best_score = 0
    for mask in range(8):
        g = _Grid(version)
        g.draw_function_patterns()
        g.draw_data(codewords)
        g.apply_mask(mask)
        g.draw_format(mask)
        score = g.penalty()
        if best is None or score < best_score:
            best, best_score = g, score
    assert best is not None
    return best.dark


def qr_svg(data: str, border: int = 4) -> str:
    """SVG of the QR, one unit per module, black on white, with a quiet zone of `border`."""
    m = qr_matrix(data)
    size = len(m) + 2 * border
    path = "".join(
        f"M{x + border},{y + border}h1v1h-1z"
        for y, row in enumerate(m)
        for x, dark in enumerate(row)
        if dark
    )
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {size} {size}" '
        f'shape-rendering="crispEdges"><rect width="100%" height="100%" fill="#fff"/>'
        f'<path d="{path}" fill="#000"/></svg>'
    )
