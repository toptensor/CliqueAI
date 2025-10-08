import math


# 92 ASCII characters. Excludes ", ', and \.
DEFAULT_BASE92_ALPHABET = (
    ''.join(chr(c) for c in range(32, 127) if c not in (34, 39, 92))
)


class GraphCodec:
    """
    Provides graph encoding and decoding functionalities.
    1. adjacency_list <-> adjacency_matrix
    2. adjacency_matrix <-> base92 encoded string (bitwise)
    """

    def __init__(
        self,
        hdr_digits: int = 5,
        chunk_bits: int = 1246,
        chunk_digits: int = 191,
    ):
        # Base92 alphabet
        self.alphabet = DEFAULT_BASE92_ALPHABET
        self.index = {ch: i for i, ch in enumerate(self.alphabet)}

        # Encoding parameters
        self.HDR_DIGITS = hdr_digits
        self.CHUNK_BITS = chunk_bits
        self.CHUNK_DIGITS = chunk_digits

    # Graph structure conversions
    @staticmethod
    def list_to_matrix(adj_list: list[list[int]], n: int) -> list[list[int]]:
        """Convert adjacency list to adjacency matrix"""
        adj_matrix = [[0] * n for _ in range(n)]
        for i, neighbors in enumerate(adj_list):
            for j in neighbors:
                adj_matrix[i][j] = 1
        return adj_matrix

    @staticmethod
    def matrix_to_list(adj_matrix: list[list[int]]) -> list[list[int]]:
        """Convert adjacency matrix to adjacency list"""
        adj_list = []
        for i, row in enumerate(adj_matrix):
            neighbors = [j for j, v in enumerate(row) if v == 1]
            adj_list.append(neighbors)
        return adj_list

    # Base92 helpers
    def _min_digits_for_bits(self, r_bits: int) -> int:
        if r_bits <= 0:
            return 0
        m = math.ceil(r_bits / math.log2(len(self.alphabet)))
        while (len(self.alphabet)**(m - 1) if m > 0 else 0) >= (1 << r_bits):
            m -= 1
        while len(self.alphabet)**m < (1 << r_bits):
            m += 1
        return m

    def _enc_fixed_int(self, val: int, digits: int) -> str:
        out = []
        base = len(self.alphabet)
        for _ in range(digits):
            val, rem = divmod(val, base)
            out.append(self.alphabet[rem])
        if val:
            raise ValueError("Overflow...")
        return ''.join(reversed(out))

    def _dec_fixed_to_int(self, s: str) -> int:
        val = 0
        base = len(self.alphabet)
        for ch in s:
            if ch not in self.index:
                raise ValueError(f"invalid base92 char: {repr(ch)}")
            val = val * base + self.index[ch]
        return val

    # Encode / decode
    def encode_matrix(self, adj_matrix: list[list[int]]) -> str:
        """Encode adjacency matrix to base92 bitwise string"""
        n = len(adj_matrix)
        if any(len(row) != n for row in adj_matrix):
            raise ValueError("adjacency matrix must be n×n")

        # Check diagonal = 0
        if any(adj_matrix[i][i] != 0 for i in range(n)):
            raise ValueError("diagonal must be all 0")

        # Check symmetric
        for i in range(n):
            for j in range(i + 1, n):
                if adj_matrix[i][j] != adj_matrix[j][i]:
                    raise ValueError("adjacency matrix must be symmetric")

        parts = [self._enc_fixed_int(n, self.HDR_DIGITS)]

        chunk_val = 0
        chunk_len = 0

        def flush(digs: int):
            nonlocal chunk_val, chunk_len
            parts.append(self._enc_fixed_int(chunk_val, digs))
            chunk_val = 0
            chunk_len = 0

        for i in range(n):
            for j in range(i + 1, n):
                v = 1 if adj_matrix[i][j] else 0
                chunk_val = (chunk_val << 1) | v
                chunk_len += 1
                if chunk_len == self.CHUNK_BITS:
                    flush(self.CHUNK_DIGITS)

        if chunk_len:
            flush(self._min_digits_for_bits(chunk_len))

        return ''.join(parts)

    def decode_matrix(self, s: str) -> list[list[int]]:
        """Decode base92 bitwise string back to adjacency matrix"""
        if len(s) < self.HDR_DIGITS:
            raise ValueError("truncated header")
        n = self._dec_fixed_to_int(s[:self.HDR_DIGITS])

        total_bits = n * (n - 1) // 2
        full = total_bits // self.CHUNK_BITS
        rem_bits = total_bits % self.CHUNK_BITS
        tail_digits = self._min_digits_for_bits(rem_bits)
        expected = self.HDR_DIGITS + full * self.CHUNK_DIGITS + tail_digits
        if len(s) != expected:
            raise ValueError(f"length mismatch: got {len(s)}, expect {expected} for n={n}")

        pos = self.HDR_DIGITS

        def next_chunk():
            nonlocal pos, full, rem_bits
            if full:
                chunk_str = s[pos:pos + self.CHUNK_DIGITS]
                pos += self.CHUNK_DIGITS
                full -= 1
                return self._dec_fixed_to_int(chunk_str), self.CHUNK_BITS
            if rem_bits:
                m = self._min_digits_for_bits(rem_bits)
                chunk_str = s[pos:pos + m]
                pos += m
                b = rem_bits
                rem_bits = 0
                return self._dec_fixed_to_int(chunk_str), b
            return None

        adj_matrix = [[0] * n for _ in range(n)]
        if total_bits == 0:
            return adj_matrix

        result = next_chunk()
        if result is None:
            raise ValueError("Unexpected end of encoded string")
        val, bits_left = result

        for i in range(n):
            for j in range(i + 1, n):
                if bits_left == 0:
                    result = next_chunk()
                    if result is None:
                        raise ValueError("Unexpected end of encoded string")
                    val, bits_left = result
                bit = (val >> (bits_left - 1)) & 1
                bits_left -= 1
                adj_matrix[i][j] = adj_matrix[j][i] = int(bit)
        return adj_matrix

    def encoded_length_for_n(self, n: int) -> int:
        """Return encoded string length for given n"""
        total_bits = n * (n - 1) // 2
        full = total_bits // self.CHUNK_BITS
        rem = total_bits % self.CHUNK_BITS
        return self.HDR_DIGITS + full * self.CHUNK_DIGITS + self._min_digits_for_bits(rem)
