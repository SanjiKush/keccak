#!/usr/bin/env python3

# Copyright (c) 2022 SALLIOT Mathieu <mathieu.salliot@epita.fr>

# Permission to use, copy, modify, and distribute this software for any
# purpose with or without fee is hereby granted, provided that the above
# copyright notice and this permission notice appear in all copies.

# THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES
# WITH REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF
# MERCHANTABILITY AND FITNESS. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR
# ANY SPECIAL, DIRECT, INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES
# WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN
# ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT OF
# OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.

from binascii import hexlify
import numpy as np


RHO_CTE = [1,  3,  6, 10, 15, 21, 28, 36, 45, 55,  2,
           14, 27, 41, 56,  8, 25, 43, 62, 18, 39, 61, 20, 44]

PI_CTE = [10,  7, 11, 17, 18, 3, 5, 16,  8, 21, 24,
          4, 15, 23, 19, 13, 12, 2, 20, 14, 22,  9,  6, 1]

KECCAK_ROUND_CTE = np.array([
    0x0000000000000001, 0x0000000000008082,
    0x800000000000808a, 0x8000000080008000,
    0x000000000000808b, 0x0000000080000001,
    0x8000000080008081, 0x8000000000008009,
    0x000000000000008a, 0x0000000000000088,
    0x0000000080008009, 0x000000008000000a,
    0x000000008000808b, 0x800000000000008b,
    0x8000000000008089, 0x8000000000008003,
    0x8000000000008002, 0x8000000000000080,
    0x000000000000800a, 0x800000008000000a,
    0x8000000080008081, 0x8000000000008080,
    0x0000000080000001, 0x8000000080008008],
    dtype=np.uint64)

SPONGE_ABSORBING = 1

SPONGE_SQUEEZING = 2


def rol(x, s):
    """Rotate x left by s."""
    return ((np.uint64(x) << np.uint64(s)) ^ (np.uint64(x) >> np.uint64(64 - s)))


class Keccak(object):
    """The Keccak-F[1600] permutation."""

    def __init__(self):
        self.state = np.zeros(25, dtype=np.uint64)

    def Routines(self):
        state = self.state
        bc = np.zeros(5, dtype=np.uint64)

        for i in range(24):
            # Parity
            for x in range(5):
                bc[x] = 0
                for y in range(0, 25, 5):
                    bc[x] ^= state[x + y]

            # Theta
            for x in range(5):
                t = bc[(x + 4) % 5] ^ rol(bc[(x + 1) % 5], 1)
                for y in range(0, 25, 5):
                    state[y + x] ^= t

            # Rho and pi
            t = state[1]
            for x in range(24):
                bc[0] = state[PI_CTE[x]]
                state[PI_CTE[x]] = rol(t, RHO_CTE[x])
                t = bc[0]

            for y in range(0, 25, 5):
                for x in range(5):
                    bc[x] = state[y + x]
                for x in range(5):
                    state[y + x] = bc[x] ^ ((~bc[(x + 1) % 5])
                                            & bc[(x + 2) % 5])

            state[0] ^= KECCAK_ROUND_CTE[i]
        self.state = state


class KeccakHash(object):

    def __init__(self, b='', rate=None, dsbyte=None):
        if rate < 0 or rate > 199:
            raise Exception("Invalid rate.")
        self.rate, self.dsbyte, self.i = rate, dsbyte, 0
        self.k = Keccak()
        self.buf = np.zeros(200, dtype=np.uint8)
        self.absorb(b)
        self.direction = SPONGE_ABSORBING

    def absorb(self, b):
        todo = len(b)
        i = 0
        while todo > 0:
            cando = self.rate - self.i
            willabsorb = min(cando, todo)
            self.buf[self.i:self.i + willabsorb] ^= \
                np.frombuffer(b[i:i+willabsorb], dtype=np.uint8)
            self.i += willabsorb
            if self.i == self.rate:
                self.permute()
            todo -= willabsorb
            i += willabsorb

    def squeeze(self, n):
        tosqueeze = n
        b = b''
        while tosqueeze > 0:
            cansqueeze = self.rate - self.i
            willsqueeze = min(cansqueeze, tosqueeze)
            b += self.k.state.view(dtype=np.uint8)[
                self.i:self.i + willsqueeze].tobytes()
            self.i += willsqueeze
            if self.i == self.rate:
                self.permute()
            tosqueeze -= willsqueeze
        return b

    def pad(self):
        self.buf[self.i] ^= self.dsbyte
        self.buf[self.rate - 1] ^= 0x80
        self.permute()

    def permute(self):
        self.k.state ^= self.buf.view(dtype=np.uint64)
        self.k.Routines()
        self.i = 0
        self.buf[:] = 0

    def update(self, b):
        if self.direction == SPONGE_SQUEEZING:
            self.permute()
            self.direction == SPONGE_ABSORBING
        self.absorb(b)
        return self

    def digest(self):
        if self.direction == SPONGE_ABSORBING:
            self.pad()
        return self.squeeze((200 - self.rate) // 2)

    def hexdigest(self):
        return hexlify(self.digest())


if __name__ == '__main__':
    import sys
    if len(sys.argv) == 1:
        print("Usage: {} <string>".format(sys.argv[0]))
        sys.exit(1)

    # print keccak hash
    res = KeccakHash(rate=200-(512 // 8),
                     dsbyte=0x06).update(sys.argv[1].encode('utf-8')).hexdigest()
    print(str(res).replace("b'", "").replace("'", ""))
