#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np

# l = 6 car 64 bits
l = 6
# size
w = 2 ** l
# size of matrix
n = 5
# rounds
rounds = 12 + 2 * l

KeccakF1600RoundConstants = [
    0x0000000000000001, 0x000000008000808B, 0x0000000000008082,
    0x800000000000008B, 0x800000000000808A, 0x8000000000008089,
    0x8000000080008000, 0x8000000000008003, 0x000000000000808B,
    0x8000000000008002, 0x0000000080000001, 0x8000000000000080,
    0x8000000080008081, 0x000000000000800A, 0x8000000000008009,
    0x800000008000000A, 0x000000000000008A, 0x8000000080008081,
    0x0000000000000088, 0x8000000000008080, 0x0000000080008009,
    0x0000000080000001, 0x000000008000000A, 0x8000000080008008]

# Une fois toutes les permutations effectuées, l’empreinte résultat est constituée des n premiers bits de l’état.
size = 64
# Capacity
c = 512

# Block Size
r = 1600 - c


def hex_to_array(hexnum):
    # Convert a hexstring to a 1-dimensional numpy array
    bitstring = '{0:064b}'.format(hexnum)
    bitstring = bitstring[-w:]
    array = np.array([int(bitstring[i]) for i in range(w)])
    array = np.flip(array)
    return array


def ROL64(a, n):
    return ((a >> 64 - n % 64) + (a << n % 64)) % (1 << 64)


def teta(lanes):

    C = [lanes[x][0] ^ lanes[x][1] ^ lanes[x][2] ^ lanes[x][3]
         ^ lanes[x][4] for x in range(5)]
    D = [C[(x + 4) % 5] ^ ROL64(C[(x + 1) % 5], 1) for x in
         range(5)]
    lanes = [[lanes[x][y] ^ D[x] for y in range(5)] for x in
             range(5)]
    return lanes


def rho_pi(lanes):
    (x, y) = (1, 0)
    current = lanes[x][y]
    for t in range(24):
        (x, y) = (y, (2 * x + 3 * y) % 5)
        (current, lanes[x][y]) = (lanes[x][y], ROL64(current, (t
                                                               + 1) * (t + 2) // 2))
    return lanes


def khi(lanes):
    for y in range(5):
        T = [lanes[x][y] for x in range(5)]
        for x in range(5):
            lanes[x][y] = T[x] ^ ~T[(x + 1) % 5] & T[(x + 2) % 5]
    return lanes


def iota(lanes, round):
    lsfr = hex_to_array(KeccakF1600RoundConstants[round])
    for m in range(l):
        lanes[0][0] ^= lsfr[m + 7 * n]
    return lanes


def KeccakF1600onLanes(lanes):
    for round in range(24):
        lanes = iota(khi(rho_pi(teta(lanes))), round)

    return lanes


def load64(b):
    return sum(b[i] << 8 * i for i in range(8))


def store64(a):
    return list((a >> 8 * i) % 256 for i in range(8))


def KeccakF1600(state):
    lanes = [[load64(state[8 * (x + 5 * y):8 * (x + 5 * y) + 8])
             for y in range(5)] for x in range(5)]
    lanes = KeccakF1600onLanes(lanes)
    state = bytearray(200)
    for x in range(5):
        for y in range(5):
            state[8 * (x + 5 * y):8 * (x + 5 * y) + 8] = \
                store64(lanes[x][y])
    return state


def Keccak(rate, capacity, inputBytes, delimitedSuffix, outputByteLen):
    outputBytes = bytearray()
    state = bytearray([0 for i in range(200)])
    rateInBytes = rate // 8
    blockSize = 0
    if rate + capacity != 1600 or rate % 8 != 0:
        return
    inputOffset = 0

    # === Absorb all the input blocks ===

    while inputOffset < len(inputBytes):
        blockSize = min(len(inputBytes) - inputOffset, rateInBytes)
        for i in range(blockSize):
            state[i] = state[i] ^ inputBytes[i + inputOffset]
        inputOffset = inputOffset + blockSize
        if blockSize == rateInBytes:
            state = KeccakF1600(state)
            blockSize = 0

    # === Do the padding and switch to the squeezing phase ===

    state[blockSize] = state[blockSize] ^ delimitedSuffix
    if delimitedSuffix & 0x80 != 0 and blockSize == rateInBytes - 1:
        state = KeccakF1600(state)
    state[rateInBytes - 1] = state[rateInBytes - 1] ^ 0x80
    state = KeccakF1600(state)

    # === Squeeze out all the output blocks ===

    while outputByteLen > 0:
        blockSize = min(outputByteLen, rateInBytes)
        outputBytes = outputBytes + state[0:blockSize]
        outputByteLen = outputByteLen - blockSize
        if outputByteLen > 0:
            state = KeccakF1600(state)
    return outputBytes


def SHA3_256(inputBytes):
    return Keccak(1088, 512, inputBytes, 0x06, 256 // 8)


print(SHA3_256(b'hello').hex())
