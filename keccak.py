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


import sys
from bitstring import BitArray
import os
import numpy as np

# l = 6 car 64 bits
l = 6
# size
w = 2 ** l
# size of matrix
n = 5
# rounds
rounds = 12 + 2 * l
# Constant for Iota
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


def parity(n):
    parity = 0
    while n:
        parity = ~parity
        n = n & (n - 1)
    return parity


def routine_teta(matrix):
    """
    a[i][j][k] = a[i][j][k] OU parite(a[0..4][j-1][k]) OU parite(a[0..4][j+1][k-1])
    """
    for i in range(n):
        for j in range(n):
            for k in range(w):
                # matrix[i][j][k] = matrix[i][j][k] ^ parity(
                #     matrix[0][j-1][k]) ^ parity(matrix[0][j+1][k-1])

                C = sum([matrix[(i-1) % 5][ji][k]
                        for ji in range(5)]) % 2
                # XOR=mod2 5 bit column "to the right"  and one position "to the front" of the original bit
                D = sum([matrix[((i+1) % 5)][ji][(k-1) % 64]
                        for ji in range(5)]) % 2
                # XORing original bit with A and B
                temp = C + D + matrix[i][j][k] % 2
                matrix[i][j][k] = temp

    print(matrix)
    return matrix


# def theta(array, w=64):
#     # For each column, XOR the parity of two adjacent columns
#     array_prime = array.copy()
#     C, D = np.zeros([5, w], dtype=int), np.zeros([5, w], dtype=int)
#     for x in range(5):
#         for y in range(5):
#             # C[x] is a lane, each entry represents the column parity
#             C[x] ^= array[x][y]
#     for x in range(5):
#         D[x] = C[(x-1) % 5] ^ np.roll(C[(x+1) % 5], 1)  # D[x] is a placeholder
#     for x in range(5):
#         for y in range(5):
#             array_prime[x][y] ^= D[x]  # For each lane, XOR the value of D[x]
#     # print(array_prime)
#     return array_prime


def routine_rho(matrix):
    """
    a[i][j][k]=a[i][j][k − (t+1)(t+2)/2]
    """
    rhomatrix = [[0, 36, 3, 41, 18], [1, 44, 10, 45, 2], [
        62, 6, 43, 15, 61], [28, 55, 25, 21, 56], [27, 20, 39, 8, 14]]
    rhom = np.array(rhomatrix, dtype=int)
    # print(rhom)
    for i in range(n):
        for j in range(n):
            for k in range(w):
                matrix[i][j][k] = matrix[i][j][k - rhom[i][j]]
    return matrix


def routine_pi(matrix):
    """
    a[3i+2j][i] = a[i][j]
    """
    for i in range(n):
        for j in range(n):
            matrix[(3*i + 2*j) % 5][i] = matrix[i][j]
    return matrix


def routine_khi(matrix):
    """
    a[i][j][k]=a[i][j][k] OR ¬(a[i][j+1][k]) & (a[i][j+2][k])
    """
    for i in range(n):
        for j in range(n):
            for k in range(w):
                matrix[i][j][k] = (matrix[i][j][k]+(((matrix[(i + 1) % 5][j][k] + 1) %
                                                     2) * (matrix[(i + 2) % 5][j][k]))) % 2

    return matrix


def routine_iota(matrix, rounds):
    """
    a[0][0][2^m-1] est Xoré avec le bit numéroté m+7n d'une séquence LFSR de degré 8
    """
    for m in range(l):
        matrix[0][0][2**m-1] = matrix[0][0][2**m -
                                            1] or KeccakF1600RoundConstants[rounds]

    return matrix


def translateToBits(text):
    array = list(text)
    bitarray = []
    for char in array:
        bits = bin(ord(char))[2:].zfill(8)
        bits = bits[::-1]  # Convert to Little-endian
        for bit in bits:
            bitarray.append(int(bit))

    res = ''.join([str(x) for x in bitarray])
    res += '01100000'  # SHA3 Suffix
    return res


def padding(text):
    """
    To ensure the message can be evenly divided into r-bit blocks, padding is required. 
    SHA-3 uses the pattern 10*1 in its padding function: 
    a 1 bit, followed by zero or more 0 bits (maximum r − 1) and a final 1 bit. 
    """
    # Convert to bits
    bitstr = translateToBits(text)

    while len(bitstr) % (r-1) != 0:
        bitstr += '0'
    # Add 1
    bitstr += '1'

    return bitstr


def init_matrix(input):
    """
    L’état interne de cette fonction sera vu comme une matrice de dimension 5×5×w.
    a[i][j][k] sera le bit (i∗5+j)∗w+k de l’entrée. 
    Le calcul des indices est effectué modulo 5 pour les deux premières dimensions, et modulo w pour la troisième
    """

    bit_str = padding(input)

    state_array = np.zeros([5, 5, w], dtype=int)
    for x in range(5):
        for y in range(5):
            for z in range(w):
                if (w*(5*x+y)+z) < len(bit_str):
                    state_array[y][x][z] = int(bit_str[w*(5*x+y)+z])

    state = np.zeros(1600, dtype=int).reshape(n, n, w)
    state = np.bitwise_xor(state, state_array)

    return state


def translateFromBits(bits):
    copy = [bit.replace('9223372039002292224', '1') for bit in bits]
    string = ''.join(copy)
    d = BitArray(bin=string)

    return d.hex[:size]


def ret_res(matrix):
    res = np.zeros(1600)

    for i in range(n):
        for j in range(n):
            for k in range(w):
                res[64*(5*j+i)+k] = matrix[i][j][k]
    return res


def keccak_256(data):

    matrix = init_matrix(data)
    # print(matrix)

    for round in range(rounds):
        matrix = routine_iota(routine_khi(routine_pi(
            routine_rho(routine_teta(matrix)))), round)

    res = ret_res(matrix)
    res = [str(int(i)) for i in res]
    # print(res)
    res = translateFromBits(res)
    return res


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python keccak.py <file>")
        sys.exit(1)

    if not os.path.isfile(sys.argv[1]):
        print("Error: %s is not a file" % sys.argv[1])
        print("Usage: python keccak.py <file>")
        sys.exit(1)

    file = sys.argv[1]
    with open(file, 'r') as f:
        data = f.read()

    keccak_256(data)
