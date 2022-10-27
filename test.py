from fnmatch import translate
import numpy as np
import hashlib
import keccak

input = "test"


# def hex_to_array(hexnum, w=64):
#     # Convert a hexstring to a 1-dimensional numpy array
#     bitstring = '{0:064b}'.format(hexnum)
#     bitstring = bitstring[-w:]
#     array = np.array([int(bitstring[i]) for i in range(w)])
#     return array


# a = 0x0000000000000001


# def translateToBits(text):
#     array = list(text)
#     bitarray = []
#     for char in array:
#         bits = bin(ord(char))[2:].zfill(8)
#         bits = bits[::-1]  # Convert to Little-endian
#         for bit in bits:
#             bitarray.append(int(bit))


# print(hex_to_array(a))
# print(translateToBits(a))

hex_original = hashlib.sha3_256(input.encode('utf-8')).hexdigest()
print("original:\n" + hex_original)

print("\nperso:")
hex_perso = keccak.keccak_256(input)
# print(hex_perso)
