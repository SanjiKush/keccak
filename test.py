import hashlib
import keccak

input = "test"

hex_original = hashlib.sha3_256(input.encode('utf-8')).hexdigest()
print(hex_original)


hex_perso = keccak.keccak_256(input)
print(hex_perso)
