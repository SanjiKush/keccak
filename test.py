import hashlib
import k

input = "test"

aTester = ["test", "", "1234", "fewofnewjkfnekfnekjnfwej"]

for aTest in aTester:
    print("Testing: " + aTest)
    print("Hashlib: " + hashlib.sha3_256(aTest.encode()).hexdigest())
    print("Perso:\t " + str(k.KeccakHash(rate=200-(512 // 8),
          dsbyte=0x06).update(aTest.encode('utf-8')).hexdigest()).replace("b'", "").replace("'", "") + "\n")

