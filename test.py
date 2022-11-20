import unittest
import hashlib
import ktemp


class TestKeccak(unittest.TestCase):

    def test_lib_vs_moi(self):
        aTester = ["test", "", "1234", "fewofnewjkfnekfnekjnfwej"]
        for aTest in aTester:
            print("Testing: " + aTest)
            lib = hashlib.sha3_256(aTest.encode()).hexdigest()
            moi = str(ktemp.KeccakHash(rate=200-(512 // 8),
                                       dsbyte=0x06).update(aTest.encode('utf-8')).hexdigest()).replace("b'", "").replace("'", "")
            print("Hashlib: " + lib)
            print("Perso:\t " + moi + "\n")

            self.assertEqual(lib, moi)


if __name__ == '__main__':
    unittest.main()
