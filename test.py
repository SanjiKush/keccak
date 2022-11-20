import unittest
import hashlib
import keccak


class TestKeccak(unittest.TestCase):

    def test_lib_vs_moi(self):
        aTester = ["test", "", "1234", "fewofnewjkfnekfnekjnfwej",
                   "s0s0s0s", "99999999999999999999999999999", ",.,.", ".,m2jfeiu8"]
        for aTest in aTester:
            print("Testing: " + aTest)
            lib = hashlib.sha3_256(aTest.encode()).hexdigest()
            moi = str(keccak.KeccakHash(rate=136).update(
                aTest.encode('utf-8')).hexdigest()).replace("b'", "").replace("'", "")
            print("Hashlib: " + lib)
            print("Perso:\t " + moi + "\n")

            self.assertEqual(lib, moi)


if __name__ == '__main__':
    unittest.main()
