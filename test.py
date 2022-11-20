import unittest
import hashlib
import keccak

aTester = ["test", "", "1234", "fewofnewjkfnekfnekjnfwej",
           "s0s0s0s", "99999999999999999999999999999", ",.,.", ".,m2jfeiu8"]


class TestKeccak(unittest.TestCase):

    def test_lib_vs_moi(self):
        for test in aTester:
            print("Testing: " + test)
            lib = hashlib.sha3_256(test.encode()).hexdigest()
            moi = str(keccak.KeccakHash(rate=136).update(
                test.encode('utf-8')).hexdigest()).replace("b'", "").replace("'", "")
            print("Hashlib: " + lib)
            print("Perso:\t " + moi + "\n")

            self.assertEqual(lib, moi)


if __name__ == '__main__':
    unittest.main()
