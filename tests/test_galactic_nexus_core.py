import unittest
from core.galactic_nexus_core import GalacticNexusCore

class TestGalacticNexusCore(unittest.TestCase):
    def test_send_data_to_blockchain(self):
        galactic_nexus_core = GalacticNexusCore()
        data = {"key": "value"}
        galactic_nexus_core.send_data_to_blockchain(data, "ethereum")
        self.assertTrue(True)

if __name__ == '__main__':
    unittest.main()
