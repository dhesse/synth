import unittest
from synth import BoomChah, PeterGunn, Config, Synth
import numpy

def mk_callback():
    sounds = zip(BoomChah(), PeterGunn())
    cfg = Config(bpm = 100, sample_rate = 44100,
                 bar_length = 4, note_resolution = 8)
    return Synth(sounds, cfg)
      

class IntegrationTest(unittest.TestCase):


    def test_wrap(self):

        N = 1_000_000
        b1, b2 = numpy.zeros((2*N, 1)), numpy.zeros((2*N, 1))

        numpy.random.seed(123)
        mk_callback()(b1[:], 2*N, None, None)

        numpy.random.seed(123)
        callback = mk_callback()
                
        callback(b2[:], 2*N, None, None)
        callback(b2[N:], N, None, None)

        numpy.testing.assert_equal(b1[:N], b2[:N])
