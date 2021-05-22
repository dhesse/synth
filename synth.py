from dataclasses import dataclass, field
import numpy
import sounddevice as sd
from typing import List
from itertools import cycle

@dataclass
class Oscillator:

    frequency: int
    sample_rate: int = 44100
    amplitude: float = 1

    def get(self, start_frame: int, frames: int) -> numpy.ndarray:
        raise NotImplementedError()

@dataclass
class SinGen(Oscillator):
    
    def get(self, start_frame: int, frames: int) -> numpy.ndarray:
        t = numpy.arange(start_frame, start_frame + frames) / self.sample_rate
        t = t.reshape(-1, 1)
        return self.amplitude*numpy.sin(2*numpy.pi*self.frequency*t)

@dataclass
class ToothGen(Oscillator):

    def get(self, start_frame: int, frames: int) -> numpy.ndarray:
        t = numpy.arange(start_frame, start_frame + frames) / self.sample_rate * self.frequency
        t = t.reshape(-1, 1)
        return self.amplitude * (t - numpy.round(t))

@dataclass
class PulseGen(Oscillator):

    def get(self, start_frame: int, frames: int) -> numpy.ndarray:
        t = numpy.arange(start_frame, start_frame + frames) / self.sample_rate * self.frequency
        t = t.reshape(-1, 1)
        return self.amplitude * (t > numpy.round(t))

@dataclass
class NoiseGen(Oscillator):

    def get(self, start_frame: int, frames: int) -> numpy.ndarray:
        return self.amplitude * numpy.random.uniform(-1.,1.,frames).reshape(-1, 1)

@dataclass
class Envelope():

    attack: float = 0.02
    decay: float = 0.02
    sustain: float = 0.9
    release: float = 0.02
    duration: float = 0.1
    sample_rate: int = 44100
    buf: numpy.ndarray = field(init=None)

    def calculate_buffer(self):
        a = int(self.attack*self.sample_rate)
        d = int(self.decay*self.sample_rate)
        s = int((self.duration - self.attack - self.release)
             * self.sample_rate)
        r = int(self.release*self.sample_rate)
        self.buf = numpy.zeros(a + d + s + r)
        self.buf[:a] = numpy.linspace(0, 1, a)
        self.buf[a:a+d] = numpy.linspace(1, self.sustain, d)
        self.buf[a+d:a+d+s] = self.sustain
        self.buf[a+d+s:] = numpy.linspace(self.sustain, 0, r)
        self.buf = self.buf.reshape(-1, 1)

    def __post_init__(self):
        self.calculate_buffer()
    
    def get(self, start_frame: int, frames: int) -> numpy.ndarray:
        if start_frame >= len(self.buf):
            return numpy.zeros((frames, 1))
        if start_frame + frames > len(self.buf):
            res = numpy.zeros((frames, 1))
            res[:len(self.buf)-start_frame] = self.buf[start_frame:]
            return res.reshape(-1, 1)
        return self.buf[start_frame:start_frame + frames]
    

@dataclass
class Note:

    gens: List[Oscillator] = field(default_factory= lambda: [SinGen(220), ToothGen(220)])
    envelope: Envelope = Envelope(duration=1)

    @property
    def duration_frames(self) -> int:
        return int(self.envelope.duration*self.envelope.sample_rate)

    def get(self, start_frame: int, frames: int) -> numpy.ndarray:
        return (sum(gen.get(start_frame, frames) for gen in self.gens) 
           / len(self.gens) * self.envelope.get(start_frame, frames))

def n_to_freq(n: int) -> float:
    return 2**((n - 49)/12)*440

PETER_GUNN = cycle([
    Note([PulseGen(n_to_freq(n), amplitude=0.2),
          ToothGen(n_to_freq(n), amplitude=0.8),
          NoiseGen(0, amplitude=0.01)])
    for n in [21, 21, 23, 21, 24, 21, 26, 25]
])

if __name__ == "__main__":

    notes = PETER_GUNN
    note = next(notes)
    bpm = 180
    bps = bpm / 60
    sample_rate = 44100
    fpn = int(sample_rate / bps)
    idx = 0
    def callback(outdata, frames, time, status):
        global idx
        global note
        global notes
        start_idx = 0
        while idx + frames > fpn:
            frames_left_on_note = fpn - idx
            outdata[start_idx:start_idx + frames_left_on_note] = note.get(idx, frames_left_on_note)
            start_idx += frames_left_on_note
            note = next(notes)
            frames -= frames_left_on_note
            idx = 0
        outdata[start_idx:] = note.get(idx, frames)
        idx += frames
    
    with sd.OutputStream(callback=callback, channels=1):
        print('#' * 80)
        print('press Return to quit')
        print('#' * 80)
        input()


    