from dataclasses import dataclass, field
import numpy
import sounddevice as sd
from typing import List
from itertools import cycle
import soundfile as sf


@dataclass
class Sound:

    sample_rate: int

    def get(self, start_frame: int, frames: int) -> numpy.ndarray:
        raise NotImplementedError()

    def __add__(self, other):
        return CombinedSound(self, other)

@dataclass
class CombinedSound(Sound):
    
    first: Sound
    second: Sound

    def get(self, start_frame: int, frames: int) -> numpy.ndarray:
        return self.first.get(start_frame, frames) + self.second.get(start_frame, frames)


@dataclass
class Oscillator(Sound):

    frequency: int
    amplitude: float = 1

    

@dataclass
class Echo:

    sound: Oscillator
    delay: int
    decay: float
    frames: numpy.ndarray = field(init=False)

    def __post_init__(self):
        self.frames = numpy.zeros((int(self.sound.sample_rate*self.delay), 1))

    def get(self, start_frame:int, frames: int) -> numpy.ndarray:
        res = self.sound.get(start_frame, frames)
        res += self.frames[:frames,:]
        self.frames[:frames,:] = 0#*= self.decay
        self.frames[:frames,:] += self.decay*res
        self.frames = numpy.roll(self.frames, -frames)
        return res

@dataclass
class WaveGen:

    samples: numpy.ndarray
    sample_rate: int = 44100

    def get(self, start_frame: int, frames: int) -> numpy.ndarray:
        high = start_frame + frames
        if start_frame > self.samples.shape[0]:
            return numpy.zeros((frames, 1))
        if high > self.samples.shape[0]:
            result = numpy.zeros((frames, 1))
            result[:self.samples.shape[0] - start_frame,:] = self.samples[start_frame:,0].reshape(-1, 1)
            return result
        return self.samples[start_frame:high,0].reshape(-1, 1)

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
    Note([PulseGen(44100, n_to_freq(n), amplitude=0.1),
          ToothGen(44100, n_to_freq(n), amplitude=0.5),
          NoiseGen(44100, 0, amplitude=0.07)])
    for n in [21, 21, 23, 21, 24, 21, 26, 25]
])

SAMPLES = {"K": "/Users/dirkhesse/code/synth/K/428__tictacshutup__prac-kick.wav",
"H": "/Users/dirkhesse/code/synth/H/426__tictacshutup__prac-hat.wav",
"S": "/Users/dirkhesse/code/synth/S/447__tictacshutup__prac-snare.wav"
} 

BOOM_CHAH = cycle([WaveGen(sf.read(SAMPLES[i])[0]) for i in "KHSH"])

@dataclass
class Config:
    bpm: int
    sample_rate: int
    bar_length: int
    note_resolution: int

    @property
    def bps(self) -> float:
        return self.bpm / 60

    @property
    def fpn(self) -> float:
        """Frames per note"""
        return int(self.sample_rate / self.bps
                   / self.note_resolution * self.bar_length)
    

if __name__ == "__main__":

    sounds = zip(BOOM_CHAH, PETER_GUNN)
    notes = next(sounds)
    cfg = Config(bpm = 100, sample_rate = 44100,
                 bar_length = 4, note_resolution = 8)
    fpn = cfg.fpn
    idx = 0
    #echo = Echo(note, 0.5, 0.12)
    def callback(outdata, frames, time, status):
        global idx
        global note
        global notes
        global echo
        start_idx = 0
        while idx + frames > fpn:
            frames_left_on_note = fpn - idx
            #data = echo.get(idx, frames_left_on_note)
            data = sum(note.get(idx, frames_left_on_note) for note in notes)
            outdata[start_idx:start_idx + frames_left_on_note] = data
            start_idx += frames_left_on_note
            #echo.sound = next(sounds)[0]
            notes = next(sounds)
            frames -= frames_left_on_note
            idx = 0
        #data = echo.get(idx, frames)
        data = sum(note.get(idx, frames) for note in notes)
        outdata[start_idx:] = data
        idx += frames
    
    with sd.OutputStream(callback=callback, channels=1):
        print('#' * 80)
        print('press Return to quit')
        print('#' * 80)
        input()


    
