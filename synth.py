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
    note_length: float
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

    buf: numpy.ndarray = field(init=None)
    attack: float = 0.02
    decay: float = 0.02
    sustain: float = 0.9
    release: float = 0.02
    duration: float = 0.1
    sample_rate: int = 44100

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
    note_length: float = 8

    @property
    def duration_frames(self) -> int:
        return int(self.envelope.duration*self.envelope.sample_rate)

    def get(self, start_frame: int, frames: int) -> numpy.ndarray:
        return (sum(gen.get(start_frame, frames) for gen in self.gens) 
           / len(self.gens) * self.envelope.get(start_frame, frames))

def n_to_freq(n: int) -> float:
    return 2**((n - 49)/12)*440

def PeterGunn():
    return cycle([
    Note([PulseGen(44100, n_to_freq(f), amplitude=0.1),
          ToothGen(44100, n_to_freq(f), amplitude=0.5),
          NoiseGen(44100, 0, amplitude=0.07)], note_length=l)
    for f,l in [[21,8], [21,8], [23,8], [21,8], [24,16], [25,16], [21,8], [26,8], [25,8]]
])

def PeterGunnLead():
    return cycle([
    Note([PulseGen(44100, n_to_freq(f)/2, amplitude=0.2),
          ToothGen(44100, n_to_freq(f), amplitude=0.5),
          ToothGen(44100, n_to_freq(f)*1.5, amplitude=0.3),
          ToothGen(44100, n_to_freq(f)*2, amplitude=0.2),
          NoiseGen(44100, 0, amplitude=0.07)], note_length=l, envelope=Envelope(attack=0.2, decay=0.5, release=0.3, sustain=0.8, duration=2/l))
    for (f,l) in [[0,1/2], [55, 8/7], [52, 8/5], [0,8/4],  [55, 8/7], [64, 8/2], [58, 8/7], [0,8], [52, 8], [55, 8], [57, 8], [58, 8/4*3], [58, 8/4*3], [58, 8/4*3],
    [58, 8/4*3], [57, 8/4*3], [55, 8/4*3], [52, 8/4*3], [50, 8/4*3], [52, 8/4*3], [48, 8], [49, 8/7]]
])

SAMPLES = {"K": "K/428__tictacshutup__prac-kick.wav",
"H": "H/426__tictacshutup__prac-hat.wav",
"S": "S/447__tictacshutup__prac-snare.wav"
} 

def BoomChah():
    return cycle([WaveGen(samples=sf.read(SAMPLES[i])[0], note_length=8) for i in "KHSHKKSH"])

@dataclass
class Config:
    bpm: int
    sample_rate: int
    bar_length: int

    @property
    def bps(self) -> float:
        return self.bpm / 60

    @property
    def frames_per_bar(self) -> float:
        """Frames per bar"""
        return self.sample_rate / self.bps * self.bar_length


class Synth:

    def __init__(self, channels, cfg):
        self.channels = channels
        self.num_channels = len(channels)
        self.channel_idx = [0 for n in range(self.num_channels)]
        self.channel_notes = [next(c) for c in self.channels]
        self.cfg = cfg

    def __call__(self, outdata, frames, time, status):
        temp_data = numpy.zeros((self.num_channels, frames))

        for chan in range(self.num_channels):
            temp_frames = frames
            start_idx = 0
            curr_note_frames = int(self.cfg.frames_per_bar / self.channel_notes[chan].note_length)
            while self.channel_idx[chan] + temp_frames > curr_note_frames:
                frames_left_on_note = curr_note_frames - self.channel_idx[chan]
                data = self.channel_notes[chan].get(self.channel_idx[chan], frames_left_on_note)
                temp_data[chan][start_idx:start_idx + frames_left_on_note] = numpy.squeeze(data)
                start_idx += frames_left_on_note
                self.channel_notes[chan] = next(self.channels[chan])
                temp_frames -= frames_left_on_note
                self.channel_idx[chan] = 0

            data = self.channel_notes[chan].get(self.channel_idx[chan], temp_frames)
            temp_data[chan][start_idx:] = numpy.squeeze(data)
            self.channel_idx[chan] += temp_frames

        outdata[0:frames] = sum(temp_data).reshape(frames, 1)

if __name__ == "__main__":

    channels = [BoomChah(), PeterGunn(), PeterGunnLead()]
    cfg = Config(bpm = 121, sample_rate = 44100,
                 bar_length = 4)
    callback = Synth(channels, cfg)
    
    with sd.OutputStream(callback=callback, channels=1):
        print('#' * 80)
        print('press Return to quit')
        print('#' * 80)
        input()

