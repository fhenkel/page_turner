import pyaudio
import wave

import numpy as np

from cyolo_score_following.utils.data_utils import SAMPLE_RATE, FRAME_SIZE


class AudioStream(object):

    def __init__(self, audio_path=None):

        self.audio_path = audio_path

        self.pa = pyaudio.PyAudio()
        self.wave_file = None

        if self.audio_path is None:
            # live audio input
            self.audio_stream = self.pa.open(format=pyaudio.paInt16, channels=1, rate=SAMPLE_RATE,
                                             input=True, frames_per_buffer=FRAME_SIZE // 2)

        else:
            # read from file
            self.wave_file = wave.open(self.audio_path, 'rb')
            self.audio_stream = self.pa.open(format=self.pa.get_format_from_width(self.wave_file.getsampwidth()),
                                             channels=self.wave_file.getnchannels(),
                                             rate=self.wave_file.getframerate(),
                                             output=True)

    def get(self):

        if self.wave_file is None:
            data = self.audio_stream.read(FRAME_SIZE // 2)
        else:
            data = self.wave_file.readframes(FRAME_SIZE // 2)

        if len(data) <= 0:
            data = None

        if data is not None:

            if self.wave_file is not None:
                # write data to audio stream
                self.audio_stream.write(data)

            data = np.frombuffer(data, dtype=np.int16) / 2 ** 15

        return data

    def close(self):
        if self.wave_file is not None:
            self.wave_file.close()

        self.audio_stream.stop_stream()
        self.audio_stream.close()
        self.pa.terminate()
