#!/usr/bin/env python
# encoding: utf-8
"""
Copyright (c) 2012, 2013 Sebastian Böck <sebastian.boeck@jku.at>
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.
2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

"""

"""
Please note that this program released together with the paper

"Maximum Filter Vibrato Suppression for Onset Detection"
by Sebastian Böck and Gerhard Widmer
in Proceedings of the 16th International Conference on Digital Audio Effects
(DAFx-13), Maynooth, Ireland, September 2013

is not tuned in any way for speed/memory efficiency. However, it can be used
as a reference implementation for the described onset detection with a maximum
filter for vibrato suppression.

If you use this software, please cite the above paper.

Please send any comments, enhancements, errata, etc. to the main author.

"""

import numpy as np
import scipy.fftpack as fft
from scipy.io import wavfile


class Filter(object):
    """
    Filter Class.

    """
    def __init__(self, ffts, fs, bands=24, fmin=27.5, fmax=16000, equal=False):
        """
        Creates a new Filter object instance.

        :param ffts: number of FFT coefficients
        :param fs: sample rate of the audio file
        :param bands: number of filter bands [default=12]
        :param fmin: the minimum frequency [in Hz, default=27.5]
        :param fmax: the maximum frequency [in Hz, default=16000]
        :param equal: normalize each band to equal energy [default=False]

        """
        # samplerate
        self.fs = fs
        # reduce fmax if necessary
        if fmax > fs / 2:
            fmax = fs / 2
        # get a list of frequencies
        frequencies = self.frequencies(bands, fmin, fmax)
        # conversion factor for mapping of frequencies to spectrogram bins
        factor = (fs / 2.0) / ffts
        # map the frequencies to the spectrogram bins
        frequencies = np.round(np.asarray(frequencies) / factor).astype(int)
        # only keep unique bins
        frequencies = np.unique(frequencies)
        # filter out all frequencies outside the valid range
        frequencies = [f for f in frequencies if f < ffts]
        # number of bands
        bands = len(frequencies) - 2
        assert bands >= 3, 'cannot create filterbank with less than 3 frequencies'
        # init the filter matrix with size: ffts x filter bands
        self.filterbank = np.zeros([ffts, bands], dtype=np.float)
        # process all bands
        for band in range(bands):
            # edge & center frequencies
            start, mid, stop = frequencies[band:band + 3]
            # create a triangular filter
            self.filterbank[start:stop, band] = self.triang(start, mid, stop, equal)

    @staticmethod
    def frequencies(bands, fmin, fmax, a=440):
        """
        Returns a list of frequencies aligned on a logarithmic scale.

        :param bands: number of filter bands per octave
        :param fmin: the minimum frequency [in Hz]
        :param fmax: the maximum frequency [in Hz]
        :param a: frequency of A0 [in Hz, default=440]
        :return a list of frequencies
        Using 12 bands per octave and a=440 corresponding to the MIDI notes.

        """
        # factor 2 frequencies are apart
        factor = 2.0 ** (1.0 / bands)
        # start with A0
        freq = a
        frequencies = [freq]
        # go upwards till fmax
        while freq <= fmax:
            # multiply once more, since the included frequency is a frequency
            # which is only used as the right corner of a (triangular) filter
            freq *= factor
            frequencies.append(freq)
        # restart with a and go downwards till fmin
        freq = a
        while freq >= fmin:
            # divide once more, since the included frequency is a frequency
            # which is only used as the left corner of a (triangular) filter
            freq /= factor
            frequencies.append(freq)
        # sort frequencies
        frequencies.sort()
        # return the list
        return frequencies

    @staticmethod
    def triang(start, mid, stop, equal=False):
        """
        Calculates a triangular window of the given size.

        :param start: starting bin (with value 0, included in the returned filter)
        :param mid: center bin (of height 1, unless norm is True)
        :param stop: end bin (with value 0, not included in the returned filter)
        :param equal: normalize the area of the filter to 1 [default=False]
        :return a triangular shaped filter

        """
        # height of the filter
        height = 1.
        # normalize the height
        if equal:
            height = 2. / (stop - start)
        # init the filter
        triang_filter = np.empty(stop - start)
        # rising edge
        triang_filter[:mid - start] = np.linspace(0, height, (mid - start), endpoint=False)
        # falling edge
        triang_filter[mid - start:] = np.linspace(height, 0, (stop - mid), endpoint=False)
        # return
        return triang_filter


class Wav(object):
    """
    Wav Class is a simple wrapper around scipy.io.wavfile.

    """
    def __init__(self, filename):
        """
        Creates a new Wav object instance of the given file.

        :param filename: name of the .wav file

        """
        # read in the audio
        self.samplerate, self.audio = wavfile.read(filename)
        # scale the audio values to the range -1...1 depending on the audio type
        self.audio = self.audio / float(np.iinfo(self.audio.dtype).max)
        # set the length
        self.samples = np.shape(self.audio)[0]
        self.length = float(self.samples) / self.samplerate
        # set the number of channels
        try:
            # multi channel files
            self.channels = np.shape(self.audio)[1]
        except IndexError:
            # catch mono files
            self.channels = 1

    def attenuate(self, attenuation):
        """
        Attenuate the audio signal.

        :param attenuation: attenuation level given in dB

        """
        self.audio /= np.power(np.sqrt(10.), attenuation / 10.)

    def downmix(self):
        """
        Down-mix the audio signal to mono.

        """
        if self.channels > 1:
            self.audio = np.sum(self.audio, -1) / self.channels

    def normalize(self):
        """
        Normalize the audio signal.

        """
        self.audio /= np.max(self.audio)


class Spectrogram(object):
    """
    Spectrogram Class.

    """
    def __init__(self, wav, window_size=2048, fps=200, online=True):
        """
        Creates a new Spectrogram object instance and performs a STFT on the given audio.

        :param wav: a Wav object
        :param window_size: is the size for the window in samples [default=2048]
        :param fps: is the desired frame rate [default=200]
        :param online: work in online mode (i.e. use only past audio information) [default=True]

        """
        # init some variables
        self.wav = wav
        self.fps = fps
        # derive some variables
        self.hop_size = float(self.wav.samplerate) / float(self.fps)  # use floats so that seeking works properly
        self.frames = int(self.wav.samples / self.hop_size)
        self.ffts = int(window_size / 2)
        self.bins = int(window_size / 2)  # initial number equal to ffts, can change if filters are used
        # init STFT matrix
        self.stft = np.empty([self.frames, self.ffts], np.complex)
        # create windowing function
        self.window = np.hanning(window_size)
        # step through all frames
        for frame in range(self.frames):
            # seek to the right position in the audio signal
            if online:
                # step back a complete window_size after moving forward 1 hop_size
                # so that the current position is at the stop of the window
                seek = int((frame + 1) * self.hop_size - window_size)
            else:
                # step back half of the window_size so that the frame represents the centre of the window
                seek = int(frame * self.hop_size - window_size / 2)
            # read in the right portion of the audio
            if seek >= self.wav.samples:
                # stop of file reached
                break
            elif seek + window_size >= self.wav.samples:
                # stop behind the actual audio stop, append zeros accordingly
                zeros = np.zeros(seek + window_size - self.wav.samples)
                signal = self.wav.audio[seek:]
                signal = np.append(signal, zeros)
            elif seek < 0:
                # start before the actual audio start, pad with zeros accordingly
                zeros = np.zeros(-seek)
                signal = self.wav.audio[0:seek + window_size]
                signal = np.append(zeros, signal)
            else:
                # normal read operation
                signal = self.wav.audio[seek:seek + window_size]
            # multiply the signal with the window function
            signal = signal * self.window
            # perform DFT
            self.stft[frame] = fft.fft(signal)[:self.ffts]
            # next frame
        # magnitude spectrogram
        self.spec = np.abs(self.stft)

    def filter(self, filterbank=None):
        """
        Filter the magnitude spectrogram with a filterbank.

        :param filterbank: Filter object which includes the filterbank [default=None]

        If no filterbank is given a standard one will be used.

        """
        if filterbank is None:
            # construct a standard filterbank
            filterbank = Filter(ffts=self.ffts, fs=self.wav.samplerate).filterbank
        # filter the magnitude spectrogram with the filterbank
        self.spec = np.dot(self.spec, filterbank)
        # adjust the number of bins
        self.bins = np.shape(filterbank)[1]

    def log(self, mul=1, add=1):
        """
        Take the logarithm of the magnitude spectrogram.

        :param mul: multiply the magnitude spectrogram with given value [default=1]
        :param add: add the given value to the magnitude spectrogram [default=1]

        """
        if add <= 0:
            raise ValueError("a positive value must be added before taking the logarithm")
        self.spec = np.log10(mul * self.spec + add)


class SpectralODF(object):
    """
    The SpectralODF class implements most of the common onset detection function
    based on the magnitude or phase information of a spectrogram.

    """
    def __init__(self, spectrogram, ratio=0.5, max_bins=3, diff_frames=None):
        """
        Creates a new ODF object instance.

        :param spectrogram: the spectrogram on which the detections functions operate
        :param ratio: calculate the difference to the frame which has the given magnitude ratio [default=0.5]
        :param max_bins: number of bins for the maximum filter [default=3]
        :param diff_frames: calculate the difference to the N-th previous frame [default=None]

        If no diff_frames are given, they are calculated automatically based on
        the given ratio.

        """
        self.s = spectrogram
        # determine the number off diff frames
        if diff_frames is None:
            # get the first sample with a higher magnitude than given ratio
            sample = np.argmax(self.s.window > ratio)
            diff_samples = self.s.window.size / 2 - sample
            # convert to frames
            diff_frames = int(round(diff_samples / self.s.hop_size))
            # set the minimum to 1
            if diff_frames < 1:
                diff_frames = 1
        self.diff_frames = diff_frames
        # number of bins used for the maximum filter
        self.max_bins = max_bins

    def diff(self, spec, pos=False, diff_frames=None):
        """
        Calculates the difference of the magnitude spectrogram.

        :param spec: the magnitude spectrogram
        :param pos: only keep positive values [default=False]
        :param diff_frames: calculate the difference to the N-th previous frame [default=None]

        """
        diff = np.zeros_like(spec)
        if diff_frames is None:
            diff_frames = self.diff_frames
        assert diff_frames >= 1, 'number of diff_frames must be >= 1'
        # calculate the diff
        diff[diff_frames:] = spec[diff_frames:] - spec[0:-diff_frames]
        # keep only positive values
        if pos:
            diff = diff * (diff > 0)
        return diff

    def max_diff(self, spec, pos=False, diff_frames=None, max_bins=None):
        """
        Calculates the difference of k-th frequency bin of the magnitude
        spectrogram relative to the maximum over m bins (e.g. m=3: k-1, k, k+1)
        of the N-th previous frame.

        :param spec: the magnitude spectrogram
        :param pos: only keep positive values [default=False]
        :param diff_frames: calculate the difference to the N-th previous frame [default=None]
        :param max_bins: number of bins the maximum is search [default=None]

        Note: This method works only properly if the number of bands for the
              filterbank is chosen carefully. A values of 24 (i.e. quarter-tone
              resolution) usually yields good results.

        """
        # import
        import scipy.ndimage as sim
        # init diff matrix
        diff = np.zeros_like(spec)
        if diff_frames is None:
            diff_frames = self.diff_frames
        assert diff_frames >= 1, 'number of diff_frames must be >= 1'
        if max_bins is None:
            max_bins = self.max_bins
        assert max_bins >= 1, 'search range for the maximum filter must be >= 1'
        # calculate the diff
        diff[diff_frames:] = spec[diff_frames:] - sim.maximum_filter(spec, size=[1, max_bins])[0:-diff_frames]
        # keep only positive values
        if pos:
            diff = diff * (diff > 0)
        return diff

    def corr_diff(self, spec, pos=False, diff_frames=None, corr_bins=None):
        """
        Calculates the difference of the magnitude spectrogram relative to the
        N-th previous frame shifted in frequency to achieve the highest
        correlation between these two frames.

        :param spec: the magnitude spectrogram
        :param pos: only keep positive values [default=False]
        :param diff_frames: calculate the difference to the N-th previous frame [default=None]
        :param corr_bins: maximum number of bins shifted for correlation calculation [default=None]

        """
        # init diff matrix
        diff = np.zeros_like(spec)
        # number of diff frames
        if diff_frames is None:
            diff_frames = self.diff_frames
        assert diff_frames >= 1, 'number of diff_frames must be >= 1'
        # correlation shift in bins
        if corr_bins is None:
            corr_bins = 1
        # calculate the diff
        frames, bins = diff.shape
        corr = np.zeros((frames, corr_bins * 2 + 1))
        for f in range(diff_frames, frames):
            # correlate the frame with the previous one
            # resulting size = bins * 2 - 1
            c = np.correlate(spec[f], spec[f - diff_frames], mode='full')
            # save the middle part
            centre = len(c) / 2
            corr[f] = c[centre - corr_bins: centre + corr_bins + 1]
            # shift the frame for difference calculation according to the
            # highest peak in correlation
            bin_offset = corr_bins - np.argmax(corr[f])
            bin_start = corr_bins + bin_offset
            bin_stop = bins - 2 * corr_bins + bin_start
            diff[f, corr_bins:-corr_bins] = spec[f, corr_bins:-corr_bins] - spec[f - diff_frames, bin_start:bin_stop]
        # keep only positive values
        if pos:
            diff = diff * (diff > 0)
        return diff

    def track_diff(self, spec, pos=False, diff_frames=None, max_bins=None):
        """
        Calculates the difference of the magnitude spectrogram relative to the
        bin of N-th previous frame with the highest path costs (i.e. along the
        magnitude trajectory).

        :param spec: the magnitude spectrogram
        :param pos: only keep positive values [default=False]
        :param diff_frames: calculate the difference to the N-th previous frame [default=None]
        :param max_bins: maximum number of bins searched for trajectory-tracking [default=None]

        Notes: This method is slow, since it uses a triple nested for-loop.
               The number of max_bins is searched in both frequency directions.

        """
        # init diff matrix
        diff = np.zeros_like(spec)
        # number of diff frames
        if diff_frames is None:
            diff_frames = self.diff_frames
        assert diff_frames >= 1, 'number of diff_frames must be >= 1'
        # number of diff bins
        if max_bins is None:
            max_bins = self.max_bins
        assert max_bins >= 1, 'number of max_bins must be >= 1'
        # calculate the diff
        frames, bins = diff.shape
        for f in range(diff_frames, frames):
            # process all frames
            for b in range(max_bins * diff_frames, bins - max_bins * diff_frames):
                # process all bins
                bin_max = b
                for step in range(1, diff_frames + 1):
                    # iteratively go backwards
                    bin_max = np.argmax(spec[f - step, bin_max - max_bins:bin_max + max_bins + 1])
                    bin_max += (b - max_bins)
                diff[f, b] = spec[f, b] - spec[f - diff_frames, bin_max]
        # keep only positive values
        if pos:
            diff = diff * (diff > 0)
        return diff

    # Onset Detection Functions
    def superflux(self):
        """
        SuperFlux with a maximum filter trajectory tracking stage.

        "Maximum Filter Vibrato Suppression for Onset Detection"
        Sebastian Böck, and Gerhard Widmer
        Proceedings of the 16th International Conferenceon Digital Audio Effects
        (DAFx-13), Maynooth, Ireland, September 2013

        """
        return np.sum(self.max_diff(self.s.spec, pos=True), axis=1)

    def sf(self):
        """
        Spectral Flux.

        "Computer Modeling of Sound for Transformation and Synthesis of Musical Signals"
        Paul Masri
        PhD thesis, University of Bristol, 1996

        Note: This method is included for comparison.

        """
        return np.sum(self.diff(self.s.spec, pos=True), axis=1)

    def sfc(self):
        """
        Spectral Flux with correlation trajectory tracking stage.

        Note: This method is included for completeness resons only.
              The superflux() method should be used instead.

        """
        return np.sum(self.corr_diff(self.s.spec, pos=True), axis=1)

    def sft(self):
        """
        Spectral Flux with 'brute-force' trajectory tracking stage.

        Note: This method is included for completeness resons only.
              The superflux() method should be used instead.

        """
        return np.sum(self.track_diff(self.s.spec, pos=True), axis=1)


class Onset(object):
    """
    Onset Class.

    """
    def __init__(self, activations, fps, online=True, sep=''):
        """
        Creates a new Onset object instance with the given activations of the
        ODF (OnsetDetectionFunction). The activations can be read in from a file.

        :param activations: an array containing the activations of the ODF
        :param fps: frame rate of the activations
        :param online: work in online mode (i.e. use only past information) [default=True]

        """
        self.activations = None     # activations of the ODF
        self.fps = fps              # framerate of the activation function
        self.online = online        # online peak-picking
        self.detections = []        # list of detected onsets (in seconds)
        # set / load activations
        if isinstance(activations, np.ndarray):
            # activations are given as an array
            self.activations = activations
        else:
            # read in the activations from a file
            self.load(activations, sep)

    def detect(self, threshold, combine=30, pre_avg=100, pre_max=30, post_avg=30, post_max=70, delay=0):
        """
        Detects the onsets.

        :param threshold: threshold for peak-picking
        :param combine: only report 1 onset for N miliseconds [default=30]
        :param pre_avg: use N miliseconds past information for moving average [default=100]
        :param pre_max: use N miliseconds past information for moving maximum [default=30]
        :param post_avg: use N miliseconds future information for moving average [default=0]
        :param post_max: use N miliseconds future information for moving maximum [default=40]
        :param delay: report the onset N miliseconds delayed [default=0]

        In online mode, post_avg and post_max are set to 0.

        Implements the peak-picking method described in:

        "Evaluating the Online Capabilities of Onset Detection Methods"
        Sebastian Böck, Florian Krebs and Markus Schedl
        Proceedings of the 13th International Society for Music Information Retrieval Conference (ISMIR), 2012

        """
        import scipy.ndimage as sim
        # online mode?
        if self.online:
            post_max = 0
            post_avg = 0
        # convert timing information to frames
        pre_avg = int(round(self.fps * pre_avg / 1000.))
        pre_max = int(round(self.fps * pre_max / 1000.))
        post_max = int(round(self.fps * post_max / 1000.))
        post_avg = int(round(self.fps * post_avg / 1000.))
        # convert to seconds
        combine /= 1000.
        delay /= 1000.
        # init detections
        self.detections = []
        # moving maximum
        max_length = pre_max + post_max + 1
        max_origin = int(np.floor((pre_max - post_max) / 2))
        mov_max = sim.filters.maximum_filter1d(self.activations, max_length, mode='constant', origin=max_origin)
        # moving average
        avg_length = pre_avg + post_avg + 1
        avg_origin = int(np.floor((pre_avg - post_avg) / 2))
        mov_avg = sim.filters.uniform_filter1d(self.activations, avg_length, mode='constant', origin=avg_origin)
        # detections are activation equal to the maximum
        detections = self.activations * (self.activations == mov_max)
        # detections must be greater or equal than the moving average + threshold
        detections = detections * (detections >= mov_avg + threshold)
        # convert detected onsets to a list of timestamps
        last_onset = 0
        for i in np.nonzero(detections)[0]:
            onset = float(i) / float(self.fps) + delay
            # only report an onset if the last N miliseconds none was reported
            if onset > last_onset + combine:
                self.detections.append(onset)
                # save last reported onset
                last_onset = onset

    def write(self, filename):
        """
        Write the detected onsets to the given file.

        :param filename: the target file name

        Only useful if detect() was invoked before.

        """
        with open(filename, 'w') as f:
            for pos in self.detections:
                f.write(str(pos) + '\n')

    def save(self, filename, sep):
        """
        Save the onset activations to the given file.

        :param filename: the target file name
        :param sep: separator between activation values

        Note: using an empty separator ('') results in a binary numpy array.

        """
        print filename, sep
        self.activations.tofile(filename, sep=sep)

    def load(self, filename, sep):
        """
        Load the onset activations from the given file.

        :param filename: the target file name
        :param sep: separator between activation values

        Note: using an empty separator ('') results in a binary numpy array.

        """
        self.activations = np.fromfile(filename, sep=sep)


def parser():
    """
    Parses the command line arguments.

    """
    import argparse
    # define parser
    p = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter, description="""
    If invoked without any parameters, the software detects all onsets in
    the given files in online mode according to the method proposed in:

    "Maximum Filter Vibrato Suppression for Onset Detection"
    by Sebastian Böck and Gerhard Widmer
    in Proceedings of the 16th International Conference on Digital Audio Effects
    (DAFx-13), Maynooth, Ireland, September 2013

    """)
    # general options
    p.add_argument('files', metavar='files', nargs='+', help='files to be processed')
    p.add_argument('-v', dest='verbose', action='store_true', help='be verbose')
    p.add_argument('-s', dest='save', action='store_true', default=False, help='save the activations of the onset detection functions')
    p.add_argument('-l', dest='load', action='store_true', default=False, help='load the activations of the onset detection functions')
    p.add_argument('--sep', action='store', default='', help='separater for saving/loading the onset detection functions [default=numpy binary]')
    # online / offline mode
    p.add_argument('--offline', dest='online', action='store_false', default=True, help='operate in offline mode')
    # wav options
    wav = p.add_argument_group('audio arguments')
    wav.add_argument('--norm', action='store_true', default=None, help='normalize the audio [switches to offline mode]')
    wav.add_argument('--att', action='store', type=float, default=None, help='attenuate the audio by ATT dB')
    # spectrogram options
    spec = p.add_argument_group('spectrogram arguments')
    spec.add_argument('--fps', action='store', default=200, type=int, help='frames per second')
    spec.add_argument('--window', action='store', type=int, default=2048, help='Hanning window length')
    spec.add_argument('--ratio', action='store', type=float, default=0.5, help='window magnitude ratio to calc number of diff frames')
    spec.add_argument('--diff_frames', action='store', type=int, default=None, help='diff frames')
    spec.add_argument('--max_bins', action='store', type=int, default=3, help='bins used for maximum filtering [default=3]')
    # spec-processing
    pre = p.add_argument_group('pre-processing arguments')
    # filter
    pre.add_argument('--filter', action='store_true', default=None, help='filter the magnitude spectrogram with a filterbank')
    pre.add_argument('--fmin', action='store', default=27.5, type=float, help='minimum frequency of filter in Hz [default=27.5]')
    pre.add_argument('--fmax', action='store', default=16000, type=float, help='maximum frequency of filter in Hz [default=16000]')
    pre.add_argument('--bands', action='store', type=int, default=24, help='number of bands per octave [default=24]')
    pre.add_argument('--equal', action='store_true', default=False, help='equalize triangular windows to have equal area')
    # logarithm
    pre.add_argument('--log', action='store_true', default=None, help='logarithmic magnitude')
    pre.add_argument('--mul', action='store', default=1, type=float, help='multiplier (before taking the log) [default=1]')
    pre.add_argument('--add', action='store', default=1, type=float, help='value added (before taking the log) [default=1]')
    # onset detection
    onset = p.add_argument_group('onset detection arguments')
    onset.add_argument('-o', dest='odf', default=None, help='use this onset detection function [superflux,sf,sfc,sft]')
    onset.add_argument('-t', dest='threshold', action='store', type=float, default=1.25, help='detection threshold')
    onset.add_argument('--combine', action='store', type=float, default=30, help='combine onsets within N miliseconds [default=30]')
    onset.add_argument('--pre_avg', action='store', type=float, default=100, help='build average over N previous miliseconds [default=100]')
    onset.add_argument('--pre_max', action='store', type=float, default=30, help='search maximum over N previous miliseconds [default=30]')
    onset.add_argument('--post_avg', action='store', type=float, default=70, help='build average over N following miliseconds [default=70]')
    onset.add_argument('--post_max', action='store', type=float, default=30, help='search maximum over N following miliseconds [default=30]')
    onset.add_argument('--delay', action='store', type=float, default=0, help='report the onsets N miliseconds delayed [default=0]')
    # version
    p.add_argument('--version', action='version', version='%(prog)spec 1.0 (2013-04-14)')
    # parse arguments
    args = p.parse_args()

    # list of offered ODFs
    methods = ['superflux', 'sf', 'sfc', 'sft']
    # use default values if no ODF is given
    if args.odf is None:
        args.odf = 'superflux'
        if args.log is None:
            args.log = True
        if args.filter is None:
            args.filter = True
    # remove mistyped methods
    assert args.odf in methods, 'at least one onset detection function must be given'

    # print arguments
    if args.verbose:
        print args

    # return args
    return args


def main():
    """
    Main program.

    """
    import os.path
    import glob
    import fnmatch

    # parse arguments
    args = parser()

    # determine the files to process
    files = []
    for f in args.files:
        # check what we have (file/path)
        if os.path.isdir(f):
            # use all files in the given path
            files = glob.glob(f + '/*.wav')
        else:
            # file was given, append to list
            files.append(f)

    # only process .wav files
    files = fnmatch.filter(files, '*.wav')
    files.sort()

    # init filterbank
    filt = None

    # process the files
    for f in files:
        if args.verbose:
            print f

        # use the name of the file without the extension
        filename = os.path.splitext(f)[0]

        # init Onset object
        o = None
        # do the processing stuff unless the activations are loaded from file
        if args.load:
            # load the activations from file
            o = Onset("%s.%s" % (filename, args.odf), args.fps, args.online, args.sep)
        else:
            # open the wav file
            w = Wav(f)
            # normalize audio
            if args.norm:
                w.normalize()
                args.online = False  # switch to offline mode
            # downmix to mono
            if w.channels > 1:
                w.downmix()
            # attenuate signal
            if args.att:
                w.attenuate(args.att)
            # spectrogram
            s = Spectrogram(w, args.window, args.fps, args.online)
            # filter
            if args.filter:
                # (re-)create filterbank if the samplerate of the audio changes
                if filt is None or filt.fs != w.samplerate:
                    filt = Filter(args.window / 2, w.samplerate, args.bands, args.fmin, args.fmax, args.equal)
                # filter the spectrogram
                s.filter(filt.filterbank)
            # log
            if args.log:
                s.log(args.mul, args.add)
            # use the spectrogram to create an SpectralODF object
            sodf = SpectralODF(s, args.ratio, args.max_bins, args.diff_frames)
            # perform detection function on the object
            act = getattr(sodf, args.odf)()
            # create an Onset object with the activations
            o = Onset(act, args.fps, args.online)
            if args.save:
                # save the raw ODF activations
                o.save("%s.%s" % (filename, args.odf), args.sep)

        # detect the onsets
        o.detect(args.threshold, args.combine, args.pre_avg, args.pre_max, args.post_avg, args.post_max, args.delay)
        # write the onsets to a file
        o.write("%s.onsets.txt" % (filename))
        # also output them to stdout if vebose
        if args.verbose:
            print 'detections:', o.detections

        # continue with next file

if __name__ == '__main__':
    main()
