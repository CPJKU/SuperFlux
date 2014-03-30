#!/usr/bin/env python
# encoding: utf-8
"""
Copyright (c) 2012 - 2014 Sebastian Böck <sebastian.boeck@jku.at>
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
from scipy.ndimage.filters import (maximum_filter, maximum_filter1d,
                                   uniform_filter1d)


class Filter(object):
    """
    Filter Class.

    """
    def __init__(self, ffts, fs, bands=24, fmin=27.5, fmax=16000, equal=False):
        """
        Creates a new Filter object instance.

        :param ffts:  number of FFT coefficients
        :param fs:    sample rate of the audio file
        :param bands: number of filter bands
        :param fmin:  the minimum frequency [Hz]
        :param fmax:  the maximum frequency [Hz]
        :param equal: normalize the area of each band to 1

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
        assert bands >= 3, 'cannot create filterbank with less than 3 ' \
                           'frequencies'
        # init the filter matrix with size: ffts x filter bands
        self.filterbank = np.zeros([ffts, bands], dtype=np.float)
        # process all bands
        for band in range(bands):
            # edge & center frequencies
            start, mid, stop = frequencies[band:band + 3]
            # create a triangular filter
            triang_filter = self.triang(start, mid, stop, equal)
            self.filterbank[start:stop, band] = triang_filter

    @staticmethod
    def frequencies(bands, fmin, fmax, a=440):
        """
        Returns a list of frequencies aligned on a logarithmic scale.

        :param bands: number of filter bands per octave
        :param fmin:  the minimum frequency [Hz]
        :param fmax:  the maximum frequency [Hz]
        :param a:     frequency of A0 [Hz]
        :returns:     a list of frequencies

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

        :param start: start bin (with value 0, included in the filter)
        :param mid:   center bin (of height 1, unless norm is True)
        :param stop:  end bin (with value 0, not included in the filter)
        :param equal: normalize the area of the filter to 1
        :returns:     a triangular shaped filter

        """
        # height of the filter
        height = 1.
        # normalize the height
        if equal:
            height = 2. / (stop - start)
        # init the filter
        triang_filter = np.empty(stop - start)
        # rising edge
        rising = np.linspace(0, height, (mid - start), endpoint=False)
        triang_filter[:mid - start] = rising
        # falling edge
        falling = np.linspace(height, 0, (stop - mid), endpoint=False)
        triang_filter[mid - start:] = falling
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
        att = np.power(np.sqrt(10.), attenuation / 10.)
        self.audio = np.asarray(self.audio / att, dtype=self.audio.dtype)

    def downmix(self):
        """
        Down-mix the audio signal to mono.

        """
        if self.channels > 1:
            self.audio = np.mean(self.audio, axis=-1, dtype=self.audio.dtype)

    def normalize(self):
        """
        Normalize the audio signal.

        """
        self.audio = self.audio.astype(np.float) / np.max(self.audio)


class Spectrogram(object):
    """
    Spectrogram Class.

    """
    def __init__(self, wav, frame_size=2048, fps=200, filterbank=None,
                 log=False, mul=1, add=1, online=True, block_size=2048):
        """
        Creates a new Spectrogram object instance and performs a STFT on the
        given audio.

        :param wav:        a Wav object
        :param frame_size: the size for the window [samples]
        :param fps:        frames per second
        :param online:     work in online mode (i.e. use only past information)

        """
        # init some variables
        self.wav = wav
        self.fps = fps
        if add <= 0:
            raise ValueError("a positive value must be added before taking "
                             "the logarithm")
        if mul <= 0:
            raise ValueError("a positive value must be multiplied before "
                             "taking the logarithm")
        # derive some variables
        # use floats so that seeking works properly
        self.hop_size = float(self.wav.samplerate) / float(self.fps)
        self.frames = int(self.wav.samples / self.hop_size)
        self.ffts = int(frame_size / 2)
        # initial number equal to ffts, can change if filters are used
        self.bins = int(frame_size / 2)
        # init spec matrix
        if filterbank is None:
            # init with number of FFT frequency bins
            self.spec = np.empty([self.frames, self.ffts])
        else:
            # init with number of filter bands
            self.spec = np.empty([self.frames, np.shape(filterbank)[1]])
            # set number of bins
            self.bins = np.shape(filterbank)[1]
            # set the block size
            if not block_size or block_size > self.frames:
                block_size = self.frames
            # init block counter
            block = 0
            # init a matrix of that size
            spec = np.zeros([block_size, self.ffts])
        # create windowing function for DFT
        self.window = np.hanning(frame_size)
        try:
            # the audio signal is not scaled, scale the window accordingly
            max_value = np.iinfo(self.wav.audio.dtype).max
            self._fft_window = self.window / max_value
        except ValueError:
            self._fft_window = self.window
        # step through all frames
        for frame in range(self.frames):
            # seek to the right position in the audio signal
            if online:
                # step back one frame_size after moving forward 1 hop_size
                # so that the current position is at the end of the window
                seek = int((frame + 1) * self.hop_size - frame_size)
            else:
                # step back half of the frame_size so that the frame represents
                # the centre of the window
                seek = int(frame * self.hop_size - frame_size / 2)
            # read in the right portion of the audio
            if seek >= self.wav.samples:
                # end of file reached
                break
            elif seek + frame_size >= self.wav.samples:
                # end behind the actual audio, append zeros accordingly
                zeros = np.zeros(seek + frame_size - self.wav.samples)
                signal = self.wav.audio[seek:]
                signal = np.append(signal, zeros)
            elif seek < 0:
                # start before the actual audio, pad with zeros accordingly
                zeros = np.zeros(-seek)
                signal = self.wav.audio[0:seek + frame_size]
                signal = np.append(zeros, signal)
            else:
                # normal read operation
                signal = self.wav.audio[seek:seek + frame_size]
            # multiply the signal with the window function
            signal = signal * self._fft_window
            # perform DFT
            stft = fft.fft(signal)[:self.ffts]
            # is block-wise processing needed?
            if filterbank is None:
                # no filtering needed, thus no block wise processing needed
                self.spec[frame] = np.abs(stft)
            else:
                # filter in blocks
                spec[frame % block_size] = np.abs(stft)
                # end of a block or end of the signal reached
                end_of_block = (frame + 1) / block_size > block
                end_of_signal = (frame + 1) == self.frames
                if end_of_block or end_of_signal:
                    start = block * block_size
                    stop = min(start + block_size, self.frames)
                    filtered_spec = np.dot(spec[:stop - start], filterbank)
                    self.spec[start:stop] = filtered_spec
                    # increase the block counter
                    block += 1
            # next frame
        # take the logarithm
        if log:
            np.log10(mul * self.spec + add, out=self.spec)


class SpectralODF(object):
    """
    The SpectralODF class implements most of the common onset detection
    function based on the magnitude or phase information of a spectrogram.

    """
    def __init__(self, spectrogram, ratio=0.5, max_bins=3, diff_frames=None):
        """
        Creates a new ODF object instance.

        :param spectrogram: a Spectrogram object on which the detection
                            functions operate
        :param ratio:       calculate the difference to the frame which has the
                            given magnitude ratio
        :param max_bins:    number of bins for the maximum filter
        :param diff_frames: calculate the difference to the N-th previous frame

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

    def diff(self, spec, pos=False, diff_frames=None, max_bins=None):
        """
        Calculates the difference of the magnitude spectrogram.

        :param spec:        the magnitude spectrogram
        :param pos:         only keep positive values
        :param diff_frames: calculate the difference to the N-th previous frame
        :param max_bins:    number of bins over which the maximum is searched

        Note: If 'max_bins' is greater than 0, a maximum filter of this size
              is applied in the frequency deirection. The difference of the
              k-th frequency bin of the magnitude spectrogram is then
              calculated relative to the maximum over m bins of the N-th
              previous frame (e.g. m=3: k-1, k, k+1).

              This method works only properly if the number of bands for the
              filterbank is chosen carefully. A values of 24 (i.e. quarter-tone
              resolution) usually yields good results.

        """
        # init diff matrix
        diff = np.zeros_like(spec)
        if diff_frames is None:
            diff_frames = self.diff_frames
        assert diff_frames >= 1, 'number of diff_frames must be >= 1'
        # apply the maximum filter if needed
        if max_bins > 0:
            max_spec = maximum_filter(spec, size=[1, max_bins])
        else:
            max_spec = spec
        # calculate the diff
        diff[diff_frames:] = spec[diff_frames:] - max_spec[0:-diff_frames]
        # keep only positive values
        if pos:
            diff *= (diff > 0)
        return diff

    # Onset Detection Functions
    def superflux(self):
        """
        SuperFlux with a maximum filter trajectory tracking stage.

        "Maximum Filter Vibrato Suppression for Onset Detection"
        Sebastian Böck, and Gerhard Widmer
        Proceedings of the 16th International Conferenceon Digital Audio
        Effects (DAFx-13), Maynooth, Ireland, September 2013

        """
        return np.sum(self.diff(self.s.spec, pos=True, max_bins=self.max_bins),
                      axis=1)


class Onset(object):
    """
    Onset Class.

    """
    def __init__(self, activations, fps, online=True, sep=''):
        """
        Creates a new Onset object instance with the given activations of the
        ODF (OnsetDetectionFunction). The activations can be read from a file.

        :param activations: an array containing the activations of the ODF
        :param fps:         frame rate of the activations
        :param online:      work in online mode (i.e. use only past
                            information)

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

    def detect(self, threshold, combine=30, pre_avg=100, pre_max=30,
               post_avg=30, post_max=70, delay=0):
        """
        Detects the onsets.

        :param threshold: threshold for peak-picking
        :param combine:   only report 1 onset for N miliseconds
        :param pre_avg:   use N miliseconds past information for moving average
        :param pre_max:   use N miliseconds past information for moving maximum
        :param post_avg:  use N miliseconds future information for mov. average
        :param post_max:  use N miliseconds future information for mov. maximum
        :param delay:     report the onset N miliseconds delayed

        In online mode, post_avg and post_max are set to 0.

        Implements the peak-picking method described in:

        "Evaluating the Online Capabilities of Onset Detection Methods"
        Sebastian Böck, Florian Krebs and Markus Schedl
        Proceedings of the 13th International Society for Music Information
        Retrieval Conference (ISMIR), 2012

        """
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
        mov_max = maximum_filter1d(self.activations, max_length,
                                   mode='constant', origin=max_origin)
        # moving average
        avg_length = pre_avg + post_avg + 1
        avg_origin = int(np.floor((pre_avg - post_avg) / 2))
        mov_avg = uniform_filter1d(self.activations, avg_length,
                                   mode='constant', origin=avg_origin)
        # detections are activation equal to the maximum
        detections = self.activations * (self.activations == mov_max)
        # detections must be greater or equal than the mov. average + threshold
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
    p = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter, description="""
    If invoked without any parameters, the software detects all onsets in
    the given files in online mode according to the method proposed in:

    "Maximum Filter Vibrato Suppression for Onset Detection"
    by Sebastian Böck and Gerhard Widmer
    Proceedings of the 16th International Conference on Digital Audio Effects
    (DAFx-13), Maynooth, Ireland, September 2013

    """)
    # general options
    p.add_argument('files', metavar='files', nargs='+',
                   help='files to be processed')
    p.add_argument('-v', dest='verbose', action='store_true',
                   help='be verbose')
    p.add_argument('-s', dest='save', action='store_true', default=False,
                   help='save the activations of the onset detection function')
    p.add_argument('-l', dest='load', action='store_true', default=False,
                   help='load the activations of the onset detection function')
    p.add_argument('--sep', action='store', default='',
                   help='separater for saving/loading the onset detection '
                        'function [default=numpy binary]')
    # online / offline mode
    p.add_argument('--offline', dest='online', action='store_false',
                   default=True, help='operate in offline mode')
    # wav options
    wav = p.add_argument_group('audio arguments')
    wav.add_argument('--norm', action='store_true', default=None,
                     help='normalize the audio (switches to offline mode)')
    wav.add_argument('--att', action='store', type=float, default=None,
                     help='attenuate the audio by ATT dB')
    # spectrogram options
    spec = p.add_argument_group('spectrogram arguments')
    spec.add_argument('--fps', action='store', default=200, type=int,
                      help='frames per second [default=%(default)s]')
    spec.add_argument('--frame_size', action='store', type=int, default=2048,
                      help='frame size [samples, default=%(default)s]')
    spec.add_argument('--ratio', action='store', type=float, default=0.5,
                      help='window magnitude ratio to calc number of diff '
                           'frames [default=%(default)s]')
    spec.add_argument('--diff_frames', action='store', type=int, default=None,
                      help='diff frames')
    spec.add_argument('--max_bins', action='store', type=int, default=3,
                      help='bins used for maximum filtering '
                           '[default=%(default)s]')
    # spec-processing
    pre = p.add_argument_group('pre-processing arguments')
    # filter
    pre.add_argument('--no_filter', dest='filter', action='store_false',
                     default=True, help='do not filter the magnitude '
                                        'spectrogram with a filterbank')
    pre.add_argument('--fmin', action='store', default=27.5, type=float,
                     help='minimum frequency of filter '
                          '[Hz, default=%(default)s]')
    pre.add_argument('--fmax', action='store', default=16000, type=float,
                     help='maximum frequency of filter '
                          '[Hz, default=%(default)s]')
    pre.add_argument('--bands', action='store', type=int, default=24,
                     help='number of bands per octave [default=%(default)s]')
    pre.add_argument('--equal', action='store_true', default=False,
                     help='equalize triangular windows to have equal area')
    pre.add_argument('--block_size', action='store', default=2048, type=int,
                     help='perform filtering in blocks of N frames '
                          '[default=%(default)s]')
    # logarithm
    pre.add_argument('--no_log', dest='log', action='store_false',
                     default=True, help='use linear magnitude scale')
    pre.add_argument('--mul', action='store', default=1, type=float,
                     help='multiplier (before taking the log) '
                          '[default=%(default)s]')
    pre.add_argument('--add', action='store', default=1, type=float,
                     help='value added (before taking the log) '
                          '[default=%(default)s]')
    # onset detection
    onset = p.add_argument_group('onset detection arguments')
    onset.add_argument('-t', dest='threshold', action='store', type=float,
                       default=1.25, help='detection threshold '
                                          '[default=%(default)s]')
    onset.add_argument('--combine', action='store', type=float, default=30,
                       help='combine onsets within N miliseconds '
                            '[default=%(default)s]')
    onset.add_argument('--pre_avg', action='store', type=float, default=100,
                       help='build average over N previous miliseconds '
                            '[default=%(default)s]')
    onset.add_argument('--pre_max', action='store', type=float, default=30,
                       help='search maximum over N previous miliseconds '
                            '[default=%(default)s]')
    onset.add_argument('--post_avg', action='store', type=float, default=70,
                       help='build average over N following miliseconds '
                            '[default=%(default)s]')
    onset.add_argument('--post_max', action='store', type=float, default=30,
                       help='search maximum over N following miliseconds '
                            '[default=%(default)s]')
    onset.add_argument('--delay', action='store', type=float, default=0,
                       help='report the onsets N miliseconds delayed '
                            '[default=%(default)s]')
    # version
    p.add_argument('--version', action='version',
                   version='%(prog)spec 1.01 (2014-03-30)')
    # parse arguments
    args = p.parse_args()
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
    filterbank = None
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
            o = Onset("%s.act" % filename, args.fps, args.online, args.sep)
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
            # create filterbank if needed
            if args.filter:
                # (re-)create filterbank if the samplerate of the audio changes
                if filt is None or filt.fs != w.samplerate:
                    filt = Filter(args.frame_size / 2, w.samplerate,
                                  args.bands, args.fmin, args.fmax, args.equal)
                    filterbank = filt.filterbank
            # spectrogram
            s = Spectrogram(w, args.frame_size, args.fps, filterbank, args.log,
                            args.mul, args.add, args.online, args.block_size)
            # use the spectrogram to create an SpectralODF object
            sodf = SpectralODF(s, args.ratio, args.max_bins, args.diff_frames)
            # perform detection function on the object
            act = sodf.superflux()
            # create an Onset object with the activations
            o = Onset(act, args.fps, args.online)
            if args.save:
                # save the raw ODF activations
                o.save("%s.act" % filename, args.sep)
        # detect the onsets
        o.detect(args.threshold, args.combine, args.pre_avg, args.pre_max,
                 args.post_avg, args.post_max, args.delay)
        # write the onsets to a file
        o.write("%s.onsets.txt" % (filename))
        # also output them to stdout if vebose
        if args.verbose:
            print 'detections:', o.detections
        # continue with next file

if __name__ == '__main__':
    main()
