# DISCLAIMER
# The majority of code for this parser was written for another
# open-source implementation of the dance-dance-convolution
# paper.

import logging
import re
import os
import librosa
import numpy as np
import copy
import math

parlog = logging

VALID_PULSES = set([4, 8, 12, 16, 24, 32, 48, 64, 96, 192])


def int_parser(x): return int(x.strip()) if x.strip() else None


def bool_parser(x): return True if x.strip() == 'YES' else False


def str_parser(x): return x.strip() if x.strip() else None


def float_parser(x): return float(x.strip()) if x.strip() else None


def kv_parser(k_parser, v_parser):
    def parser(x):
        if not x:
            return (None, None)
        k, v = x.split('=', 1)
        return k_parser(k), v_parser(v)
    return parser


def list_parser(x_parser):
    def parser(l):
        l_strip = l.strip()
        if len(l_strip) == 0:
            return []
        else:
            return [x_parser(x) for x in l_strip.split(',')]
    return parser


def bpms_parser(x):
    bpms = list_parser(kv_parser(float_parser, float_parser))(x)

    if len(bpms) == 0:
        raise ValueError('No BPMs found in list')
    if bpms[0][0] != 0.0:
        raise ValueError('First beat in BPM list is {}'.format(bpms[0][0]))

    # make sure changes are nonnegative, take last for equivalent
    beat_last = -1.0
    bpms_cleaned = []
    for beat, bpm in bpms:
        if beat == None or bpm == None:
            raise ValueError('Empty BPM found')
        if bpm <= 0.0:
            raise ValueError('Non positive BPM found {}'.format(bpm))
        if beat == beat_last:
            bpms_cleaned[-1] = (beat, bpm)
            continue
        bpms_cleaned.append((beat, bpm))
        if beat <= beat_last:
            raise ValueError('Descending list of beats in BPM list')
        beat_last = beat
    if len(bpms) != len(bpms_cleaned):
        parlog.warning(
            'One or more (beat, BPM) pairs begin on the same beat, using last listed')

    return bpms_cleaned


def stops_parser(x):
    stops = list_parser(kv_parser(float_parser, float_parser))(x)

    beat_last = -1.0
    for beat, stop_len in stops:
        if beat == None or stop_len == None:
            raise ValueError('Bad stop formatting')
        if beat < 0.0:
            raise ValueError('Bad beat in stop')
        if stop_len == 0.0:
            continue
        if beat <= beat_last:
            raise ValueError('Nonascending list of beats in stops')
        beat_last = beat
    return stops


def notes_parser(x):
    pattern = r'([^:]*):' * 5 + r'([^;:]*)'
    notes_split = re.findall(pattern, x)
    if len(notes_split) != 1:
        raise ValueError('Bad formatting of notes section')
    notes_split = notes_split[0]
    if (len(notes_split) != 6):
        raise ValueError('Bad formatting within notes section')

    # parse/clean measures
    measures = [measure.splitlines() for measure in notes_split[5].split(',')]
    measures_clean = []
    for measure in measures:
        measure_clean = list(filter(lambda pulse: not pulse.strip(
        ).startswith('//') and len(pulse.strip()) > 0, measure))
        measures_clean.append(measure_clean)
    if len(measures_clean) > 0 and len(measures_clean[-1]) == 0:
        measures_clean = measures_clean[:-1]

    # check measure lengths
    for measure in measures_clean:
        if len(measure) == 0:
            raise ValueError('Found measure with 0 notes')
        if not len(measure) in VALID_PULSES:
            parlog.warning(
                'Nonstandard subdivision {} detected, allowing'.format(len(measure)))

    chart_type = str_parser(notes_split[0])
    if chart_type not in ['dance-single', 'dance-double', 'dance-couple', 'lights-cabinet']:
        raise ValueError(
            'Nonstandard chart type {} detected'.format(chart_type))

    return (str_parser(notes_split[0]),
            str_parser(notes_split[1]),
            str_parser(notes_split[2]),
            int_parser(notes_split[3]),
            list_parser(float_parser)(notes_split[4]),
            measures_clean
            )


def unsupported_parser(attr_name):
    def parser(x):
        raise ValueError(
            'Unsupported attribute: {} with value {}'.format(attr_name, x))
        return None
    return parser


ATTR_NAME_TO_PARSER = {
    'title': str_parser,
    'subtitle': str_parser,
    'artist': str_parser,
    'titletranslit': str_parser,
    'subtitletranslit': str_parser,
    'artisttranslit': str_parser,
    'genre': str_parser,
    'credit': str_parser,
    'banner': str_parser,
    'background': str_parser,
    'lyricspath': str_parser,
    'cdtitle': str_parser,
    'music': str_parser,
    'offset': float_parser,
    'bpms': bpms_parser,
    'stops': stops_parser,
    'samplestart': float_parser,
    'samplelength': float_parser,
    'displaybpm': str_parser,
    'selectable': bool_parser,
    'bgchanges': str_parser,
    'bgchanges2': str_parser,
    'fgchanges': str_parser,
    'keysounds': str_parser,
    'musiclength': float_parser,
    'musicbytes': int_parser,
    'attacks': str_parser,
    'timesignatures': list_parser(kv_parser(float_parser, kv_parser(int_parser, int_parser))),
    'warps': unsupported_parser('warps'),
    'notes': notes_parser
}
ATTR_MULTI = ['notes']


def parse_sm_txt(sm_txt):
    attrs = {attr_name: [] for attr_name in ATTR_MULTI}

    for attr_name, attr_val in re.findall(r'#([^:]*):([^;]*);', sm_txt):
        attr_name = attr_name.lower()

        if attr_name not in ATTR_NAME_TO_PARSER:
            parlog.warning(
                'Found unexpected attribute {}:{}, ignoring'.format(attr_name, attr_val))
            continue

        attr_val_parsed = ATTR_NAME_TO_PARSER[attr_name](attr_val)
        if attr_name in attrs:
            if attr_name not in ATTR_MULTI:
                if attr_val_parsed == attrs[attr_name]:
                    continue
                else:
                    raise ValueError(
                        'Attribute {} defined multiple times'.format(attr_name))
            attrs[attr_name].append(attr_val_parsed)
        else:
            attrs[attr_name] = attr_val_parsed

    return {k: v for k, v in attrs.items() if v is not None and v != []}


class StepManiaChart(object):
    def __init__(self, dirpath):
        self.dirpath = dirpath
        self.audio = None
        self.features = None
        self.steps = None
        for filename in os.listdir(dirpath):
            if filename.endswith('.sm'):
                with open(os.path.join(dirpath, filename), 'r') as file:
                    for k, v in parse_sm_txt(file.read()).items():
                        setattr(self, k, v)
                break

    def to_ddc_io(self):
        STFT_WINDOW_SIZES_MS = [23, 46, 93]
        STRIDE_MS = 10

        if self.audio is None:
            try:
                self.audio, sr = librosa.load(os.path.join(
                    self.dirpath, self.music), mono=True)
            except:
                for filename in os.listdir(self.dirpath):
                    if filename.endswith('.mp3') or filename.endswith('.ogg'):
                        self.audio, sr = librosa.load(os.path.join(
                            self.dirpath, filename), mono=True)

        if self.features is None:
            def ms_to_n_samples(ms): return int(ms * sr / 1000)

            feature_list = []

            for window_size_ms in STFT_WINDOW_SIZES_MS:
                feature_list.append(librosa.logamplitude(librosa.feature.melspectrogram(
                    self.audio, sr, n_fft=ms_to_n_samples(window_size_ms),
                    hop_length=ms_to_n_samples(STRIDE_MS), n_mels=80
                )))

            self.features = np.array(feature_list)

        def song_time_to_features_idx(song_time):
            return song_time / (len(self.audio.shape) / sr) \
                * self.features.shape[1]

        if self.steps is None:
            self.steps = {
                # step placements - 1x(self.features_time_length) float tensor
                # 1.0 for any step (or release), 0.0 for nothing
                'placement': np.zeros(self.features.shape[1]),
                # selection list of dicts. One for each 1.0 in step placements.
                # Each containing: {
                #   'bag-of-arrows': 2d (arrow)x(one-hot step type),
                #   'delta-time': 1x2 float tensor
                #       [secs since last step, secs till next step],
                #   'delta-beat': 1x2 float tensor
                #       [quater-note beats since last step, till next step],
                #   'beat-phase': 1x4 one-hot tensor which 16th note subdivision
                #       most closely aligns to the current step
                # }
                #
                'selection': []
            }

            stops = []
            if hasattr(self, 'stops'):
                stops = self.stops

            warps = []
            if hasattr(self, 'warps'):
                warps = self.warps

            print(self.bpms)

            for step_chart in self.notes:
                chart_type, _artist_name, difficulty_class, \
                    difficulty, _groove_stats, measures = \
                    step_chart
                bpms = copy.copy(self.bpms)
                current_bpm = None
                next_beat_song_time = 0.0
                last_beat_i = 0.0

                for measure_i, measure in enumerate(measures):
                    for step_i, step_str in enumerate(measure):
                        beat_i = (measure_i + step_i / len(measure)) * 4

                        bpm_change = calc_bpm_change(beat_i, last_beat_i, bpms)
                        if bpm_change is not None:
                            current_bpm, bpms = bpm_change

                        next_beat_song_time += 60 / \
                            current_bpm * 4 / len(measure)

                        if should_be_warp_skipped(beat_i):
                            break

                        last_beat_i = beat_i

        return self.features


def calc_bpm_change(beat_i, last_beat_i, bpms):
    newest_bpm_beat_i, newest_bpm, next_bpms = 0, None, []
    for (bpm_beat_i, bpm) in bpms:
        if beat_i - 1 <= bpm_beat_i and bpm_beat_i <= beat_i:
            if bpm_beat_i >= newest_bpm_beat_i:
                newest_bpm = bpm
                newest_bpm_beat_i = bpm_beat_i
        else:
            next_bpms.append(bpm)

    if newest_bpm is None:
        return None
    else:
        return newest_bpm, bpms
