

_likert_samples = [
    {'strongly disagree': 0, 'disagree': 1, 'undecided': 2, 'agree': 3, 'strongly agree': 4},
    {'disagree strongly': 0, 'disagree moderately': 1, 'disagree slightly': 2, 'agree slightly': 3, 'agree moderately': 4, 'agree strongly': 5},
    {'disagree': 0, 'undecided': 1, 'agree': 2},
    {'disagree': 0, 'agree': 1},
    {'disagree very strongly': 0, 'disagree strongly': 1, 'disagree': 2, 'agree': 3, 'agree strongly': 4, 'agree very strongly': 5},
    {'completely disagree': 0, 'mostly disagree': 1, 'slightly disagree': 2, 'slightly agree': 3, 'mostly agree': 4, 'completely agree': 5},
    {'agree strongly': 0, 'agree': 1, 'slightly agree': 2, 'slightly disagree': 3, 'disagree': 4, 'disagree strongly': 5},
    {'none': 0, 'low': 1, 'moderate': 2, 'high': 3},
    {'low': 0, 'moderate': 1, 'high': 3},
    {'low': 0, 'medium': 1, 'high': 3},
    {'poor': 0, 'somewhat': 1, 'excellent': 2},
    {'not at all': 0, 'partially': 1, 'entirely': 2},
    {'never': 0, 'very rarely': 1, 'rarely': 2, 'occasionally': 3, 'frequently': 4, 'very frequently': 5},
    {'never': 0, 'very rarely': 1, 'rarely': 2, 'occasionally': 3, 'very frequently': 4, 'always': 5},
    {'never': 0, 'seldom': 1, 'about half the time': 2, 'usually': 3, 'always': 4},
    {'never': 0, 'rarely': 1, 'sometimes': 2, 'very often': 3, 'always': 4},
    {'unfair': 0, 'fair': 1},
    {'disagree': 0, 'agree': 1},
    {'false': 0, 'true': 1},
    {'no': 0, 'yes': 1}
]


def _get_possible_scales(count):
    possible_scales = []
    for each_scale in _likert_samples:
        if len(each_scale) == count:
            possible_scales.append(each_scale)
    return possible_scales
