
import numpy as np
from scipy import signal


class FFT(object):
    ''' Dummy class for bundling FFT relevant variables together. '''
    pass


def analyze_with_FFT(peaks, num_padding, verbose=False):
    """ Apply FFT and perform calculations and checks for analysis.

    Parameters
    ----------
    peaks : dict
        The dictionary containing the peaks that should be analyzed.
    num_padding : list
        Number of zeros that should be padded to the left and right side of the
        peak profiles, respectively. num_padding[0] is the number of zeros
        added on the left side, num_padding[1] the number added to the right."
    verbose : bool (default: False)
        Print information to console when set to True.
    """

    if verbose:
        print("[NOTE] Analyze peaks with FFT.")

    for p_id in sorted(peaks.keys()):
        peak = peaks[p_id]

        peak.fft = FFT()

        peak.fft.num_padding = num_padding
        peak.fft.f = np.pad(peak.coverage, num_padding)

        peak.fft.n = len(peak.fft.f)
        peak.fft.freq = np.fft.rfftfreq(peak.fft.n)

        peak.fft.fhat = np.fft.rfft(peak.fft.f, peak.fft.n)
        peak.fft.fhat_re = np.real(peak.fft.fhat)
        peak.fft.fhat_im = np.imag(peak.fft.fhat)
        peak.fft.fhat_abs = np.abs(peak.fft.fhat)
        peak.fft.fhat_power_spectrum = np.abs(peak.fft.fhat)**2
        peak.fft.fhat_phase_spectrum = np.angle(peak.fft.fhat)

        peak.fft.fhat_norm = np.fft.rfft(peak.fft.f, peak.fft.n, norm="ortho")
        peak.fft.fhat_norm_re = np.real(peak.fft.fhat_norm)
        peak.fft.fhat_norm_im = np.imag(peak.fft.fhat_norm)
        peak.fft.fhat_norm_abs = np.abs(peak.fft.fhat_norm)
        peak.fft.fhat_norm_power_spectrum = np.abs(peak.fft.fhat_norm)**2
        peak.fft.fhat_norm_phase_spectrum = np.angle(peak.fft.fhat_norm)

        # The argsort results should be the same for the different norms.
        for i, (fhat1, fhat2) in enumerate(
            [(peak.fft.fhat, peak.fft.fhat_norm),
             (peak.fft.fhat_re, peak.fft.fhat_norm_re),
             (peak.fft.fhat_im, peak.fft.fhat_norm_im),
             (peak.fft.fhat_abs, peak.fft.fhat_norm_abs),
             (peak.fft.fhat_power_spectrum, peak.fft.fhat_norm_power_spectrum),
             (peak.fft.fhat_phase_spectrum, peak.fft.fhat_norm_phase_spectrum)]
             ):
            assert np.all(np.argsort(fhat1, kind='stable')
                          == np.argsort(fhat2, kind='stable')), \
                   "Error with enumeration index {}".format(i)

        # Different results for:
        #     fhat(= fhat_re) ; fhat_im ; fhat_abs (= fhat_power_spectrum) ;
        #     fhat_phase_spectrum
        peak.fft.index_argmax = np.flip(
            np.argsort(peak.fft.fhat_abs, kind='stable')
            )

        peak.fft.filter_mask = {}
        peak.fft.filtered_fhat = {}
        peak.fft.filtered_f = {}
        peak.fft.local_maxs = {}
        peak.fft.local_mins = {}
        # min_height = 10
        # distance = 5
        for filter_num in np.arange(len(peak.fft.index_argmax)):
            peak.fft.filter_mask[filter_num] = \
                np.zeros(len(peak.fft.index_argmax), dtype=bool)
            peak.fft.filter_mask[filter_num][
                    peak.fft.index_argmax[:(filter_num+1)]
                ] = True
            peak.fft.filtered_fhat[filter_num] = \
                (peak.fft.filter_mask[filter_num] * peak.fft.fhat)
            peak.fft.filtered_f[filter_num] = \
                np.fft.irfft(peak.fft.filtered_fhat[filter_num], peak.fft.n)

            peak.fft.local_maxs[filter_num] = \
                signal.find_peaks(peak.fft.filtered_f[filter_num])[0]
            peak.fft.local_mins[filter_num] = \
                signal.find_peaks(-peak.fft.filtered_f[filter_num])[0]
            #                  # distance = distance,
            #                  # height=[minimal_height_new, max(y)])


def deconvolute_with_FFT(peaks, num_padding, approach='map_FFT_signal',
                         distance=10, height=10, verbose=False):
    """ Deconvolutes the given peaks by using a FFT approach.

    Parameters
    ----------
    peaks : dict
        The dictionary containing the peaks that should be deconvoluted.
    num_padding : list
        Number of zeros that should be padded to the left and right side of the
        peak profiles, respectively. num_padding[0] is the number of zeros
        added on the left side, num_padding[1] the number added to the right.
    approach : {'smooth', 'map_profile', 'map_FFT_signal'},
               (default: 'map_FFT_signal')
        'smooth': Uses FFT to smooth the peak profile and then use the smoothed
                  profile to identify the local minima and" maxima. These
                  maxima define the new peaks with the minima as boundaries.
        'map_profile': Calculates the peaks (maxima) of the original profile
                       and the underlying FFT frequencies. Maps the peaks of
                       the profile to the frequency that has the maximum
                       closest to the peak maximum and is also part of the
                       frequencies contributing the most to the signal. Uses
                       the mapped maximum of the frequency and the distance
                       to its minima as peak center and width.
       'map_FFT_signal': Calculates the peaks (maxima) of the original profile
                         and the underlying FFT frequencies. Maps each of the
                         frequencies that contributes most to the signal to one
                         maximum of the profile. Uses the mapped maximum of the
                         frequency and the distance to its minima as peak
                         center and width.
    distance : int (default: 10)
        Used as parameter for function 'find_peaks' when calculating the maxima
        of the original profile. Only used for the 'map_profile' and the
        'map_FFT_signal' approaches.
    height : int (default: 10)
        Used as parameter for function 'find_peaks' when calculating the maxima
        of the original profile. Only used for the 'map_profile' and the
        'map_FFT_signal' approaches.
    verbose : bool (default: False)
        Print information to console when set to True.

    Raises
    ------
    ValueError
        When an unsupported value for parameter 'approach' is given.

    """

    if verbose:
        print("[NOTE] Deconvolute peaks with FFT.")

    for p_id in sorted(peaks.keys()):
        peak = peaks[p_id]

        peak.fft = FFT()
        peak.fft.approach = approach
        peak.fft.num_padding = num_padding
        peak.fft.f = np.pad(peak.coverage, num_padding)

        peak.fft.n = len(peak.fft.f)
        peak.fft.freq = np.fft.rfftfreq(peak.fft.n)

        peak.fft.fhat = np.fft.rfft(peak.fft.f, peak.fft.n)
        peak.fft.fhat_abs = np.abs(peak.fft.fhat)

        # Get indices, sorted after frequencies with the largest contribution.
        peak.fft.index_argmax = np.flip(
            np.argsort(peak.fft.fhat_abs, kind='stable')
            )

        peak.fft.new_peaks = []

        if approach == 'smooth':
            # First approach, using the 4 largest frequencies for filtering
            # the profile and calculating the maxima and minima on this
            # profile.
            filter_num = 3
            peak.fft.filter_mask = np.zeros(len(peak.fft.index_argmax),
                                            dtype=bool)
            peak.fft.filter_mask[
                    peak.fft.index_argmax[:(filter_num+1)]
                ] = True
            peak.fft.filtered_fhat = (peak.fft.filter_mask * peak.fft.fhat)
            peak.fft.filtered_f = \
                np.fft.irfft(peak.fft.filtered_fhat, peak.fft.n)

            # Note: The complete signal is zero-based, therefore differs when
            #       padding is is used both to the relative and absolute
            #       nucleotide positions of the peak.
            peak.fft.local_maxs = signal.find_peaks(peak.fft.filtered_f)[0]
            peak.fft.local_mins = signal.find_peaks(-peak.fft.filtered_f)[0]

            boundary_indices = np.argwhere(peak.fft.filtered_f > 0)

            peak.fft.local_mins = \
                np.concatenate(([boundary_indices.min()], peak.fft.local_mins,
                                [boundary_indices.max()]))

            for m in peak.fft.local_maxs:
                if ((peak.fft.filtered_f[m] <= 0)
                        or (m < num_padding[0])
                        or (m >= peak.fft.n - num_padding[1])):
                    continue

                left_boundary = peak.fft.local_mins[
                        np.argwhere(peak.fft.local_mins < m).max()
                    ]
                right_boundary = peak.fft.local_mins[
                        np.argwhere(peak.fft.local_mins > m).min()
                    ]

                offset = peak.chrom_start - peak.fft.num_padding[0]
                peak.fft.new_peaks.append(
                    [left_boundary + offset, m + offset,
                     right_boundary + offset
                     + 1  # Chrom end is non-inclusive, therefore + 1
                     ]
                    )

        elif approach in ['map_profile', 'map_FFT_signal']:
            # Estimate number of possible subpeaks by calculating maxima of
            # the peak profile. The result defines the number of considered
            # frequencies.

            if max(peak.fft.f) < height:
                height = max(peak.fft.f)

            peak.fft.local_maxs = \
                signal.find_peaks(peak.fft.f, distance=distance,
                                  height=[height])[0]

            # First "frequency" 0 should be ignored, it is only constant.
            frequencies_to_consider = \
                peak.fft.index_argmax[1:len(peak.fft.local_maxs)+2]

            # Calculate the single frequency values.
            peak.fft.frequencies = {}
            peak.fft.frequency_max_pos = {}
            peak.fft.frequency_max_distance = {}
            peak.fft.frequency_min_pos = {}
            for i in frequencies_to_consider:
                indices = np.zeros(len(peak.fft.fhat))
                indices[i] = 1
                peak.fft.frequencies[i] = \
                    np.fft.irfft(indices * peak.fft.fhat, peak.fft.n)
                peak.fft.frequency_max_pos[i] = \
                    signal.find_peaks(peak.fft.frequencies[i])[0]
                peak.fft.frequency_min_pos[i] = \
                    signal.find_peaks(-peak.fft.frequencies[i])[0]
                peak.fft.frequency_min_pos[i] = \
                    np.concatenate(([0], peak.fft.frequency_min_pos[i],
                                    [peak.fft.n - 1]))

            if approach == 'map_profile':
                for m in peak.fft.local_maxs:
                    best_freq = [-1, -1, np.Inf]   # Index of best frequency,
                    #     position of maximum in this frequency, distance.

                    # For each maximum, calculate the distance to peaks of the
                    # frequencies and choose the frequency which closest
                    # maximum.
                    for i in frequencies_to_consider:
                        l_maxs = \
                            np.argwhere(peak.fft.frequency_max_pos[i] <= m)
                        if len(l_maxs) > 0:
                            l_max = peak.fft.frequency_max_pos[i][l_maxs.max()]
                            if (m - l_max) < best_freq[2]:
                                best_freq[0] = i
                                best_freq[1] = l_max
                                best_freq[2] = m - l_max
                        r_maxs = \
                            np.argwhere(peak.fft.frequency_max_pos[i] >= m)
                        if len(r_maxs) > 0:
                            r_max = peak.fft.frequency_max_pos[i][r_maxs.min()]
                            if (r_max - m) < best_freq[2]:
                                best_freq[0] = i
                                best_freq[1] = r_max
                                best_freq[2] = r_max - m
                        if best_freq[2] == 0:
                            break

                    # Find the minima of the given frequency for calculating
                    # the peak width.
                    left_boundary = peak.fft.frequency_min_pos[best_freq[0]][
                        np.argwhere(peak.fft.frequency_min_pos[best_freq[0]]
                                    < m).max()
                        ]
                    right_boundary = peak.fft.frequency_min_pos[best_freq[0]][
                        np.argwhere(peak.fft.frequency_min_pos[best_freq[0]]
                                    > m).min()
                        ]

                    offset = peak.chrom_start - peak.fft.num_padding[0]
                    peak.fft.new_peaks.append(
                        [left_boundary + offset, best_freq[1] + offset,
                         right_boundary + offset
                         + 1  # Chrom end is non-inclusive, therefore + 1
                         ]
                        )
            else:
                maxima_to_assign = peak.fft.local_maxs.tolist()
                for f in frequencies_to_consider[:-1]:
                    best_map = [-1, -1, np.Inf]   # Index of best max in
                    #     profile, index of best max of frequency, distance

                    for i_m, m in enumerate(maxima_to_assign):
                        distances = abs(peak.fft.frequency_max_pos[f] - m)
                        dist_freq_index = np.argmin(distances)
                        if distances[dist_freq_index] < best_map[2]:
                            best_map[0] = i_m
                            best_map[1] = dist_freq_index
                            best_map[2] = distances[dist_freq_index]

                    maxima_to_assign.pop(best_map[0])
                    m = peak.fft.frequency_max_pos[f][dist_freq_index]

                    # Find the minima of the given frequency for calculating
                    # the peak width.
                    left_boundary = peak.fft.frequency_min_pos[f][
                        np.argwhere(peak.fft.frequency_min_pos[f]
                                    < m).max()
                        ]
                    right_boundary = peak.fft.frequency_min_pos[f][
                        np.argwhere(peak.fft.frequency_min_pos[f]
                                    > m).min()
                        ]

                    offset = peak.chrom_start - peak.fft.num_padding[0]
                    peak.fft.new_peaks.append(
                        [left_boundary + offset, m + offset,
                         right_boundary + offset
                         + 1  # Chrom end is non-inclusive, therefore + 1
                         ]
                        )

        else:
            raise ValueError("Parameter 'approach' must have one of the"
                             " following values:"
                             " 'smooth', 'map_profile', 'map_FFT_signal'."
                             " Given value: '{}'.".format(approach))

        if not peak.fft.new_peaks:
            # If peak could not be deconvoluted, use original peak values.
            peak.fft.new_peaks.append(
                [peak.chrom_start,
                 peak.chrom_start + peak.coverage.argmax(),
                 peak.chrom_end])
