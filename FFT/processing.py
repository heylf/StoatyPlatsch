
import numpy as np
from scipy import signal

from FFT.data import Mapping


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


def add_subpeak(peak, left_boundary, center, right_boundary):
    """ Adds the given subpeak to the list of new peaks.

    Parameters
    ----------
    peak : Peak
        The original peak with the peak data.
    left_boundary : int
        The left boundary of the new subpeak, coordinates of the padded input
        peak, zero-based.
    center : int
        The center of the new subpeak, coordinates of the padded input peak,
        zero-based.
    right_boundary : int
        The right boundary of the new subpeak, coordinates of the padded input
        peak, zero-based.
    """
    offset = peak.chrom_start - peak.fft.num_padding[0]

    left_boundary = left_boundary + offset
    center = center + offset
    right_boundary = right_boundary + offset

    if peak.fft.clip_boundary == 'peak':
        if left_boundary < peak.chrom_start:
            left_boundary = peak.chrom_start
        if right_boundary >= peak.chrom_end:
            right_boundary = peak.chrom_end - 1
    else:
        # Can be also outside padding range, due to frequency shift.
        if left_boundary < peak.chrom_start - peak.fft.num_padding[0]:
            left_boundary = peak.chrom_start - peak.fft.num_padding[0]
        if right_boundary >= peak.chrom_end + peak.fft.num_padding[1]:
            right_boundary = peak.chrom_end + peak.fft.num_padding[1] - 1

    peak.fft.new_peaks.append(
        [left_boundary, center,
         right_boundary + 1  # Chrom end is non-inclusive, therefore + 1
         ]
        )


def deconvolute_with_FFT(peaks, num_padding, approach='map_FFT_signal',
                         clip_boundary='peak', distance=10, height=10,
                         prominence=2,
                         disable_frequency_shift=False,
                         main_freq_filter_value=3, verbose=False):
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
                       to its minima as peak center and width. May be shifted
                       towards the mapped maximum of the peak profile,
                       depending on parameter 'disable_frequency_shift'.
       'map_FFT_signal': Calculates the peaks (maxima) of the original profile
                         and the underlying FFT frequencies. Maps each of the
                         frequencies that contributes most to the signal to one
                         maximum of the profile. Uses the mapped maximum of the
                         frequency and the distance to its minima as peak
                         center and width. May be shifted towards the mapped
                         maximum of the peak profile, depending on parameter
                         'disable_frequency_shift'.
    clip_boundary : {'peak', 'padding'}, (default: 'peak')
        'peak': Uses the original, unpadded peak boundaries as outermost
                boundaries.
        'padding': Uses the padded peak boundaries as outermost boundaries.
    distance : int (default: 10)
        Used as parameter for function 'find_peaks' when calculating the maxima
        of the original profile. Only used for the 'map_profile' and the
        'map_FFT_signal' approaches.
    height : int (default: 10)
        Used as parameter for function 'find_peaks' when calculating the maxima
        of the original profile. Only used for the 'map_profile' and the
        'map_FFT_signal' approaches.
    prominence : int (default: 2)
        Used as parameter for function 'find_peaks' when calculating the maxima
        of the original profile. Only used for the 'map_profile' and the
        'map_FFT_signal' approaches.
    disable_frequency_shift : bool (default: False)
        Disables shifting the underlying FFT frequencies to the mapped maxima
        of the profiles. Only used for the 'map_profile' and the
        'map_FFT_signal' approaches.
    main_freq_filter_value : int (default: 3)
        Defines the number of found maxima in the peak profile that are used
        for filtering the main frequency. Is the number of maxima equal to the
        defined value, the main frequency will be ignored for the
        deconvolution. A value <= 0 disables filtering the main frequency. Only
        used for the 'map_profile' and the 'map_FFT_signal' approaches.
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
        peak.fft.clip_boundary = clip_boundary
        peak.fft.disable_frequency_shift = disable_frequency_shift
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

                add_subpeak(peak, left_boundary, m, right_boundary)

        elif approach in ['map_profile', 'map_FFT_signal']:
            # Estimate number of possible subpeaks by calculating maxima of
            # the peak profile. The result defines the number of considered
            # frequencies.

            peak.fft.local_maxs = \
                signal.find_peaks(peak.fft.f, distance=distance,
                                  height=[max(peak.fft.f)
                                          if max(peak.fft.f) < height
                                          else height
                                          ],
                                  prominence=prominence
                                  )[0]

            # First "frequency" 0 should be ignored, it is only constant.
            frequencies_to_consider = \
                np.delete(peak.fft.index_argmax,
                          np.where(peak.fft.index_argmax == 0)[0])
            # Second "main frequency" should be ignored depending on value
            # of main_freq_filter_value and the number of found maximums on
            # the profile.
            if ((main_freq_filter_value > 0)
                    and (main_freq_filter_value <= len(peak.fft.local_maxs))):
                frequencies_to_consider = \
                    np.delete(frequencies_to_consider,
                              np.where(frequencies_to_consider == 1)[0])
            # Pick the number of considered frequencies depending on the found
            # maxima.
            frequencies_to_consider = \
                frequencies_to_consider[:len(peak.fft.local_maxs)+1]

            # Calculate the single frequency values.
            peak.fft.frequencies = {}
            peak.fft.frequency_max_pos = {}
            peak.fft.frequency_min_pos = {}

            for idx_freq in frequencies_to_consider:
                indices = np.zeros(len(peak.fft.fhat))
                indices[idx_freq] = 1
                peak.fft.frequencies[idx_freq] = \
                    np.fft.irfft(indices * peak.fft.fhat, peak.fft.n)
                peak.fft.frequency_max_pos[idx_freq] = \
                    signal.find_peaks(peak.fft.frequencies[idx_freq])[0]
                peak.fft.frequency_min_pos[idx_freq] = \
                    signal.find_peaks(-peak.fft.frequencies[idx_freq])[0]
                peak.fft.frequency_min_pos[idx_freq] = \
                    np.concatenate(([0], peak.fft.frequency_min_pos[idx_freq],
                                    [peak.fft.n - 1]))

            peak.fft.mappings = []
            if approach == 'map_profile':
                for idx_max_profile, m in enumerate(peak.fft.local_maxs):
                    mapping = Mapping(idx_max_profile=idx_max_profile,
                                      max_pos_profile=m)
                    peak.fft.mappings.append(mapping)

                    # For each maximum, calculate the distance to peaks of the
                    # frequencies and choose the frequency and its
                    # corresponding maximum that is closest to the peaks
                    # maximum.
                    for idx_freq in frequencies_to_consider:
                        l_maxs = \
                            np.argwhere(peak.fft.frequency_max_pos[idx_freq]
                                        <= m)
                        if len(l_maxs) > 0:
                            l_max = peak.fft.frequency_max_pos[idx_freq][
                                l_maxs.max()]
                            if (m - l_max) < mapping.distance:
                                mapping.idx_freq = idx_freq
                                mapping.max_pos_freq = l_max
                                mapping.distance = m - l_max
                        r_maxs = \
                            np.argwhere(peak.fft.frequency_max_pos[idx_freq]
                                        >= m)
                        if len(r_maxs) > 0:
                            r_max = peak.fft.frequency_max_pos[idx_freq][
                                r_maxs.min()]
                            if (r_max - m) < mapping.distance:
                                mapping.idx_freq = idx_freq
                                mapping.max_pos_freq = r_max
                                mapping.distance = r_max - m
                        if mapping.distance == 0:
                            break

            else:
                maxima_to_assign = peak.fft.local_maxs.tolist()

                # Assign each of the considered frequencies to one maximum of
                # the peak profile.
                for idx_freq in frequencies_to_consider[:-1]:
                    mapping = Mapping(idx_freq=idx_freq)
                    peak.fft.mappings.append(mapping)

                    # For each maximum of the profile that is not mapped yet,
                    # calculate the distances to the maxima of the current
                    # frequency and choose the maximum of the profile and the
                    # maximum of the frequency that have the smallest distance.
                    for idx_max_profile, max_pos_profile in enumerate(
                            maxima_to_assign):
                        distances = abs(peak.fft.frequency_max_pos[idx_freq]
                                        - max_pos_profile)
                        dist_freq_index = np.argmin(distances)
                        if distances[dist_freq_index] < mapping.distance:
                            mapping.idx_max_profile = idx_max_profile
                            mapping.max_pos_profile = max_pos_profile
                            mapping.max_pos_freq = peak.fft.frequency_max_pos[
                                idx_freq][dist_freq_index]
                            mapping.distance = distances[dist_freq_index]

                    maxima_to_assign.pop(mapping.idx_max_profile)

            for mapping in peak.fft.mappings:
                # Find the minima of the given frequency for calculating
                # the peak width.
                # Note: This could be probably optimized by using the
                #       frequency information directly to calculate the minima
                #       instead of applying calculations with argwhere.
                left_boundary = \
                    peak.fft.frequency_min_pos[mapping.idx_freq][
                        np.argwhere(peak.fft.frequency_min_pos[
                                        mapping.idx_freq]
                                    < mapping.max_pos_freq).max()
                        ]
                right_boundary = \
                    peak.fft.frequency_min_pos[mapping.idx_freq][
                        np.argwhere(peak.fft.frequency_min_pos[
                                        mapping.idx_freq]
                                    > mapping.max_pos_freq).min()
                    ]

                # Shift the new subpeak towards the maximum of the profile, if
                # not disabled.
                if disable_frequency_shift:
                    shift = 0
                    mapping.shift = None
                    mapping.shifted_freq = None
                else:
                    shift = mapping.max_pos_profile - mapping.max_pos_freq
                    mapping.shift = shift
                    mapping.shifted_freq = \
                        np.roll(peak.fft.frequencies[mapping.idx_freq], shift)

                add_subpeak(peak, left_boundary + shift,
                            mapping.max_pos_freq + shift,
                            right_boundary + shift)

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
