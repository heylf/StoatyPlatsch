
import numpy as np
from scipy import signal


def analyze_with_FFT(peaks, num_padding, verbose=False):
    """ TODO

    Parameters
    ----------
    peaks : dict
        The dictionary containing the peaks that should be analyzed and
        deconvoluted.
    num_padding : list
        Number of zeros that should be padded to the left and right side of the
        peak profiles, respectively. num_padding[0] is the number of zeros
        added on the left side, num_padding[1] the number added to the right."
    verbose : bool (default: False)
        Print information to console when set to True.
    """

    # Dummy class for bundling FFT relevant variables together.
    class FFT(object):
        pass

    if verbose:
        print("[NOTE] Analyze peaks with FFT.")

        # TODO: Add changes made to class description.
        for p_id in sorted(peaks.keys()):
            peak = peaks[p_id]

            peak.fft = FFT()

            peak.fft.num_padding = num_padding
            peak.fft.f = np.pad(peak.coverage, num_padding)

            n = len(peak.fft.f)
            peak.fft.fhat = np.fft.rfft(peak.fft.f, n)
            peak.fft.fhat_re = np.real(peak.fft.fhat)
            peak.fft.fhat_im = np.imag(peak.fft.fhat)
            peak.fft.fhat_abs = np.abs(peak.fft.fhat)
            peak.fft.fhat_power_spectrum = np.abs(peak.fft.fhat)**2
            peak.fft.fhat_phase_spectrum = np.angle(peak.fft.fhat)

            peak.fft.fhat_norm = np.fft.rfft(peak.fft.f, n, norm="ortho")
            peak.fft.fhat_norm_re = np.real(peak.fft.fhat_norm)
            peak.fft.fhat_norm_im = np.imag(peak.fft.fhat_norm)
            peak.fft.fhat_norm_abs = np.abs(peak.fft.fhat_norm)
            peak.fft.fhat_norm_power_spectrum = np.abs(peak.fft.fhat_norm)**2
            peak.fft.fhat_norm_phase_spectrum = np.angle(peak.fft.fhat_norm)

            peak.fft.index_argmax = np.argsort(np.abs(peak.fft.fhat_abs),
                                               kind='stable')

            peak.fft.fhat_filtered_index = {}
            peak.fft.fhat_filtered = {}
            peak.fft.f_filtered = {}
            for filter_num in np.arange(len(peak.fft.index_argmax)):
                peak.fft.fhat_filtered_index[filter_num] = \
                    peak.fft.fhat_abs >= peak.fft.fhat_abs[
                        peak.fft.index_argmax[-(filter_num+1)]]
                peak.fft.fhat_filtered[filter_num] = \
                    (peak.fft.fhat_filtered_index[filter_num]
                     * peak.fft.fhat)
                peak.fft.f_filtered[filter_num] = \
                    np.fft.irfft(peak.fft.fhat_filtered[filter_num], n)


def deconvolute_with_FFT(peaks, num_padding, verbose=False):
    """ TODO

    Parameters
    ----------
    peaks : dict
        The dictionary containing the peaks that should be analyzed and
        deconvoluted.
    num_padding : list
        Number of zeros that should be padded to the left and right side of the
        peak profiles, respectively. num_padding[0] is the number of zeros
        added on the left side, num_padding[1] the number added to the right."
    verbose : bool (default: False)
        Print information to console when set to True.
    """

    if verbose:
        print("[NOTE] Deconvolute peaks with FFT.")

    for p_id in sorted(peaks.keys()):
        peak = peaks[p_id]

        # peak =
        peak.fft.local_maxs = {}
        peak.fft.local_mins = {}

        # min_height = 10
        # distance = 5
        # For testing, calculate for all filter indices.
        for filter_num in np.arange(len(peak.fft.index_argmax)):
            peak.fft.local_maxs[filter_num] = \
                signal.find_peaks(peak.fft.f_filtered[filter_num])[0]
            peak.fft.local_mins[filter_num] = \
                signal.find_peaks(-peak.fft.f_filtered[filter_num])[0]
                              # distance = distance,
                              # height=[minimal_height_new, max(y)])

        # For now, user filter 3
        max_values = peak.fft.local_maxs[3]
        min_values = peak.fft.local_mins[3]

        boundary_indices = np.argwhere(peak.fft.f_filtered[3] > 0)

        min_values = np.concatenate(([boundary_indices.min()], min_values,
                                     [boundary_indices.max()]))

        # print("p_id: ", p_id)
        # print("max_values: ", max_values)
        # print("min_values: ", min_values)
        # print("-------------")

        peak.fft.new_peaks = []
        for m in max_values:
            if peak.fft.f_filtered[3][m] <= 0:
                continue
            left_boundary = min_values[np.argwhere(min_values < m).max()]
            right_boundary = min_values[np.argwhere(min_values > m).min()]

            offset = peak.fft.num_padding[0] + peak.chrom_start
            peak.fft.new_peaks.append(
                [left_boundary + offset, m + offset, right_boundary + offset]
                )
        # print(peak.fft.new_peaks)
        # print("-------------")
        # print("-------------")
