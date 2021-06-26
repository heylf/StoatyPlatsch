
import numpy as np
import scipy.signal


def deconvolute_peak_with_STFT(peak, stft_args, eval_params_peak_lengths=None):
    """ Deconvolutes the given peak with the given parameters.

    Parameters
    ----------
    peak : OriginalPeak, RefinedPeak, UnrefinedPeak
        The peak that should be deconvoluted.
    stft_args : dict
        The parameters that should be used for the calculation of the STFT.
    eval_params_peak_lengths : dict (default: None)
        If a dictionary is passed, additional parameters for an evaluation are
        calculated.

    Returns
    -------
    stft_f : np.ndarray
        Array with the sample frequencies.
    stft_t : np.ndarray
        Array with the segment times.
    Zxx : np.ndarray
        The STFT of the peak.
    """

    stft_f, stft_t, Zxx = scipy.signal.stft(**stft_args)

    istft_args = {}
    istft_args['fs'] = stft_args['fs']
    istft_args['window'] = stft_args['window']
    istft_args['nperseg'] = stft_args['nperseg']
    istft_args['noverlap'] = stft_args['noverlap']
    istft_args['nfft'] = stft_args['nfft']
    istft_args['input_onesided'] = stft_args['return_onesided']
    istft_args['boundary'] = stft_args['boundary']
    istft_args['time_axis'] = -1
    istft_args['freq_axis'] = -2

    peak.deconv_peaks_rel = []

    Zxx_abs = np.abs(Zxx)

    index_argmax = np.flip(np.argsort(Zxx_abs, kind='stable', axis=0), axis=0)
    frequencies_to_consider = np.delete(
        index_argmax, np.where(index_argmax == 0)[0], axis=0
        )    # Remove the 'frequency' corresponding to a constant value.
    if len(frequencies_to_consider) == 0:
        # If the frequency list is empty (should only happen to peaks of
        # length 1. In this case, keep original "peak".
        peak.deconv_peaks_rel.append(
            [0, np.round(len(peak.coverage)/2).astype(int), len(peak.coverage)]
            )
        return

    main_freq_indices = frequencies_to_consider[0]
    main_freq_values = np.zeros(len(main_freq_indices))
    for i in range(Zxx.shape[1]):
        main_freq_values[i] = Zxx_abs[main_freq_indices[i], i]

    segments_to_consider = np.flip(np.argsort(main_freq_values, kind='stable'))

    peaks_defined = 0
    i_segment_to_consider = 0
    while ((peaks_defined < peak.num_peaks_estimated)
           and ((i_segment_to_consider < len(segments_to_consider)))):
        i_segment = segments_to_consider[i_segment_to_consider]
        i_segment_to_consider += 1

        main_freq_index = main_freq_indices[i_segment]
        z = np.zeros(Zxx.shape, dtype=Zxx.dtype)
        z[main_freq_index, i_segment] = Zxx[main_freq_index, i_segment]
        istft_args['Zxx'] = z

        _istft_t, istft_x = scipy.signal.istft(**istft_args)

        padding_left = (len(stft_args['x']) - len(peak.coverage)) // 2

        center = istft_x.argmax() - padding_left
        if center >= len(peak.coverage):
            # Can happen, as input can be extended to the left when applying
            # stft.
            continue

        wavelength = 1/stft_f[main_freq_index]
        distance_center_min = np.round(wavelength/2).astype(int)

        # Check if peak is too close to an already defined peak
        too_close = False
        for _l, c, _r in peak.deconv_peaks_rel:
            if (np.abs(center - c) < (distance_center_min // 2)):
                too_close = True
                break

        if too_close:
            continue

        left_boundary = max(center - distance_center_min, 0)
        right_boundary = min(center + distance_center_min + 1,
                             len(peak.coverage))

        if eval_params_peak_lengths:
            eppl = eval_params_peak_lengths  # Abbreviation
            unmodified_length = distance_center_min * 2 + 1
            if not eppl['unmodified'].get(unmodified_length):
                eppl['unmodified'][unmodified_length] = 1
            else:
                eppl['unmodified'][unmodified_length] += 1
            actual_length = right_boundary - left_boundary
            if unmodified_length != actual_length:
                if not eppl['to_be_clipped'].get(unmodified_length):
                    eppl['to_be_clipped'][unmodified_length] = 1
                else:
                    eppl['to_be_clipped'][unmodified_length] += 1
                if not eppl['after_clipped'].get(actual_length):
                    eppl['after_clipped'][actual_length] = 1
                else:
                    eppl['after_clipped'][actual_length] += 1
            if not eppl['final_all'].get(actual_length):
                eppl['final_all'][actual_length] = 1
            else:
                eppl['final_all'][actual_length] += 1

        peak.deconv_peaks_rel.append([left_boundary, center, right_boundary])
        peaks_defined += 1

    if not peak.deconv_peaks_rel:
        if eval_params_peak_lengths:
            raise ValueError("Attention, special case! Not conisdered yet!")

        # If peak could not be deconvoluted, keep original peak.
        peak.deconv_peaks_rel.append(
            [0, np.round(len(peak.coverage)/2).astype(int), len(peak.coverage)]
            )

    return stft_f, stft_t, Zxx


def deconvolute_peaks_with_STFT(peaks,
                                stft_window='boxcar',
                                stft_detrend='constant',
                                stft_noverlap=10,
                                find_peaks_distance=10,
                                find_peaks_height=5,
                                find_peaks_prominence=3,
                                verbose=False,
                                eval_params_peak_lengths=None):
    """ Deconvolutes the given peaks with a STFT approach.

    Parameters
    ----------
    peaks : OrderedDict
        The dictionary containing the peaks that should be deconvoluted.
    stft_window : str, tuple, array_like (default: 'boxcar')
        The STFT window parameter.
    stft_detrend : str, function, False (default: 'constant')
        The STFT detrend parameter.
    stft_noverlap : int (default: 10)
        The STFT noverlap parameter.
    find_peaks_distance : int (default: 10)
        The distance parameter of function 'find_peaks' to estimate the number
        of subpeaks that should be defined.
    find_peaks_height : int (default: 5)
        The height parameter of function 'find_peaks' to estimate the number
        of subpeaks that should be defined.
    find_peaks_prominence : int (default: 3)
        The prominence parameter of function 'find_peaks' to estimate the
        number of subpeaks that should be defined.
    verbose : bool (default: False)
        Print information to console when set to True.
    eval_params_peak_lengths : dict (default: None)
        Extract some additional evaluation parameters used for the thesis.
    """
    if verbose:
        print("[NOTE] Deconvolute peaks with STFT.")

    # Sets the parameters for function 'stft'.
    stft_args = {}
    stft_args['fs'] = 1    # Should always be 1, as the sampling
    # frequency is always 1 value per nucleotide.
    stft_args['window'] = stft_window
    stft_args['nfft'] = None
    stft_args['detrend'] = stft_detrend
    stft_args['return_onesided'] = True    # Should always be True, as
    # input data is always real, therefore two-sided spectrum is symmetric
    # and one-sided result is sufficient.
    stft_args['boundary'] = None
    stft_args['padded'] = True
    stft_args['axis'] = -1

    for peak in peaks.values():

        find_peaks_result = scipy.signal.find_peaks(
            peak.coverage, distance=find_peaks_distance,
            height=[max(peak.coverage)
                    if max(peak.coverage) < find_peaks_height
                    else find_peaks_height],
            prominence=find_peaks_prominence)
        peak.num_peaks_estimated = max(1, len(find_peaks_result[0]))

        stft_args['x'] = peak.coverage
        stft_args['nperseg'] = np.ceil(len(peak.coverage)
                                       / peak.num_peaks_estimated
                                       ).astype(int)
        stft_args['noverlap'] = min(stft_noverlap,
                                    stft_args['nperseg']-1)

        deconvolute_peak_with_STFT(peak, stft_args, eval_params_peak_lengths)
