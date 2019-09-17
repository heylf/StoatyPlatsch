import numpy
import matplotlib.pyplot as plt
import scipy.stats

from dist_check import dist_check_main

from scipy import signal

def update_spec_from_peaks(spec, minimal_height, distance, **kwargs):
    x = spec['x']
    y = spec['y']

    # Find maxima and minima threshold=[None, 100],
    found_local_maximas = signal.find_peaks(y, distance = distance, height=[minimal_height, max(y)])

    # inverse data for local minima
    inv_y = y * -1
    found_local_minima = signal.find_peaks(inv_y, distance = distance)[0]

    # I had problems with the edges (start and end of the peak profile).
    # If I have a long stretch of zeros I get possibly not the start and end of the peak as a minimum.
    diff_y = numpy.diff(y)

    start_index = 0
    pos = 0
    while( diff_y[pos] == 0 and pos != (len(diff_y)-1) ):
        start_index += 1
        pos += 1

    end_index = len(diff_y)-1
    pos = len(diff_y)-1
    while( diff_y[pos] == 0 and pos != 0 ):
        end_index -= 1
        pos -= 1

    found_local_minima = numpy.concatenate( ([start_index], found_local_minima, [end_index]) )

    peak_indicies = numpy.array(found_local_maximas[0])
    peak_indicies.sort()

    left_local_minimum_dict = {}
    right_local_minimum_dict = {}

    num_local_maxima = len(peak_indicies)

    for i in range(0, num_local_maxima):
        left_local_minimun = 0
        right_local_minimum = len(x)
        summit = peak_indicies[i]

        terminate = 1
        m = 0
        while( terminate == 1 ):
            if ( summit > found_local_minima[m] ):
                left_local_minimun = found_local_minima[m]
            else:
                right_local_minimum = found_local_minima[m]
                terminate = 0
            m += 1
            if ( m  == len(found_local_minima) ):
                terminate = 0

        left_local_minimum_dict[peak_indicies[i]] = left_local_minimun
        right_local_minimum_dict[peak_indicies[i]] = right_local_minimum

    # Check if two maximas share the same width.
    # This happens if one local maxima is actually a false positive.
    true_positives = []
    if ( len(peak_indicies) > 1 ):
        for a in range(0, len(peak_indicies)):
            peak_to_keep = peak_indicies[a]
            for b in range(1, len(peak_indicies)):
                if ( left_local_minimum_dict[peak_indicies[a]] == left_local_minimum_dict[peak_indicies[b]] and
                     right_local_minimum_dict[peak_indicies[a]] == right_local_minimum_dict[peak_indicies[b]] and
                     y[peak_indicies[a]] < y[peak_indicies[b]] ):
                    peak_to_keep = peak_indicies[b]
            if ( peak_to_keep not in true_positives ):
                true_positives.append(peak_to_keep)
    else:
        true_positives = peak_indicies

    peak_indicies = numpy.array(true_positives)
    peak_indicies.sort()

    print(peak_indicies)

    fitted_profiles_dict = dict()

    std = [-1] * len(peak_indicies)
    for i in range(0, len(peak_indicies)):

        s = left_local_minimum_dict[peak_indicies[i]]
        e = right_local_minimum_dict[peak_indicies[i]]+1

        subpeak_y = y[s:e]
        subpeak_x = [x for x in range(len(subpeak_y))]

        best_subpeak_dist_type = dist_check_main(subpeak_y,1)

        print(best_subpeak_dist_type)

        subpeak_params = eval("scipy.stats." + best_subpeak_dist_type + ".fit(subpeak_y)")
        subpeak_f = eval("scipy.stats." + best_subpeak_dist_type + ".freeze" + str(subpeak_params))
        subpeak_fitted_quantiles = numpy.linspace(subpeak_f.ppf(0.00001), subpeak_f.ppf(0.99999), len(subpeak_x))
        fitted_y = subpeak_f.pdf(subpeak_fitted_quantiles)

        # There are no negative read counts
        for j in range(0, len(fitted_y)):
            if ( fitted_y[j] < 0.0 ):
                fitted_y[j] = 0.0

        # max min normalize fitted_profile so fitted profile fits better the real data
        max_of_fitted = numpy.max(fitted_y)
        min_of_fitted = numpy.min(fitted_y)
        for j in range(0, len(fitted_y)):
            a = ( fitted_y[j] - min_of_fitted )
            b = ( max_of_fitted - min_of_fitted)
            fitted_y[j] =  ( a / b ) * numpy.max(subpeak_y)

        # shift profile based on the correct summit
        summit_index_y = numpy.argmax(subpeak_y)
        summit_index_fitted_y = numpy.argmax(fitted_y)
        shift = summit_index_y - summit_index_fitted_y
        s += shift
        e += shift

        full_fitted_y = numpy.zeros(len(y))
        full_fitted_y[s:e] = fitted_y

        fitted_profiles_dict[peak_indicies[i]] = full_fitted_y

        # scale = variance
        std[i] = numpy.sqrt(subpeak_params[1])

        print(std[i])

    return [peak_indicies, found_local_minima, std, fitted_profiles_dict]
