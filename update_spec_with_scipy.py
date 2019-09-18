import numpy
import matplotlib.pyplot as plt
import scipy.stats

from dist_check import dist_check_main

from scipy import signal

def update_spec_from_peaks(spec, minimal_height, distance, peak_width, processes, **kwargs):
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

    fitted_profiles_dict = dict()
    std = [-1] * len(peak_indicies)
    symetric = [0] * len(peak_indicies)
    for i in range(0, len(peak_indicies)):

        s = left_local_minimum_dict[peak_indicies[i]]
        e = right_local_minimum_dict[peak_indicies[i]]+1

        subpeak_y = y[s:e]
        subpeak_x = [x for x in range(len(subpeak_y))]

        best_subpeak_dist_type = dist_check_main(subpeak_y, 1, processes)

        print(best_subpeak_dist_type)

        subpeak_params = eval("scipy.stats." + best_subpeak_dist_type + ".fit(subpeak_y)")
        subpeak_f = eval("scipy.stats." + best_subpeak_dist_type + ".freeze" + str(subpeak_params))
        subpeak_fitted_quantiles = numpy.linspace(subpeak_f.ppf(0.001), subpeak_f.ppf(0.999), len(subpeak_x))

        fitted_y = subpeak_f.pdf(subpeak_fitted_quantiles)

        # There are no negative read counts and nans
        min_of_fitted = numpy.min(fitted_y)
        for j in range(0, len(fitted_y)):
            if ( numpy.isnan(fitted_y[j]) ):
                fitted_y[j] = 0.0
            fitted_y[j] += abs(min_of_fitted)

        # max min normalize fitted_profile so fitted profile fits better the real data
        max_of_fitted = numpy.max(fitted_y)
        for j in range(0, len(fitted_y)):
            a = ( fitted_y[j] - min_of_fitted )
            b = ( max_of_fitted - min_of_fitted)
            fitted_y[j] =  ( a / b )

        # There are no negative read counts and nans
        for j in range(0, len(fitted_y)):
            if ( fitted_y[j] < 0.0 or numpy.isnan(fitted_y[j]) ):
                fitted_y[j] = 0.0

        # Change amplitude of fitted function
        fitted_y = fitted_y * numpy.max(subpeak_y)

        # Testing for skewness
        testing_skewness_quantiles = numpy.linspace(subpeak_f.ppf(0.001), subpeak_f.ppf(0.999), 100)
        testing_skewness_pdfs = subpeak_f.pdf(testing_skewness_quantiles)

        max_of_skew = numpy.max(testing_skewness_pdfs)
        min_of_skew = numpy.min(testing_skewness_pdfs)

        for j in range(0, len(testing_skewness_pdfs)):
            a = ( testing_skewness_pdfs[j] - min_of_skew )
            b = ( max_of_skew - min_of_skew)
            testing_skewness_pdfs[j] =  ( a / b ) * numpy.max(subpeak_y)
            if( numpy.isnan(testing_skewness_pdfs[j]) ):
                testing_skewness_pdfs[j] = 0.0

        # profile_fig = plt.figure(figsize=(20, 4), dpi=80)
        # ax = profile_fig.add_subplot(1,1,1)
        # ax.plot([x for x in range(100)], testing_skewness_pdfs)
        # ax.set_xlabel('Relative Nucleotide Position')
        # ax.set_ylabel('Intensity')
        # profile_fig.savefig('{}/testprofile.pdf'.format("/home/florian/Documents/github/StoatyPlatsch"), bbox_inches='tight')

        # first condittion for constant profiles
        if ( not all(i == numpy.max(testing_skewness_pdfs) for i in testing_skewness_pdfs) ):
            if( scipy.stats.skewtest(testing_skewness_pdfs)[1] < 0.05 ):
                t = numpy.argmax(testing_skewness_pdfs)
                if (  t < int((len(testing_skewness_pdfs)-1)/2) ):
                    symetric[i] = 1
                else:
                    symetric[i] = -1

        # Check if I have to flip the fitted distribution
        check_flip_a = numpy.sum(abs(subpeak_y - fitted_y))
        check_flip_b = numpy.sum(abs(subpeak_y - numpy.flip(fitted_y)))

        if( check_flip_a > check_flip_b ):
            fitted_y = numpy.flip(fitted_y)
            symetric[i] = symetric[i] * (-1)

        rounded_fitted_y = numpy.round(fitted_y)
        argmax_indices_fitted_y = numpy.argwhere(rounded_fitted_y == numpy.amax(rounded_fitted_y)).flatten()
        argmax_indices_fitted_y.sort()

        if ( len(argmax_indices_fitted_y) != 0 ):
            summit_index_fitted_y = argmax_indices_fitted_y[int(numpy.ceil(len(argmax_indices_fitted_y)/2)-1)]
        else:
            summit_index_fitted_y = numpy.ceil(len(fitted_y)/2)-1

        summit_index_fitted_y = s + summit_index_fitted_y

        shift = peak_indicies[i] - summit_index_fitted_y
        s += shift
        e += shift

        if ( s < 0 ):
            e += abs(s)
            s = 0

        if ( e > (len(y)-1) ):
            s -= abs(len(y)-1-e)
            e = (len(y) - 1)

        full_fitted_y = numpy.zeros(len(y))
        full_fitted_y[s:e] = fitted_y

        fitted_profiles_dict[peak_indicies[i]] = full_fitted_y

        # std = variance^2
        std[i] = numpy.sqrt(subpeak_params[-1])

        # Check if variance is not bigger than probable peak width.
        if ( std[i] >  peak_width ):
            std[i] = peak_width

    return [peak_indicies, found_local_minima, std, fitted_profiles_dict, symetric]
