import numpy
import matplotlib.pyplot as plt

from scipy import signal

def update_spec_from_peaks(spec, model_indicies, minimal_height, distance, std, **kwargs):
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

    start_index = 1
    pos = 0
    while( diff_y[pos] == 0 and pos != (len(diff_y)-1) ):
        start_index += 1
        pos += 1

    end_index = len(x)-2
    pos = len(x)-2
    while( diff_y[pos] == 0 and pos != 0 ):
        end_index -= 1
        pos -= 1

    found_local_minima = numpy.concatenate( ([start_index], found_local_minima, [end_index]) )

    peak_indicies = numpy.array(found_local_maximas[0])
    peak_indicies.sort()

    std_dict = {}
    width_dict = {}
    left_local_minimun_dict = {}
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

        left_local_minimun_dict[peak_indicies[i]] = left_local_minimun
        right_local_minimum_dict[peak_indicies[i]] = right_local_minimum
        dist_width = right_local_minimum - left_local_minimun

        # Dist / len(x) * 100 = relative length of peak.
        # Relative Length / 4 = Variance.
        # Because Z = (X-mu)/sigma, and I want to have a range of 2*sigma = X-mu, I have Z = 2.
        # Because I make a relation to the standard normal distribution N(0,1) I have finally 2=X or -2=X.
        # So to standardize my peak width in accordance to a standard normal distritbuion I have to divide
        # by a range of 4 (from -2 to 2).
        width_dict[peak_indicies[i]] = dist_width
        std_dict[peak_indicies[i]] = (dist_width / len(x) * 100.0) / std

    # Check if two maximas share the same width.
    # This happens if one local maxima is actually a false positive.
    true_positives = []
    if ( len(peak_indicies) > 1 ):
        for a in range(0, len(peak_indicies)):
            peak_to_keep = peak_indicies[a]
            for b in range(1, len(peak_indicies)):
                if ( left_local_minimun_dict[peak_indicies[a]] == left_local_minimun_dict[peak_indicies[b]] and
                     right_local_minimum_dict[peak_indicies[a]] == right_local_minimum_dict[peak_indicies[b]] and
                     y[peak_indicies[a]] < y[peak_indicies[b]] ):
                    peak_to_keep = peak_indicies[b]
            if ( peak_to_keep not in true_positives ):
                true_positives.append(peak_to_keep)
    else:
        true_positives = peak_indicies

    peak_indicies = numpy.array(true_positives)

    numpy.random.shuffle(peak_indicies)
    for peak_indicie, model_indicie in zip(peak_indicies.tolist(), model_indicies):
        model = spec['model'][model_indicie]
        if model['type'] in ['GaussianModel', 'LorentzianModel', 'VoigtModel']:
            params = {
                'height': y[peak_indicie],
                #'sigma': x_range / len(x) * numpy.min(peak_widths),
                'sigma': std_dict[peak_indicie], #numpy.ceil((numpy.std(y)/len(y))*2),#x_range / len(x) * 5,
                'center': x[peak_indicie],
                'width': width_dict[peak_indicie]
            }
            if 'params' in model:
                model.update(params)
            else:
                model['params'] = params
        else:
            raise NotImplemented("Function type not implemented")
    return [peak_indicies,found_local_minima]
