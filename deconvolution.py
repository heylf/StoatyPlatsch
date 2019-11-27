import numpy
import sys

from update_spec import update_spec_from_peaks
from generate_model import generate_model

POSSIBLE_DIST = ['GaussianModel', 'LorentzianModel', 'VoigtModel', 'SkewedGaussianModel', 'SkewedVoigtModel', 'DonaichModel']

def deconvolution(peak, pre_y, peak_model, min_width, max_width, min_height, distance, num_padding, deconvolution_dict):

    # without x+1 because genome coordinates starts at zero (end-1, see info bedtools coverage)
    pre_x = numpy.array([x for x in range(0, len(pre_y))])

    # without x+1 because genome coordinates starts at zero (end-1, see info bedtools coverage)
    x = numpy.array([x for x in range(0, len(pre_x) + num_padding)])
    y = numpy.pad(pre_y, (int(num_padding / 2), int(num_padding / 2)), 'constant', constant_values=(0, 0))

    models_dict_array = [{'type': 'GaussianModel'} for i in range(0, 100)]

    if ( peak_model != "None" ):
        models_dict_array = [{'type': peak_model} for i in range(0, 100)]

    spec = {
        'x': x,
        'y': y,
        'model': models_dict_array
    }

    peaks_indices_array = [i for i in range(0, 100)]

    # Peak Detection Plot
    list_of_update_spec_from_peaks = update_spec_from_peaks(spec, peaks_indices_array, minimal_height=min_height,
                                                            distance=distance, std=2, localheight=True)
    peaks_found = list_of_update_spec_from_peaks[0]
    found_local_minima = list_of_update_spec_from_peaks[1]

    # Check number of potential local maxima
    if ( len(peaks_found) > 1 and len(peaks_found) < 9 ):

        # Check for distributions to be deleted
        dist_index = 0
        while dist_index < len(spec['model']):
            if 'params' not in spec['model'][dist_index]:
                del spec['model'][dist_index]
            else:
                dist_index += 1

        # Fitting Plot
        if (peak_model == "None"):
            for m in spec['model']:

                bic_dict = dict()

                for d in POSSIBLE_DIST:
                    m['type'] = d
                    #print(d)
                    model, params = generate_model(spec, min_width, max_width)
                    try:
                        output = model.fit(spec['y'], params, x=spec['x'], nan_policy='propagate')
                        bic_dict[d] = output.bic
                    except:
                        print("[ERROR 1] Fitting Problem. Model will be discarded and newly optimized.")
                        bic_dict[d] = 1000000

                m['type'] = min(bic_dict, key=bic_dict.get)

        model, params = generate_model(spec, min_peak_width=min_width, max_peak_width=max_width)

        output = None
        optimizer_counter = 0
        while output is None and optimizer_counter != 10:
            try:
                output = model.fit(spec['y'], params, x=spec['x'], nan_policy='raise')
            except:
                print("[ERROR 2] Fitting Problem. Model will be discarded and newly optimized.")
                optimizer_counter += 1
                pass

        if ( optimizer_counter == 10 ):
            sys.exit("[FATAL ERROR 1] Optimization problem occured. Try to change hyperparameters.")

        sigma_of_peaks = []
        best_values=output.best_values
        for i, model in enumerate(spec['model']):
            sigma_key = f'm{i}_' + "sigma"
            if ( sigma_key in best_values ):
                sigma_of_peaks.append(best_values[f'm{i}_' + "sigma"])
            else:
                sigma_of_peaks.append((best_values[f'm{i}_' + "sigma1"] + best_values[f'm{i}_' + "sigma2"])/2)

        components = output.eval_components(x=x)

        print("yes")
        print(peak)
        deconvolution_dict[peak] = [peaks_found, found_local_minima, spec, sigma_of_peaks, components]
    else:
        print("no")
        print(peak)
