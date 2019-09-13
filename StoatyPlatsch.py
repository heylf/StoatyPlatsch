
import argparse
import subprocess as sb
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os
import numpy
import sys
import random

from scipy import optimize, signal
from lmfit import models

font = {'family' : 'serif'}
matplotlib.rc('font', **font)

##############
##   FUNC   ##
##############

def hex_to_RGB(hex):
  ''' "#FFFFFF" -> [255,255,255] '''
  # Pass 16 to the integer function for change of base
  return [int(hex[i:i+2], 16) for i in range(1,6,2)]


def RGB_to_hex(RGB):
  ''' [255,255,255] -> "#FFFFFF" '''
  # Components need to be integers for hex to make sense
  RGB = [int(x) for x in RGB]
  return "#"+"".join(["0{0:x}".format(v) if v < 16 else
            "{0:x}".format(v) for v in RGB])

def color_dict(gradient):
  ''' Takes in a list of RGB sub-lists and returns dictionary of
    colors in RGB and hex form for use in a graphing function
    defined later on '''
  return {"hex":[RGB_to_hex(RGB) for RGB in gradient],
      "r":[RGB[0] for RGB in gradient],
      "g":[RGB[1] for RGB in gradient],
      "b":[RGB[2] for RGB in gradient]}

def color_linear_gradient(start_hex, finish_hex="#FFFFFF", n=10):
  ''' returns a gradient list of (n) colors between
    two hex colors. start_hex and finish_hex
    should be the full six-digit color string,
    inlcuding the number sign ("#FFFFFF") '''
  # Starting and ending colors in RGB form
  s = hex_to_RGB(start_hex)
  f = hex_to_RGB(finish_hex)
  # Initilize a list of the output colors with the starting color
  RGB_list = [s]
  # Calcuate a color at each evenly spaced value of t from 1 to n
  for t in range(1, n):
    # Interpolate RGB vector for color at the current value of t
    curr_vector = [
      int(s[j] + (float(t)/(n-1))*(f[j]-s[j]))
      for j in range(3)
    ]
    # Add it to our list of output colors
    RGB_list.append(curr_vector)

  return color_dict(RGB_list)

# Function to obtain the number of lines in a file.
def get_line_count(file):
    count = 0
    for line in file:
        count += 1
    return count

def generate_model(spec, num_padding):
    composite_model = None
    params = None
    x = spec['x']
    y = spec['y']
    x_min = numpy.min(x)
    x_max = numpy.max(x)
    x_range = x_max - x_min
    y_max = numpy.max(y)
    for i, basis_func in enumerate(spec['model']):
        prefix = f'm{i}_'
        model = getattr(models, basis_func['type'])(prefix=prefix)
        if basis_func['type'] in ['GaussianModel', 'LorentzianModel', 'VoigtModel']: # for now VoigtModel has gamma constrained to sigma
            # model.set_param_hint('sigma', min=1e-6, max=x_range-num_padding)
            # model.set_param_hint('center', min=x_min, max=x_max)
            # model.set_param_hint('height', min=1e-6, max=1.1*y_max)
            # model.set_param_hint('amplitude', min=1e-6)

            model.set_param_hint('sigma', min=1e-6, max=(x_range-num_padding)/2 )
            model.set_param_hint('center', min=x_min, max=x_max)
            model.set_param_hint('height', min=1e-6, max=1.1*y_max)
            model.set_param_hint('amplitude', min=1e-6)
        else:
            raise NotImplemented(f'model {basis_func["type"]} not implemented yet')
        if 'help' in basis_func:  # allow override of settings in parameter
            for param, options in basis_func['help'].items():
                model.set_param_hint(param, **options)
        model_params = model.make_params(**basis_func.get('params', {}))
        if params is None:
            params = model_params
        else:
            params.update(model_params)
        if composite_model is None:
            composite_model = model
        else:
            composite_model = composite_model + model
    return composite_model, params

def update_spec_from_peaks(spec, model_indicies, output_folder, minimal_height, peak_width, distance, std, **kwargs):
    x = spec['x']
    y = spec['y']

    # Find maxima and minima
    found_local_maximas = signal.find_peaks(y, width = peak_width, distance = distance, height= [minimal_height, max(y)])

    # inverse data for local minima
    inv_y = y * -1
    found_local_minima = signal.find_peaks(inv_y, width = 2, distance = distance)[0]

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

    print(found_local_minima)

    peak_indicies = numpy.array(found_local_maximas[0])
    peak_indicies.sort()
    print(found_local_maximas)
    print(peak_indicies)

    std_dict = {}
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

        dist_width = right_local_minimum - left_local_minimun

        # Dist / len(x) * 100 = relative length of peak.
        # Relative Length / 4 = Variance.
        # Because Z = (X-mu)/sigma, and I want to have a range of 2*sigma = X-mu, I have Z = 2.
        # Because I make a relation to the standard normal distribution N(0,1) I have finally 2=X or -2=X.
        # So to standardize my peak width in accordance to a standard normal distritbuion I have to divide
        # by a range of 4 (from -2 to 2).
        std_dict[peak_indicies[i]] = (dist_width / len(x) * 100.0) / std

    # Check if two maximas share the same width.
    # This happens if one local maxima is actually a false positive.
    true_positives = []
    if ( len(peak_indicies) > 1 ):
        for a in range(0, len(peak_indicies)):
            peak_to_keep = peak_indicies[a]
            for b in range(1, len(peak_indicies)):
                if ( std_dict[peak_indicies[a]] == std_dict[peak_indicies[b]]
                        and y[peak_indicies[a]] < y[peak_indicies[b]] ):
                    peak_to_keep = peak_indicies[b]
            if ( peak_to_keep not in true_positives ):
                true_positives.append(peak_to_keep)
    else:
        true_positives = peak_indicies

    peak_indicies = numpy.array(true_positives)

    print("Final indices")
    print(peak_indicies)

    if ( len(peak_indicies) != 0):
        fig_extremas, ax = plt.subplots()
        ax.plot(y)
        ax.plot(inv_y)
        ax.plot(peak_indicies, y[peak_indicies], "o")
        ax.plot(found_local_minima, inv_y[found_local_minima], "x")
        ax.set_xlabel('Relative Nucleotide Position')
        ax.set_ylabel('Intensity')
        ax.axes.get_xaxis().set_ticks([])
        fig_extremas.savefig('{}/profile_peaks.pdf'.format(output_folder))

    print(std_dict)

    numpy.random.shuffle(peak_indicies)
    for peak_indicie, model_indicie in zip(peak_indicies.tolist(), model_indicies):
        model = spec['model'][model_indicie]
        if model['type'] in ['GaussianModel', 'LorentzianModel', 'VoigtModel']:
            params = {
                'height': y[peak_indicie],
                #'sigma': x_range / len(x) * numpy.min(peak_widths),
                'sigma': std_dict[peak_indicie], #numpy.ceil((numpy.std(y)/len(y))*2),#x_range / len(x) * 5,
                'center': x[peak_indicie]
            }
            if 'params' in model:
                model.update(params)
            else:
                model['params'] = params
        else:
            raise NotImplemented("Function type not implemented")
    return peak_indicies

def get_best_values(spec, output):
    model_params = {
        'GaussianModel':   ['amplitude', 'sigma'],
        'LorentzianModel': ['amplitude', 'sigma'],
        'VoigtModel':      ['amplitude', 'sigma', 'gamma']
    }
    best_values = output.best_values
    print(best_values)
    centeres = []
    sigma = []
    print('center    model :  amplitude     sigma      gamma')
    for i, model in enumerate(spec['model']):
        prefix = f'm{i}_'
        values = ', '.join(f'{best_values[prefix+param]:8.3f}' for param in model_params[model["type"]])
        print(f'[{best_values[prefix+"center"]:3.3f}] {model["type"]:16}: {values}')
        centeres.append(numpy.floor(best_values[prefix + "center"]))
        sigma.append(best_values[prefix + "sigma"])
    return([centeres, sigma])

def main():

    ####################
    ##   ARGS INPUT   ##
    ####################

    tool_description = """

    """

    # parse command line arguments
    parser = argparse.ArgumentParser(description=tool_description,
                                     usage='%(prog)s [-h] [options] -a *.bed -b *.bam/*bed -c *.txt',
                                     formatter_class=argparse.RawDescriptionHelpFormatter)

    # version
    parser.add_argument(
        "-v", "--version", action="version", version="%(prog)s 0.0")

    # positional arguments
    parser.add_argument(
        "-a", "--input_bed",
        metavar='*.bed',
        required=True,
        help="Path to the peak file in bed6 format.")
    parser.add_argument(
        "-b", "--input_bam",
        metavar='*.bam/*.bed',
        required=True,
        help="Path to the read file used for the peak calling in bed or bam format.")
    parser.add_argument(
        "-c", "--chr_file",
        metavar='*.txt',
        required=True,
        help="Path to the chromosome length file.")

    # optional arguments
    parser.add_argument(
        "-o", "--output_folder",
        metavar='path/',
        default=os.getcwd(),
        help="Write results to this path. [Default: Operating Path]")
    parser.add_argument(
        "--length_norm",
        action='store_true',
        help="Set length normalization. StoatyDive will expand every peak to the maximal length.")
    parser.add_argument(
        "--length_norm_value",
        metavar='int',
        help="Set length normalization value (maximum peak length).")
    parser.add_argument(
        "--max_peaks",
        metavar='int',
        default=100,
        help="Maximal number of possible peaks in a peak profile. [Default: 100]")
    parser.add_argument(
        "--peak_model",
        metavar='str',
        default="GaussianModel",
        help="Type of model for the peaks. Either GaussianModel, LorentzianModel, or VoigtModel. [Default: GaussianModel]")
    parser.add_argument(
        "--min_width",
        metavar='int',
        default=10,
        help="The parameter defines how many peak the tool will find inside the profile. "
             "It defines the distance each peak must be apart. Reducing it might increase the number (overfit). [Default: ]")
    parser.add_argument(
        "--max_width",
        metavar='int',
        default=50,
        help="The parameter defines how many peak the tool will find inside the profile. "
             "It defines the distance each peak must be apart. Increasing it might decrease the number (underfit). [Default: ]")
    parser.add_argument(
        "--steps",
        metavar='int',
        default=1,
        help="")
    parser.add_argument(
        "--std",
        metavar='int',
        default=1,
        help="This parameter defines the width of the newly defined peak binding sites. It is the number of standard"
             "deviations of the undelying modelled distribution.")

    ######################
    ##   CHECKS INPUT   ##
    ######################

    print("[START]")

    args = parser.parse_args()

    # Check if peak file is in bed6 format
    bedfile = open(args.input_bed, "r")
    firstline = bedfile.readline()
    if ( len(firstline.strip("\n").split("\t")) <  6 ):
        sys.exit("[ERROR] Peakfile has to be in bed6 format!")
    bedfile.close()


    # Get the outfile name from the input read file.
    outfilename = args.input_bam.split("/")
    outfilename = outfilename[len(outfilename)-1]
    outfilename = outfilename.replace(".bam", "").replace(".bed", "")

    extended_peak_file_name = "{}/peaks_extended.bed".format(args.output_folder)

    # Find maximal peak length and get the number of peaks from the peak file.
    peaks_file = open(args.input_bed, "r")
    max_peak_len = 0
    num_peaks = 0
    for line in peaks_file:
        num_peaks += 1
        data = line.strip("\n").split("\t")
        start = data[1]
        end = data[2]
        length = int(end) - int(start)

        if (length > max_peak_len):
            max_peak_len = length
    peaks_file.close()

    if ( args.length_norm_value ):
        max_peak_len = int(args.length_norm_value)

    if ( max_peak_len <  10 ):
        sys.exit("[ERROR] Maximal Peak Length has to be at least 10 bases.")

    print("[NOTE] Maximal peak length {}.".format(max_peak_len))

    # Extend the peaks to the maximal length if the parameter is set to true.
    bool_length_norm = 0
    if ( args.length_norm ):

        print("[NOTE] Activate length normalization.")

        # Read in chromosome sizes
        chr_sizes_dict = dict()
        chr_sizes_file = open(args.chr_file, "r")
        for line in chr_sizes_file:
            data = line.strip("\n").split("\t")
            if ( data[0] not in chr_sizes_dict ):
                chr_sizes_dict[data[0]] = int(data[1])
        chr_sizes_file.close()

        # Define new coorindate for peaks. Extend to maximal length.
        peaks_file = open(args.input_bed, "r")
        extended_peak_file = open(extended_peak_file_name, "w")

        for line in peaks_file:
            data = line.strip("\n").split("\t")
            start = int(data[1])
            end = int(data[2])
            peak_length = end - start
            extention_left = numpy.round((max_peak_len-peak_length)/2)
            extentions_right = numpy.round((max_peak_len-peak_length)/2)

            # Check if extention left and right make up the max_peak_length, if not,
            # then add or substract randomly either to left or right some extra bases. This happends
            # because of the rounding.
            current_peak_length = extention_left + extentions_right + peak_length
            if ( current_peak_length < max_peak_len ):
                numpy.random.seed(123)

                if ( numpy.random.randint(low=2, size=1) == 0 ):
                    extention_left +=  max_peak_len - current_peak_length
                else:
                    extentions_right += max_peak_len - current_peak_length

            if ( current_peak_length > max_peak_len):
                numpy.random.seed(123)

                if (numpy.random.randint(low=2, size=1) == 0):
                    extention_left -= current_peak_length - max_peak_len
                else:
                    extentions_right -= current_peak_length - max_peak_len

            # Check if extension goes beyond the borders of the chromosome.
            beyond_left = "false"
            if ( (start - extention_left) < 0 ):
                beyond_left = "true"
            beyond_right = "false"
            if ((end + extentions_right) > chr_sizes_dict[data[0]]):
                beyond_right = "true"

            if ( beyond_left == "true" and beyond_right == "false" ):
                extentions_right += extention_left-start
                extention_left = start

            if ( beyond_left == "false" and beyond_right == "true" ):
                extention_left += (end + extentions_right) - chr_sizes_dict[data[0]]
                extentions_right = chr_sizes_dict[data[0]] - end

            if ( beyond_left == "true" and beyond_right == "true" ):
                extention_left = start
                extentions_right = chr_sizes_dict[data[0]] - end

            start = start - extention_left
            end = end + extentions_right

            # A last checkup if peak length is maximum length.
            if ( (end - start) != max_peak_len and not (beyond_left == "true" and beyond_left == "true") ):
                print("[ERROR] Max length of peaks not reached.")
                print(data)
                print(start)
                print(end)
                print(end - start)
                print(max_peak_len)

            # Write extended peak to file.
            extended_peak_file.write("{}\t{}\t{}\t{}\n".format(data[0], int(start), int(end), "\t".join(data[3:])))

        peaks_file.close()
        extended_peak_file.close()

    else:
        extended_peak_file_name = args.input_bed
        bool_length_norm = 1

    # Generate Coverage file with bedtools
    coverage_file_name = "{}/{}_coverage.tsv".format(args.output_folder, outfilename)
    sb.Popen("bedtools coverage -a {} -b {} -d -s > {}".format(extended_peak_file_name, args.input_bam,
                                                             coverage_file_name), shell=True).wait()

    print("[NOTE] {} peaks will be evaluated.".format(num_peaks))

    # Dictionaries for the algorithm.
    cov_matrix = numpy.empty([num_peaks, max_peak_len])

    # Get the number of lines of the coverage file of bedtools.
    coverage_file = open(coverage_file_name, "r")
    num_coverage_lines = get_line_count(coverage_file)
    coverage_file.close()

    # Calculate mean and variance of peak coverage profiles
    peak_cov_list = []

    coverage_file = open(coverage_file_name, "r")

    peak_counter = -1
    line_count = 0

    # Go over each line of the bedtools coverage file.
    for line in coverage_file:
        line_count += 1
        data = line.strip("\n").split("\t")
        bp = int(data[len(data) - 2])  # bp of the peak
        cov = int(data[len(data)-1])    # Coverage at that bp

        # If the bp == 1 do the negative binomial estimation an start a new peak entry.
        if(bp == 1):
            if( peak_counter != -1 ):

                cov_matrix[peak_counter] = numpy.array(peak_cov_list)

            peak_cov_list = []
            peak_cov_list.append(cov)
            peak_counter += 1
        else:
            peak_cov_list.append(cov)
            # This condition takes the last line of the coverage file into account. Else I will miss the last entry.
            if ( line_count == num_coverage_lines ):
                cov_matrix[peak_counter] = numpy.array(peak_cov_list)

    coverage_file.close()

    ##########################
    ## Start Deconvolution ###
    ##########################

    print("[NOTE] Start Deconvolution")

    x = numpy.array([int(x+1) for x in range(0, len(cov_matrix[1]))])
    #y = cov_matrix[9]
    #y = cov_matrix[15]
    #y = cov_matrix[21]
    #y = cov_matrix[18]
    #y = cov_matrix[47]
    y = cov_matrix[21]

    #TODO
    #Introduce Regression

    # Padding with zero makes sure I will not screw up the fitting. Sometimes if a peak is too close to the border
    # The Gaussian is too big to be fitted and a very borad Guassian will matched to the data.
    num_padding = 10
    x = numpy.array([int(x+1) for x in range(0, len(cov_matrix[1])+num_padding)])
    y = numpy.pad(y, (int(num_padding/2), int(num_padding/2)), 'constant', constant_values=(0, 0))

    profile_fig, ax = plt.subplots()
    ax.plot(y)
    ax.set_xlabel('Relative Nucleotide Position')
    ax.set_ylabel('Intensity')
    ax.axes.get_xaxis().set_ticks([])
    profile_fig.savefig('{}/profile.pdf'.format(args.output_folder))

    models_dict_array = [{'type': args.peak_model} for i in range(0,args.max_peaks)]

    spec = {
        'x': x,
        'y': y,
        'model': models_dict_array
    }

    peaks_indices_array = [i for i in range(0, args.max_peaks)]

    # Peak Detection Plot
    peaks_found = update_spec_from_peaks(spec, peaks_indices_array, args.output_folder, minimal_height=5, peak_width=2, distance=10, std=8)

    output_table_overview = open('{}/final_tab_overview.tsv'.format(args.output_folder), "w")
    output_table_summits = open('{}/final_tab_summits.bed'.format(args.output_folder), "w")
    output_table_new_peaks = open('{}/final_tab_all_peaks.bed'.format(args.output_folder), "w")

    # Check number of potential local maxima
    if( len(peaks_found) != 0 ):

        # Check for distributions to be deleted
        dist_index = 0
        while dist_index < len(spec['model']):
            if 'params' not in spec['model'][dist_index]:
                del spec['model'][dist_index]
            else:
                dist_index += 1

        # Fitting Plot
        model, params = generate_model(spec, num_padding)

        print("Model")
        print(model)
        print("PARAMs")
        print(params)

        output = model.fit(spec['y'], params, x=spec['x'])
        output.plot(data_kws={'markersize': 1})
        plt.savefig('{}/profile_fit.pdf'.format(args.output_folder))

        # Get new peaks
        peaks_in_profile = get_best_values(spec, output)
        num_deconvoluted_peaks = len(peaks_in_profile[0])
        components = output.eval_components(x=spec['x'])

        # Change Coordinates
        peak_start_list = [-1] * num_deconvoluted_peaks
        peak_end_list = [-1] * num_deconvoluted_peaks
        if (num_deconvoluted_peaks > 1):
            for i in range(0, num_deconvoluted_peaks):
                left_right_extension = numpy.floor((peaks_in_profile[1][i] * args.std))
                peak_start_list[i] = peaks_in_profile[0][i] - left_right_extension
                peak_end_list[i] = peaks_in_profile[0][i] + left_right_extension

                if ( peak_start_list[i] < 0 ):
                    peak_start_list[i] = 0

                # TODO
                # if ( end > chr_sizes_dict["chr"] ):
                #    end = chr_sizes_dict["chr"]


        # Write Output tables
        # Check number of potential found peaks
        if ( num_deconvoluted_peaks > 1 ):
            output_table_overview.write("{0}\t{1}\t{2}\n".format("peakid", len(peaks_in_profile), peaks_in_profile))
            for i in range(0, num_deconvoluted_peaks):
                output_table_summits.write("{0}\t{1}\t{2}\t{3}\t{4}\t{5}\n".format("chr", peaks_in_profile[0][i],
                                                                                    peaks_in_profile[0][i], "peakid_" + str(i),
                                                                                    "0.0", "strand"))

                output_table_new_peaks.write("{0}\t{1}\t{2}\t{3}\t{4}\t{5}\n".format("chr", peak_start_list[i], peak_end_list[i],
                                                                                     "peakid_" + str(i), "-1", "strand"))
        else:
            output_table_overview.write("{0}\t{1}\t{2}\n".format("peakid", "1", "just this one"))
            output_table_summits.write("{0}\t{1}\t{2}\t{3}\t{4}\t{5}\n".format("chr", "start",
                                                                           "end", "peakid",
                                                                           "0.0", "strand"))

        # Deconvolution Plot
        # get maximum value of components
        max_fitted_y = 0
        for i, model in enumerate(spec['model']):
            if ( max_fitted_y < numpy.max(components[f'm{i}_']) ):
                max_fitted_y = numpy.max(components[f'm{i}_'])

        max_y_plot = max_fitted_y
        if ( max_fitted_y < max(y) ):
            max_y_plot = max(y)

        c = color_linear_gradient(start_hex="#FF0000", finish_hex="#0000ff", n=num_deconvoluted_peaks)['hex']
        fig, ax = plt.subplots()
        for i, model in enumerate(spec['model']):
            ax.plot(spec['x'], components[f'm{i}_'], color=c[i])
            rect = patches.Rectangle( (peak_start_list[i], 0),
                                      width=peak_end_list[i]-peak_start_list[i],
                                      height=max_y_plot, facecolor=c[i], alpha=0.3)
            ax.add_patch(rect)
        ax.bar(spec['x'], spec['y'], width=1.0, color="black", edgecolor="black")
        ax.set_xlabel('Relative Nucleotide Position')
        ax.set_ylabel('Intensity')
        ax.set_ylim([0, max_y_plot])
        ax.axes.get_xaxis().set_ticks([])
        fig.savefig('{}/profile_deconvolution.pdf'.format(args.output_folder))

    else:
        output_table_overview.write("{0}\t{1}\t{2}\n".format("peakid", "1", "just this one"))
        output_table_summits.write("{0}\t{1}\t{2}\t{3}\t{4}\t{5}\n".format("chr", "start",
                                                                       "end", "peakid",
                                                                       "0.0", "strand"))
    output_table_overview.close()
    output_table_summits.close()
    output_table_new_peaks.close()

if __name__ == '__main__':
    main()