
from scipy import signal

import argparse
import subprocess as sb
import matplotlib.pyplot as plt
import os
import numpy
import math
import sys
import random

from scipy import optimize, signal
from lmfit import models

plt.switch_backend('agg')

##############
##   FUNC   ##
##############

# Function to obtain the number of lines in a file.
def get_line_count(file):
    count = 0
    for line in file:
        count += 1
    return count

def generate_model(spec):
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
            model.set_param_hint('sigma', min=1e-6, max=x_range)
            model.set_param_hint('center', min=x_min, max=x_max)
            model.set_param_hint('height', min=1e-6, max=1.1*y_max)
            model.set_param_hint('amplitude', min=1e-6)
            # default guess is horrible!! do not use guess()
            default_params = {
                prefix+'center': x_min + x_range * random.random(),
                prefix+'height': y_max * random.random(),
                prefix+'sigma': x_range * random.random()
            }
        else:
            raise NotImplemented(f'model {basis_func["type"]} not implemented yet')
        if 'help' in basis_func:  # allow override of settings in parameter
            for param, options in basis_func['help'].items():
                model.set_param_hint(param, **options)
        model_params = model.make_params(**default_params, **basis_func.get('params', {}))
        if params is None:
            params = model_params
        else:
            params.update(model_params)
        if composite_model is None:
            composite_model = model
        else:
            composite_model = composite_model + model
    return composite_model, params

def update_spec_from_peaks(spec, model_indicies, peak_widths, **kwargs):
    x = spec['x']
    y = spec['y']
    x_range = numpy.max(x) - numpy.min(x)
    peak_indicies = signal.find_peaks_cwt(y, peak_widths)
    numpy.random.shuffle(peak_indicies)
    for peak_indicie, model_indicie in zip(peak_indicies.tolist(), model_indicies):
        model = spec['model'][model_indicie]
        if model['type'] in ['GaussianModel', 'LorentzianModel', 'VoigtModel']:
            params = {
                'height': y[peak_indicie],
                'sigma': x_range / len(x) * numpy.min(peak_widths),
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
    print('center    model   amplitude     sigma      gamma')
    for i, model in enumerate(spec['model']):
        prefix = f'm{i}_'
        if (numpy.floor(best_values[prefix + 'amplitude']) != 0.0 and numpy.floor(best_values[prefix + 'sigma']) < 100):
            values = ', '.join(f'{best_values[prefix+param]:8.3f}' for param in model_params[model["type"]])
            print(f'[{best_values[prefix+"center"]:3.3f}] {model["type"]:16}: {values}')
            centeres.append(numpy.floor(best_values[prefix + "center"]))
    return(centeres)

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
        default=4,
        help="Maximal number of possible peaks in a peak profile. [Default: 10]")
    parser.add_argument(
        "--peak_model",
        metavar='str',
        default="GaussianModel",
        help="Type of model for the peaks. Either GaussianModel, LorentzianModel, or VoigtModel. [Default: GaussianModel]")
    parser.add_argument(
        "--min_width",
        metavar='int',
        default=3,
        help="The parameter defines how many peak the tool will find inside the profile. "
             "It defines the distance each peak must be apart. Reducing it might increase the number (overfit). [Default: ]")
    parser.add_argument(
        "--max_width",
        metavar='int',
        default=20,
        help="The parameter defines how many peak the tool will find inside the profile. "
             "It defines the distance each peak must be apart. Increasing it might decrease the number (underfit). [Default: ]")
    parser.add_argument(
        "--steps",
        metavar='int',
        default=5,
        help="")


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
    y = cov_matrix[18]

    plt.plot(y)
    plt.xlabel('Relative Nucleotide Position')
    plt.ylabel('Intensity')
    plt.savefig('{}/profile.pdf'.format(args.output_folder))

    models_dict_array = [{'type': args.peak_model} for i in range(0,args.max_peaks)]

    spec = {
        'x': x,
        'y': y,
        'model': models_dict_array
    }

    peaks_indices_array = [i for i in range(0, args.max_peaks)]

    # Peak Detection Plot
    peaks_found = update_spec_from_peaks(spec, peaks_indices_array, peak_widths=(numpy.linspace(args.min_width, args.max_width, args.steps)))

    output_table_overview = open('{}/final_tab1.tsv'.format(args.output_folder), "w")
    output_table_bed = open('{}/final_tab2.bed'.format(args.output_folder), "w")

    if( len(peaks_found) != 0 ):

        plt.plot(peaks_found, y[peaks_found], "x")
        plt.savefig('{}/profile_peaks.pdf'.format(args.output_folder))

        # Check for distributions to be deleted
        dist_index = 0
        while dist_index < len(spec['model']):
            if 'params' not in spec['model'][dist_index]:
                del spec['model'][dist_index]
            else:
                dist_index += 1

        # Fitting Plot
        model, params = generate_model(spec)
        output = model.fit(spec['y'], params, x=spec['x'])
        output.plot(data_kws={'markersize': 1})
        plt.savefig('{}/profile_fit.pdf'.format(args.output_folder))

        # Deconvolution Plot
        fig, ax = plt.subplots()
        ax.scatter(spec['x'], spec['y'], s=4)
        components = output.eval_components(x=spec['x'])
        for i, model in enumerate(spec['model']):
            ax.plot(spec['x'], components[f'm{i}_'])
        fig.savefig('{}/profile_deconvolution.pdf'.format(args.output_folder))

        # Output
        peaks_in_profile = get_best_values(spec, output)

        if ( len(peaks_in_profile) > 1 ):
            output_table_overview.write("{0}\t{1}\t{2}\n".format("peakid", len(peaks_in_profile), peaks_in_profile))
            for i in range(0, len(peaks_in_profile)):
                output_table_bed.write("{0}\t{1}\t{2}\t{3}\t{4}\t{5}\n".format("chr", peaks_in_profile[i],
                                                                                    peaks_in_profile[i], "peakid_" + str(i),
                                                                                    "0.0", "strand"))
        else:
            output_table_overview.write("{0}\t{1}\t{2}\n".format("peakid", "1", "just this one"))
            output_table_bed.write("{0}\t{1}\t{2}\t{3}\t{4}\t{5}\n".format("chr", "start",
                                                                           "end", "peakid",
                                                                           "0.0", "strand"))
    else:
        output_table_overview.write("{0}\t{1}\t{2}\n".format("peakid", "1", "just this one"))
        output_table_bed.write("{0}\t{1}\t{2}\t{3}\t{4}\t{5}\n".format("chr", "start",
                                                                       "end", "peakid",
                                                                       "0.0", "strand"))
    output_table_overview.close()
    output_table_bed.close()

if __name__ == '__main__':
    main()