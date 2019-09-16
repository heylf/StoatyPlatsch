
import argparse
import subprocess as sb
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os
import numpy
import sys

from scipy.interpolate import UnivariateSpline
from update_spec import update_spec_from_peaks
from generate_model import generate_model
from rainbow_colors import color_linear_gradient
from get_best_values import get_best_values

font = {'family' : 'serif'}
matplotlib.rc('font', **font)

##############
##   FUNC   ##
##############

# Function to obtain the number of lines in a file.
def get_line_count(file):
    count = 0
    for line in file:
        count += 1
    return count

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
    y = cov_matrix[49]

    profile_fig, ax = plt.subplots()
    ax.plot(y)
    ax.set_xlabel('Relative Nucleotide Position')
    ax.set_ylabel('Intensity')
    ax.axes.get_xaxis().set_ticks([])
    profile_fig.savefig('{}/profile.pdf'.format(args.output_folder))

    arg_s = 0.01
    arg_s = 3500 * arg_s

    #Introduce Regression
    regression_s = UnivariateSpline(x, y, s=arg_s)
    ys = regression_s(x)
    profile_smoothed_fig, ax = plt.subplots()
    ax.plot(x, ys)
    ax.set_xlabel('Relative Nucleotide Position')
    ax.set_ylabel('Intensity')
    ax.axes.get_xaxis().set_ticks([])
    profile_smoothed_fig.savefig('{}/profile_smoothed.pdf'.format(args.output_folder))

    # Padding with zero makes sure I will not screw up the fitting. Sometimes if a peak is too close to the border
    # The Gaussian is too big to be fitted and a very borad Guassian will matched to the data.
    num_padding = 40
    x = numpy.array([int(x+1) for x in range(0, len(cov_matrix[1])+num_padding)])
    y = numpy.pad(y, (int(num_padding/2), int(num_padding/2)), 'constant', constant_values=(0, 0))

    models_dict_array = [{'type': args.peak_model} for i in range(0,args.max_peaks)]

    spec = {
        'x': x,
        'y': y,
        'model': models_dict_array
    }

    peaks_indices_array = [i for i in range(0, args.max_peaks)]

    # Peak Detection Plot
    peaks_found = update_spec_from_peaks(spec, peaks_indices_array, args.output_folder, minimal_height=5, distance=10, std=2)

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
        model, params = generate_model(spec, min_peak_width=5, max_peak_width=10)

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