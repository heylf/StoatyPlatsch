
import argparse
import subprocess as sb
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os
import numpy
import sys
import multiprocessing
import time

from rainbow_colors import color_linear_gradient
from deconvolution import deconvolution
from deconvolution_parallel import deconvolution_parallel

font = {'family' : 'serif'}
matplotlib.rc('font', **font)

POSSIBLE_DIST = ['GaussianModel', 'LorentzianModel', 'VoigtModel', 'SkewedGaussianModel', 'SkewedVoigtModel', 'DonaichModel']

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
        "--peak_model",
        metavar='str',
        default="None",
        help="Type of model for the peaks. Either GaussianModel, LorentzianModel, or VoigtModel. [Default: None]")
    parser.add_argument(
        "--min_width",
        metavar='int',
        default=3,
        help="The parameter defines how many peak the tool will find inside the profile. "
             "It defines the distance each peak must be apart. Reducing it might increase the number (overfit). [Default: 3]")
    parser.add_argument(
        "--max_width",
        metavar='int',
        default=10,
        help="The parameter defines how many peak the tool will find inside the profile. "
             "It defines the distance each peak must be apart. Increasing it might decrease the number (underfit). [Default: 10]")
    parser.add_argument(
        "--min_height",
        metavar='int',
        default=10,
        help="In order to find peaks in the profile, the tool searches for local maxima. Increasing the minimal"
             "height will decrease the number of peaks (the number of local maxima). It is the minimal amount"
             "of required reads (events) to be considered. [Default: 10]")
    parser.add_argument(
        "--distance",
        metavar='int',
        default=5,
        help="It is the minimal required distance each local maxima have to be apart. Decreasing the parameter"
             "will result in overfitting (more peaks). [Default: 5]")
    parser.add_argument(
        "--std",
        metavar='int',
        default=2,
        help="This parameter defines the width of the newly defined peak binding sites. It is the number of standard"
             "deviations of the undelying modelled distribution. [Default: 2]")
    parser.add_argument(
        "--min_profile_length",
        metavar='int',
        default=20,
        help="This parameter defines the minimal length of the profile to take it "
             "into considaration for a deconvolution. [Default: 20]")
    parser.add_argument(
        "--max_summits",
        metavar='int',
        default=30,
        help="This parameter defines the maximal number of summits in a profile. The tool takes much longer "
             "if you increase it. Profiles with more summits will be filtered out (> max_summits). It is recommended to change"
             "the parameters for those profiles or check the length of the profiles. It is worth to preprocess them again or "
             "check the peak calling algorithm and the parameters of the peak caller."
             "[Default: 30]")
    parser.add_argument(
        "-t", "--threads",
        metavar='int',
        default=1,
        help="Number of threads for parallelization. [Default: 1]")

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

    if ( args.peak_model != "None" and args.peak_model not in POSSIBLE_DIST ):
        sys.exit("[ERROR] Peakmodel not supported.")


    #TODO include a forcing to use a specific model

    #######################
    ##   PREPROCESSING   ##
    #######################

    # Get the outfile name from the input read file.
    outfilename = args.input_bam.split("/")
    outfilename = outfilename[len(outfilename)-1]
    outfilename = outfilename.replace(".bam", "").replace(".bed", "")

    extended_peak_file_name = "{}/peaks_extended.bed".format(args.output_folder)

    # Get peak lengths and get the number of peaks from the peak file.
    peaks_file = open(args.input_bed, "r")
    max_peak_len = 0
    num_peaks = 0
    data_dict = {}
    peak_length_dict = {}
    for line in peaks_file:
        data = line.strip("\n").split("\t")
        data_dict[num_peaks] = data
        start = data[1]
        end = data[2]
        length = int(end) - int(start)
        peak_length_dict[num_peaks] = length

        if (length > max_peak_len):
            max_peak_len = length

        num_peaks += 1
    peaks_file.close()

    if ( args.length_norm_value ):
        max_peak_len = int(args.length_norm_value)

    if ( max_peak_len <  10 ):
        sys.exit("[ERROR] Maximal Peak Length has to be at least 10 bases.")

    # Read in chromosome sizes
    chr_sizes_dict = dict()
    chr_sizes_file = open(args.chr_file, "r")
    for line in chr_sizes_file:
        data = line.strip("\n").split("\t")
        if (data[0] not in chr_sizes_dict):
            chr_sizes_dict[data[0]] = int(data[1])
    chr_sizes_file.close()

    print("[NOTE] Maximal peak length {}.".format(max_peak_len))

    # Extend the peaks to the maximal length if the parameter is set to true.
    if ( args.length_norm ):

        print("[NOTE] Activate length normalization.")

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

    # Generate Coverage file with bedtools
    coverage_file_name = "{}/{}_coverage.tsv".format(args.output_folder, outfilename)
    sb.Popen("bedtools coverage -a {} -b {} -d -s > {}".format(extended_peak_file_name, args.input_bam,
                                                             coverage_file_name), shell=True).wait()

    print("[NOTE] {} peaks will be evaluated.".format(num_peaks))

    # Dictionaries for the algorithm.
    cov_matrix = dict()

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

    number_of_threads = int(args.threads)

    print("[NOTE] Using " + str(number_of_threads) + " threads")
    print("[NOTE] Start Deconvolution")

    # Read in extended peak file of the coverage calculation for peak meta data.
    for_data_peaks_file = open(extended_peak_file_name, "r")
    data_dict = {}
    counter_peaks = 0
    for line in for_data_peaks_file:
        data_dict[counter_peaks] = line.strip("\n").split("\t")
        counter_peaks += 1
    for_data_peaks_file.close()

    manager = multiprocessing.Manager()
    deconvolution_dict = manager.dict()

    # Padding with zero makes sure I will not screw up the fitting. Sometimes if a peak is too close to the border
    # The Gaussian is too big to be fitted and a very borad Guassian will matched to the data.
    num_padding = 40

    # for peak in range(145, 146):
    #
    #     if ( peak_length_dict[peak] > 20 ):
    #         start_time = time.time()
    #         deconvolution(args.output_folder, peak, cov_matrix[peak], args.peak_model, args.max_peaks,
    #                                               args.min_width, args.max_width, args.min_height, args.distance,
    #                                               num_padding, deconvolution_dict)
    #         end_time = time.time()
    #
    #         print('function took {} s'.format(end_time - start_time))
    #
    #         if ( (end_time - start_time) > 10 ):
    #             print('function took {} s'.format( end_time - start_time ) )
    #             print(peak)
    #             print(peak_length_dict[peak])
    #             print(data_dict[peak])
    # sys.exit()

    print("[NOTE] Fitting Model")
    start_time = time.time()

    # First deconvolute all profiles with less than 10 summits in parallel.
    pool = multiprocessing.Pool(number_of_threads)
    for peak in range(0, num_peaks):

        if ( peak_length_dict[peak] > args.min_profile_length ):
            print(peak)
            pool.apply_async(deconvolution, args=(peak, cov_matrix[peak], args.peak_model,
                                                  args.min_width, args.max_width, args.min_height, args.distance,
                                                  num_padding, deconvolution_dict))
    pool.close()
    pool.join()

    # Then deconvoltue profiles with more than 10 summits in serial but the convolution is in parallel.
    # This will increase speed, because the deconvolution has a exponential runtime (30 summit ~ 2-5 minutes).
    for peak in range(0, num_peaks):

        if ( peak_length_dict[peak] > args.min_profile_length and peak not in deconvolution_dict):
            print(peak)
            deconvolution_parallel(peak, cov_matrix[peak], args.peak_model,
                                   args.min_width, args.max_width, args.min_height, args.distance,
                                   num_padding, deconvolution_dict, number_of_threads, args.max_summits)

    end_time = time.time()
    print('function took {} s'.format( end_time - start_time ) )

    print("[NOTE] Generate Output")

    output_table_overview = open('{}/final_tab_overview.tsv'.format(args.output_folder), "w")
    output_table_summits = open('{}/final_tab_summits.bed'.format(args.output_folder), "w")
    output_table_new_peaks = open('{}/final_tab_all_peaks.bed'.format(args.output_folder), "w")

    fig_profile = plt.figure(figsize=(20, 4), dpi=80)
    fig_extremas = plt.figure(figsize=(20, 4), dpi=80)
    fig_deconvolution = plt.figure(figsize=(20, 4), dpi=80)

    fig_profile.subplots_adjust(hspace=0.3, wspace=0.3)
    fig_extremas.subplots_adjust(hspace=0.3, wspace=0.3)
    fig_deconvolution.subplots_adjust(hspace=0.3, wspace=0.3)

    plot_counter = 1

    for peak in range(0, num_peaks):

        # data of the peak
        data = data_dict[peak]
        chr = data[0]
        start = int(data[1])
        end = int(data[2])
        id = data[3]
        score = data[4]
        strand = data[5]

        # without x+1 because genome coordinates starts at zero (end-1, see info bedtools coverage)
        pre_x = numpy.array([x for x in range(0, len(cov_matrix[peak]))])
        pre_y = cov_matrix[peak]

        # without x+1 because genome coordinates starts at zero (end-1, see info bedtools coverage)
        x = numpy.array([x for x in range(0, len(pre_x) + num_padding)])
        y = numpy.pad(pre_y, (int(num_padding / 2), int(num_padding / 2)), 'constant', constant_values=(0, 0))
        inv_y = y * -1

        new_start = start - int(num_padding / 2)
        new_end = end + int(num_padding / 2)
        real_coordinates_list = numpy.array([x for x in range(new_start, new_end)])

        # Check number of potential local maxima
        if (peak in deconvolution_dict):

            # Get new peaks
            peaks_found = deconvolution_dict[peak][0]
            found_local_minima = deconvolution_dict[peak][1]
            spec = deconvolution_dict[peak][2]
            sigma_of_peaks = deconvolution_dict[peak][3]
            components = deconvolution_dict[peak][4]

            # Get new peaks
            num_deconvoluted_peaks = len(peaks_found)

            peak_start_list = [start] * num_deconvoluted_peaks
            peak_end_list = [end] * num_deconvoluted_peaks
            peak_center_list = [-1] * num_deconvoluted_peaks

            rectangle_start_list = [-1] * num_deconvoluted_peaks
            rectangle_end_list = [-1] * num_deconvoluted_peaks

            for i in range(0, num_deconvoluted_peaks):
                peak_center = int(peaks_found[i])
                peak_center_list[i] = real_coordinates_list[peak_center]
                peak_sigma = sigma_of_peaks[i]

                # Change Coordinates
                left_right_extension = numpy.floor((peak_sigma * args.std))

                peak_start_list[i] = real_coordinates_list[peak_center] - left_right_extension
                peak_end_list[i] = real_coordinates_list[peak_center] + left_right_extension + 1
                # end+1 because genome coordinates starts at zero (end-1, see info bedtools coverage)

                rectangle_start_list[i] = peak_center - left_right_extension
                rectangle_end_list[i] = peak_center + left_right_extension

                if ( peak_start_list[i] < 0 ):
                    peak_start_list[i] = 0

                if ( peak_end_list[i] > chr_sizes_dict[chr] ):
                   peak_end_list[i] = chr_sizes_dict[chr]

                # Write Output tables
                # Check number of potential found peaks
                output_table_summits.write("{0}\t{1}\t{1}\t{2}\t{3}\t{4}\n".format(chr, int(peak_center_list[i]), id + "_" + str(i),
                                                                                    score, strand))
                output_table_new_peaks.write("{0}\t{1}\t{2}\t{3}\t{4}\t{5}\n".format(chr, int(peak_start_list[i]), int(peak_end_list[i]),
                                                                                     id + "_" + str(i), score, strand))
            output_table_overview.write("{0}\t{1}\t{2}\n".format(id, len(peak_center_list), peak_center_list))

            # Plot Area
            if (len(peaks_found) > 1 and plot_counter <= 10):
                ax = fig_profile.add_subplot(2, 5, plot_counter)
                ax.plot(pre_x, pre_y)
                ax.set_xlabel('Relative Nucleotide Position')
                ax.set_ylabel('Intensity')
                #ax.axes.get_xaxis().set_ticks([])

                # get maximum value of components
                max_fitted_y = 0
                for i, model in enumerate(spec['model']):
                    if (max_fitted_y < numpy.max(components[f'm{i}_'])):
                        max_fitted_y = numpy.max(components[f'm{i}_'])

                max_y_plot = max_fitted_y
                if (max_fitted_y < max(y)):
                    max_y_plot = max(y)

                ax2 = fig_extremas.add_subplot(2, 5, plot_counter)
                ax2.plot(x, y)
                ax2.plot(x, inv_y)
                ax2.plot(peaks_found, y[peaks_found], "o", markersize=2)
                ax2.plot(found_local_minima, inv_y[found_local_minima], "x", markersize=2)
                ax2.set_xlabel('Relative Nucleotide Position')
                ax2.set_ylabel('Intensity')
                ax2.set_ylim([min(inv_y), max_y_plot])

                # Deconvolution Plot
                c = color_linear_gradient(start_hex="#FF0000", finish_hex="#0000ff", n=num_deconvoluted_peaks)['hex']
                ax3 = fig_deconvolution.add_subplot(2, 5, plot_counter)
                # Add rectangles
                for i, model in enumerate(spec['model']):
                    ax3.plot(spec['x'], components[f'm{i}_'], color=c[i])
                    rect = patches.Rectangle((rectangle_start_list[i], 0),
                                             width=rectangle_end_list[i] - rectangle_start_list[i],
                                             height=max_y_plot, facecolor=c[i], alpha=0.3)
                    ax3.add_patch(rect)
                ax3.bar(spec['x'], spec['y'], width=1.0, color="black", edgecolor="black")
                ax3.set_xlabel('Relative Nucleotide Position')
                ax3.set_ylabel('Intensity')
                ax3.set_ylim([0, max_y_plot])
                #ax3.axes.get_xaxis().set_ticks([])

                plot_counter += 1
        else:
            output_table_overview.write("{0}\t{1}\t{2}\n".format(id, "1", start))
            output_table_new_peaks.write("{0}\t{1}\t{2}\t{3}\t{4}\t{5}\n".format(chr, int(start), int(end), id, score, strand))
            summit = real_coordinates_list[numpy.argmax(pre_y)]
            output_table_summits.write("{0}\t{1}\t{2}\t{3}\t{4}\t{5}\n".format(chr, int(summit), int(summit), id, score, strand))

    fig_profile.savefig('{}/profile.pdf'.format(args.output_folder), bbox_inches='tight')
    fig_extremas.savefig('{}/profile_peaks.pdf'.format(args.output_folder), bbox_inches='tight')
    fig_deconvolution.savefig('{}/profile_deconvolution.pdf'.format(args.output_folder), bbox_inches='tight')

    output_table_overview.close()
    output_table_summits.close()
    output_table_new_peaks.close()

    print("[FINISH]")

if __name__ == '__main__':
    main()