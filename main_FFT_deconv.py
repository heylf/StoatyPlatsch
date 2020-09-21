
import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

from FFT.plotting import (create_deconv_profile_plots,
                          create_FFT_analysis_plots, create_profile_plots)
from FFT.postprocessing import (create_output_files,
                                refine_peaks_with_annotations)
from FFT.preprocessing import create_coverage_file, read_coverage_file
from FFT.processing import analyze_with_FFT, deconvolute_with_FFT


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description=("Deconvolutes the given peak profiles with a FFT"
                     " approach."),
        usage="%(prog)s [-h] [options] -a <bed> -b <bam>/<bed>"
              "    |    %(prog)s [-h] [options] -i <tsv>"
        )

    group_required = parser.add_argument_group(
        "required arguments (-a <bed> -b <bam>/<bed> | -i <tsv>)"
        )

    group_required.add_argument(
        "-a", "--input_bed",
        metavar="<bed>",
        help="Path to the peak file in bed6 format."
        )
    group_required.add_argument(
        "-b", "--input_bam",
        metavar="<bam>/<bed>",
        help=("Path to the read file used for the peak calling in bed or bam"
              " format."
              )
        )

    group_required.add_argument(
        "-i", "--input_coverage_file",
        help="Path to the coverage file in tsv format.",
        metavar="<tsv>"
        )

    parser.add_argument(
        "-o", "--output_folder",
        default=os.getcwd(),
        help=("Write results to this path."
              " (default: current working directory)"),
        metavar='path/to/output_folder'
        )
    parser.add_argument(
        "--verbose",
        action='store_true',
        help="Print information to console."
        )
    parser.add_argument(
        "--no_sorting",
        action='store_true',
        help=("If the coverage file is not provided, it is calculated by"
              " sorting the input file and using the bedtools with the sorted"
              " option, thus reducing the memory consumption. Add this"
              " argument to calculate the coverage file without sorting. If"
              " the coverage file is already given as input file, this"
              " argument has no effect.")
        )
    parser.add_argument(
        "--fft_analysis",
        action='store_true',
        help=("Performs additional calculations and creates additional plots"
              " for analyzing the FFT approach.")
        )

    # Define arguments for configuring the deconvolution behavior.
    parser.add_argument(
        "--num_padding",
        default=[10, 10],
        nargs=2,
        help=("Number of zeros that should be padded to the left and right"
              " side of the peak profiles, respectively. (default: [10, 10])"),
        type=int
        )
    parser.add_argument(
        "--deconv_approach",
        default="map_FFT_signal",
        choices=["smooth", "map_profile", "map_FFT_signal"],
        help=("Define the FFT approach for deconvoluting the peaks."
              " 'smooth': Uses FFT to smooth the peak profile and then use the"
              "           smoothed profile to identify the local minima and"
              "           maxima. These maxima define the new peaks with the"
              "           minima as boundaries."
              " 'map_profile': Calculates the peaks (maxima) of the original"
              "                profile and the underlying FFT frequencies."
              "                Maps the peaks of the profile to the frequency"
              "                that has the maximum closest to the peak"
              "                maximum and is also part of the frequencies"
              "                contributing the most to the signal. Uses"
              "                the mapped maximum of the frequency and the"
              "                distance to its minima as peak center and"
              "                width."
              " 'map_FFT_signal': Calculates the peaks (maxima) of the"
              "                   original profile and the underlying FFT"
              "                   frequencies. Maps each of the frequencies"
              "                   that contributes most to the signal to one"
              "                   maximum of the profile. Uses the mapped"
              "                   maximum of the frequency and the distance"
              "                   to its minima as peak center and width."
              " (default: 'map_FFT_signal')"
              )
        )
    parser.add_argument(
        "--clip_boundary",
        default="peak",
        choices=["peak", "padding"],
        help=("Defines the limits of the subpeaks for the peaks boundaries."
              " 'peak': Uses the original, unpadded peak boundaries as"
              "         outermost boundaries."
              " 'padding': Uses the padded peak boundaries as outermost"
              "            boundaries."
              " (default: 'peak')"
              )
        )
    parser.add_argument(
        "--distance",
        default=10,
        help=("Used as parameter for function 'find_peaks' when calculating"
              " the maxima of the original profile. Only used for the"
              " 'map_profile' and the 'map_FFT_signal' approaches."
              "  (default: 10)"),
        type=int
        )
    parser.add_argument(
        "--height",
        default=10,
        help=("Used as parameter for function 'find_peaks' when calculating"
              " the maxima of the original profile. Only used for the"
              " 'map_profile' and the 'map_FFT_signal' approaches."
              "  (default: 10)"),
        type=int
        )
    parser.add_argument(
        "--prominence",
        default=2,
        help=("Used as parameter for function 'find_peaks' when calculating"
              " the maxima of the original profile. Only used for the"
              " 'map_profile' and the 'map_FFT_signal' approaches."
              "  (default: 2)"),
        type=int
        )
    parser.add_argument(
        "--disable_frequency_shift",
        action='store_true',
        help=("Disables shifting the underlying FFT frequencies to the mapped"
              " maxima of the profiles. Only used for the 'map_profile' and"
              " the 'map_FFT_signal' approaches.")
        )
    parser.add_argument(
        "--main_freq_filter_value",
        default=3,
        help=("Defines the number of found maxima in the peak profile that are"
              " used for filtering the main frequency. Is the number of maxima"
              " equal to the defined value, the main frequency will be ignored"
              " for the deconvolution. A value <= 0 disables filtering the"
              " main frequency. Only used for the 'map_profile' and the"
              " 'map_FFT_signal' approaches.(default: 3)"),
        metavar="FILTER_VALUE",
        type=int
        )

    # Define arguments for configuring the plotting behavior.
    plot_group = parser.add_mutually_exclusive_group()
    plot_group.add_argument(
        "--plot_limit",
        default=10,
        help=("Defines the maximum number of peaks for which plots are"
              " created. If the number of peaks is larger than this value, the"
              " peaks for plotting are chosen equally distributed. For values"
              " < 0 plots are created for all peaks. Cannot be used with"
              " argument 'plot_peak_ids'. (default: 10)"),
        type=int
        )
    plot_group.add_argument(
        "--plot_peak_ids",
        help=("Defines the ids of the peaks for which the plots should be"
              " created. If no list is provided the ids are calculated"
              " using the argument 'plot_limit' and the number of peaks"
              " provided by the input file."),
        metavar="PEAK_ID",
        nargs="+",
        type=int
        )
    parser.add_argument(
        "--plot_format",
        default="svg",
        help=("The file format which is used for saving the plots. See"
              " matplotlib documentation for supported values.")
        )

    # Optional arguments for annotations
    parser.add_argument(
        "--gene_file",
        metavar='<bed>',
        help="Path to the gene annotation file.")
    parser.add_argument(
        "--exon_file",
        metavar='<bed>',
        help="Path to the exon boundary file.")

    args = parser.parse_args()

    if args.verbose:
        print("[START]")

    # Determine what files are provided as input and create coverage file if
    # it was not given as input.
    provided_input_args_ok = \
        ((args.input_bed is not None and args.input_bam is not None
          and args.input_coverage_file is None)
         or (args.input_bed is None and args.input_bam is None
             and args.input_coverage_file is not None)
         )
    if (not provided_input_args_ok):
        sys.exit("[ERROR] The input files must be provided either with"
                 " arguments '-a' and '-b' or with argument '-i'.")
    if args.input_coverage_file is None:
        args.input_coverage_file = \
            create_coverage_file(args.input_bed, args.input_bam,
                                 args.no_sorting, args.output_folder,
                                 args.verbose)
    peaks = read_coverage_file(args.input_coverage_file, args.verbose)

    # Determine which peaks should be plotted.
    if args.plot_peak_ids is not None:
        peak_ids_to_plot = np.unique(args.plot_peak_ids)
    elif args.plot_limit < 0:
        peak_ids_to_plot = np.arange(0, len(peaks))
    else:
        peak_ids_to_plot = np.unique(
            np.linspace(start=0, stop=(len(peaks)-1), num=args.plot_limit,
                        dtype=int)
            )
    peaks_to_plot = {p_id: peaks[p_id] for p_id in peak_ids_to_plot}

    if args.verbose:
        print("[NOTE] With the given parameters and input files the plots for"
              " {} peak(s) will be created.".format(len(peaks_to_plot))
              )

    # Switches for enabling or disabling creating specific plots.
    plot_peak_profiles = True
    plot_fft_values = True
    plot_fft_transformations = True

    # Switch for changing some plot styles to improve quality when embedding
    # plots into a Latex document.
    paper_plots = False
    if paper_plots:
        plt.rcParams.update({'font.size': 15})
        args.plot_format = 'pdf'

    if plot_peak_profiles:
        create_profile_plots(peaks_to_plot,
                             os.path.join(args.output_folder, 'plot_profiles'),
                             output_format=args.plot_format,
                             verbose=args.verbose,
                             paper_plots=paper_plots
                             )

    if args.fft_analysis:

        analyze_with_FFT(peaks_to_plot, args.num_padding, args.verbose)

        create_FFT_analysis_plots(
            peaks_to_plot,
            os.path.join(args.output_folder, 'FFT_analysis_plots'),
            output_format=args.plot_format,
            plot_fft_values=plot_fft_values,
            plot_fft_transformations=plot_fft_transformations,
            verbose=args.verbose,
            paper_plots=paper_plots
            )

    deconvolute_with_FFT(peaks=peaks, num_padding=args.num_padding,
                         approach=args.deconv_approach,
                         clip_boundary=args.clip_boundary,
                         distance=args.distance, height=args.height,
                         prominence=args.prominence,
                         disable_frequency_shift=args.disable_frequency_shift,
                         main_freq_filter_value=args.main_freq_filter_value,
                         verbose=args.verbose)

    create_deconv_profile_plots(
        peaks_to_plot,
        os.path.join(args.output_folder, 'plot_profiles_deconv'),
        output_format=args.plot_format,
        verbose=args.verbose,
        paper_plots=paper_plots
        )

    _file_path_overview, _file_path_summits, file_path_all_peaks = \
        create_output_files(peaks, args.output_folder, args.verbose)

    refine_peaks_with_annotations(file_path_all_peaks,
                                  args.gene_file, args.exon_file,
                                  args.output_folder, verbose=args.verbose)

    if args.verbose:
        print("[FINISH]")
