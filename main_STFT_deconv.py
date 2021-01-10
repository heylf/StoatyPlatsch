
import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

from FFT.plotting import create_profile_plots
from FFT.preprocessing import create_coverage_file, read_coverage_file


def create_argument_parser():
    """ Creates the argument parser for parsing the program arguments.

    Returns
    -------
    parser : ArgumentParser
        The argument parser for parsing the program arguments.
    """
    parser = argparse.ArgumentParser(
        description=("Deconvolutes the given peak profiles with a STFT"
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

    return parser


if __name__ == '__main__':

    parser = create_argument_parser()
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

    if args.verbose:
        print("[FINISH]")
