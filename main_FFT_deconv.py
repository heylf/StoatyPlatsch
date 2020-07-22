
import argparse
import os

import numpy as np

# from FFT.development.development import analyze_FFT, create_FFT_plots
from FFT.preprocessing import read_coverage_file
from FFT.plotting import create_profile_plots


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-i", "--input_coverage_file",
        required=True,
        help="Path to the coverage file in tsv format",
        metavar="<tsv>"
        )
    parser.add_argument(
        "-o", "--output_folder",
        default=os.getcwd(),
        help="Write results to this path (default: current working directory)",
        metavar='path/to/output_folder'
        )
    parser.add_argument("--verbose", action='store_true',
                        help="Print information to console.")

    # Define arguments for configuring the plotting behavior.
    plot_group = parser.add_mutually_exclusive_group()
    plot_group.add_argument(
        "--plot_limit",
        default=10,
        help=("Defines the maximum number of peaks for which plots are "
              "created (default: 10). If the number of peaks is larger than "
              "this value, the peaks for plotting are chosen equally "
              "distributed. For values < 0 plots are created for all peaks. "
              "Cannot be used with argument 'plot_peak_ids'."),
        type=int
        )
    plot_group.add_argument(
        "--plot_peak_ids",
        help=("Defines the ids of the peaks for which the plots should be "
              "created. If no list is provided the ids are calculated "
              "using the argument 'plot_limit' and the number of peaks "
              "provided by the input file."),
        metavar="PEAK_ID",
        nargs="+",
        type=int
        )

    args = parser.parse_args()

    if args.verbose:
        print("[START]")

    peaks = read_coverage_file(args.input_coverage_file)

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
        print("[NOTE] With the given parameters and input files the plots for "
              "{} peaks will be created.".format(len(peaks_to_plot))
              )

    create_profile_plots(peaks_to_plot,
                         os.path.join(args.output_folder, 'plot_profiles'))

    # create_FFT_plots(peaks, os.path.join(args.output_folder, 'FFT_profiles'))

    # analyze_FFT_old(peaks, os.path.join(args.output_folder, 'FFT_tests'))

    # analyze_FFT(peaks, os.path.join(args.output_folder, 'FFT_results'))

    if args.verbose:
        print("[FINISH]")
