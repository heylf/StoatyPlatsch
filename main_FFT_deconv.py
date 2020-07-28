
import argparse
import os
import sys

import numpy as np

# from FFT.development.development import (analyze_FFT_old, analyze_FFT_old2,
#                                          create_FFT_plots_old)
from FFT.plotting import create_FFT_plots, create_profile_plots
from FFT.postprocessing import (create_output_files,
                                refine_peaks_with_annotations)
from FFT.preprocessing import create_coverage_file, read_coverage_file
from FFT.processing import analyze_with_FFT, deconvolute_with_FFT


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
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
              ))

    group_required.add_argument(
        "-i", "--input_coverage_file",
        help="Path to the coverage file in tsv format.",
        metavar="<tsv>"
        )

    parser.add_argument(
        "-o", "--output_folder",
        default=os.getcwd(),
        help=("Write results to this path"
              " (default: current working directory)."),
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

    # Define arguments for configuring the deconvolution behavior.
    parser.add_argument(
        "--num_padding",
        default=[20, 20],
        nargs=2,
        help=("Number of zeros that should be padded to the left and right"
              " side of the peak profiles, respectively. (default: [20, 20])"),
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
                 " arguments '-a' and 'b' or with argument '-i'.")
    if args.input_coverage_file is None:
        args.input_coverage_file = \
            create_coverage_file(args.input_bed, args.input_bam,
                                 args.no_sorting, args.output_folder,
                                 args.verbose)
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
        print("[NOTE] With the given parameters and input files the plots for"
              " {} peaks will be created.".format(len(peaks_to_plot))
              )

    create_profile_plots(peaks_to_plot,
                         os.path.join(args.output_folder, 'plot_profiles'))

    analyze_with_FFT(peaks, args.num_padding, args.verbose)

    deconvolute_with_FFT(peaks, args.num_padding, args.verbose)

    # create_FFT_plots_old(peaks_to_plot,
    #                      os.path.join(args.output_folder, 'FFT_profiles'))

    # analyze_FFT_old(peaks_to_plot,
    #                 os.path.join(args.output_folder, 'FFT_tests'))

    create_FFT_plots(peaks_to_plot,
                     os.path.join(args.output_folder, 'FFT_plots'))

    # print("Old FFT")
    # analyze_FFT_old2(peaks_to_plot,
    #                  os.path.join(args.output_folder, 'FFT_results'))

    # create_FFT_plots_old(peaks_to_plot,
    #                      os.path.join(args.output_folder, 'FFT_plots'))

    _file_path_overview, _file_path_summits, file_path_all_peaks = \
        create_output_files(peaks, args.output_folder, args.verbose)

    refine_peaks_with_annotations(file_path_all_peaks,
                                  args.gene_file, args.exon_file,
                                  args.output_folder, args.verbose)

    if args.verbose:
        print("[FINISH]")
