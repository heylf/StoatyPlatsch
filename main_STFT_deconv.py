
import argparse
import os
import sys

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from STFT.processing import deconvolute_peaks_with_STFT
from tools.plotting import create_deconv_profile_plots, create_profile_plots
from tools.postprocessing import (create_output_files,
                                  refine_peaks_with_annotations)
from tools.preprocessing import (add_transcript_annotations,
                                 create_coverage_file, read_coverage_file)


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

    # Define arguments for configuring the deconvolution behavior.
    parser.add_argument(
        "--window",
        default='boxcar',
        help=("The STFT window parameter. See the SciPy documentation for the"
              " STFT for possible values and description. (default: 'boxcar')"
              ),
        )
    parser.add_argument(
        "--detrend",
        default='constant',
        help=("The STFT detrend parameter. See the SciPy documentation for the"
              " STFT for possible values and description."
              " (default: 'constant')"),
        )
    parser.add_argument(
        "--noverlap",
        default=10,
        help=("The STFT noverlap parameter. See the SciPy documentation for"
              " the STFT for description. (default: 10)"),
        type=int
        )
    parser.add_argument(
        "--distance",
        default=10,
        help=("Used as parameter for function 'find_peaks' to estimate the"
              " number of subpeaks that should be defined."
              "  (default: 10)"),
        type=int
        )
    parser.add_argument(
        "--height",
        default=5,
        help=("Used as parameter for function 'find_peaks' to estimate the"
              " number of subpeaks that should be defined."
              "  (default: 5)"),
        type=int
        )
    parser.add_argument(
        "--prominence",
        default=3,
        help=("Used as parameter for function 'find_peaks' to estimate the"
              " number of subpeaks that should be defined."
              "  (default: 3)"),
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
    parser.add_argument(
        "--paper_plots",
        action='store_true',
        help=("Activates some plot style changes and other modifications,"
              " for creating plots that should be embedded in a LaTeX paper.")
        )

    # Optional arguments for annotations
    parser.add_argument(
        "--transcript_file",
        metavar='<gtf>',
        help="Path to the transcript annotation file.")
    parser.add_argument(
        "--exon_peak_boundary_distance",
        default=10,
        help=("Defines the distance between peak and exon boundaries, in"
              " which peaks should still be considered for refinement,"
              " although the peak is not overlapping with the exon boundary."),
        type=int
        )
    parser.add_argument(
        "--gene_file",
        metavar='<bed>',
        help="Path to the gene annotation file.")
    parser.add_argument(
        "--exon_file",
        metavar='<bed>',
        help="Path to the exon boundary file.")

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

    if args.transcript_file:
        transcripts, exons, peaks = \
            add_transcript_annotations(
                peaks=peaks, transcript_file=args.transcript_file,
                output_path=os.path.join(args.output_folder,
                                         '00_transcript_annotations'),
                exon_peak_boundary_distance=args.exon_peak_boundary_distance,
                verbose=args.verbose)

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
    plot_deconv_profiles = True
    calc_eval_params = False

    if calc_eval_params:
        eval_params_peak_lengths = {
            'unmodified': {},
            'to_be_clipped': {},
            'after_clipped': {},
            'final_all': {}
            }
    else:
        eval_params_peak_lengths = None

    mpl.use('Agg')    # To speed up creating the plots.

    if args.paper_plots:
        plt.rcParams.update({'font.size': 15})

    if plot_peak_profiles:
        create_profile_plots(peaks_to_plot,
                             os.path.join(args.output_folder, 'plot_profiles'),
                             output_format=args.plot_format,
                             verbose=args.verbose,
                             paper_plots=args.paper_plots
                             )

    deconvolute_peaks_with_STFT(
        peaks=peaks,
        stft_window=args.window,
        stft_detrend=args.detrend,
        stft_noverlap=args.noverlap,
        find_peaks_distance=args.distance,
        find_peaks_height=args.height,
        find_peaks_prominence=args.prominence,
        verbose=args.verbose,
        eval_params_peak_lengths=eval_params_peak_lengths
        )

    if plot_deconv_profiles:
        create_deconv_profile_plots(
            peaks_to_plot,
            os.path.join(args.output_folder, 'plot_profiles_deconv'),
            output_format=args.plot_format,
            verbose=args.verbose,
            paper_plots=args.paper_plots
            )

    file_path_all_peaks = \
        create_output_files(peaks, args.output_folder, args.verbose)

    refine_peaks_with_annotations(file_path_all_peaks,
                                  args.gene_file, args.exon_file,
                                  args.output_folder, verbose=args.verbose)

    if args.verbose:
        print("[FINISH]")
