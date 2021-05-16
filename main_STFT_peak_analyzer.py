
import argparse
import os
import sys

from STFT.plotting import PeakAnalyzer
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
        description=("Starts an interactive peak analyzer for evaluating"
                     " the effect of different STFT parameters."),
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

    # Optional arguments for plotting
    parser.add_argument(
        "--create_additional_plots",
        action='store_true',
        help="Create additional plots."
        )

    return parser


if __name__ == '__main__':

    parser = create_argument_parser()
    args = parser.parse_args()

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

    pa = PeakAnalyzer(peaks=peaks, verbose=args.verbose,
                      create_additional_plots=args.create_additional_plots)
