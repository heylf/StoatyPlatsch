
import argparse

from evaluation.plotting import PeakViewer


def create_argument_parser():
    """ Creates the argument parser for parsing the program arguments.

    Returns
    -------
    parser : ArgumentParser
        The argument parser for parsing the program arguments.
    """
    parser = argparse.ArgumentParser(
        description=("Starts an interactive peak viewer for evaluating"
                     " deconvolution results. The results are calculated"
                     " using default values."),
        usage="%(prog)s [-h] [options] -i <tsv>"
        )
    parser.add_argument(
        "-i", "--input_coverage_file",
        help="Path to the coverage file in tsv format.",
        metavar="<tsv>",
        required=True
        )
    parser.add_argument(
        "-o", "--output_folder",
        help="Write intermediate results to this path.",
        metavar='path/to/output_folder',
        required=True
        )
    parser.add_argument(
        "-t", "--transcript_file",
        help="Path to the transcript annotation file.",
        metavar="<gtf>",
        required=True
        )
    parser.add_argument(
        "-m", "--motifs",
        help="Defines the motifs that should be highlighted if found.",
        metavar="MOTIF",
        nargs="+",
        type=str
        )
    parser.add_argument(
        "-f", "--fasta_files",
        help="Paths to fasta files for extracting and adding genomic data.",
        metavar="<fasta>",
        nargs="+",
        type=str
        )
    parser.add_argument(
        "--verbose",
        action='store_true',
        help="Print additional information to console."
        )
    parser.add_argument(
        "--paper_plots",
        action='store_true',
        help=("Activates some plot style changes and other modifications."
              " For creating plots that should be embedded in a paper.")
        )
    return parser


if __name__ == '__main__':
    parser = create_argument_parser()
    args = parser.parse_args()

    PeakViewer(input_coverage_file=args.input_coverage_file,
               output_folder=args.output_folder,
               transcript_file=args.transcript_file,
               motifs=args.motifs, fasta_files=args.fasta_files,
               verbose=args.verbose, paper_plots=args.paper_plots)
