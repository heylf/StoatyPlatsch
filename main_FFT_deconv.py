
import argparse
import os

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

    args = parser.parse_args()

    peaks = read_coverage_file(args.input_coverage_file)

    peaks_to_plot = {p: peaks[p] for p in sorted(peaks.keys())[:100]}
    create_profile_plots(peaks_to_plot,
                         os.path.join(args.output_folder, 'plot_profiles'))

    # create_FFT_plots(peaks, os.path.join(args.output_folder, 'FFT_profiles'))

    # analyze_FFT(peaks, os.path.join(args.output_folder, 'FFT_tests'))
