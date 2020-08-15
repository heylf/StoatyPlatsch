
import os
import pathlib
import subprocess as sb
import sys

import numpy as np

from .data import Peak


def create_coverage_file(input_bed, input_bam, no_sorting=False,
                         output_path=os.getcwd(), verbose=False):
    """ Creates the coverage file for the given peak and read file.

    Creates the coverage file for the peaks of the input files. The default
    behavior is sorting the peak file first and then calculating the coverage
    by using 'bedtools coverage' with the sorted argument, which reduces the
    required memory significantly. This behavior can be changed by setting the
    parameter 'no_sorting' to True.

    Parameters
    ----------
    input_bed : str
        The file path of the peak file in bed6 format, containing the peaks for
        which the coverage should be calculated.
    input_bam : str
        The file path of the read file used for the peak calling in bed or bam
        format.
    no_sorting : bool (default: False)
        When set to True the coverage is calculated without using the sorted
        parameter of 'bedtools coverage'.
    output_path : str (default: path of the current working directory)
        The folder path where the coverage file should be saved. Non existing
        folders will be created.
    verbose : bool (default: False)
        Print information to console when set to True.

    Returns
    -------
    coverage_file_path : str
        The file path of the created coverage file.
    """

    # Check if peak file is in bed6 format
    bed_file = open(input_bed, "r")
    first_line = bed_file.readline()
    if (len(first_line.strip("\n").split("\t")) < 6):
        sys.exit("[ERROR] Peak file has to be in bed6 format!")
    bed_file.close()

    os.makedirs(output_path, exist_ok=True)

    # Generate coverage file with bedtools. If sorting is used, create
    # intermediate sorted peak file first.
    coverage_file_path = os.path.join(
        output_path,
        "{}__coverage.tsv".format(pathlib.Path(input_bam).stem)
        )
    if no_sorting:
        cmd_coverage = "bedtools coverage -a {} -b {} -d -s > {}".format(
            input_bed, input_bam, coverage_file_path
            )
    else:
        input_bed_sorted = os.path.join(
            output_path,
            "{}__sorted.bed".format(pathlib.Path(input_bed).stem)
            )
        cmd_sort = "sort -V -k1,1 -k2,2n {} > {}".format(input_bed,
                                                         input_bed_sorted)
        if verbose:
            print("[NOTE] Create sorted peak file. Cmd: {}".format(cmd_sort))
        sb.Popen(cmd_sort, shell=True).wait()

        cmd_coverage = \
            "bedtools coverage -sorted -a {} -b {} -g {} -d -s > {}".format(
                input_bed_sorted, input_bam, input_bed_sorted,
                coverage_file_path
                )

    if verbose:
        print("[NOTE] Create coverage file. Cmd: {}".format(cmd_coverage))
    sb.Popen(cmd_coverage, shell=True).wait()

    return coverage_file_path


def read_coverage_file(input_coverage_file, verbose=False):
    """ Reads and processes the given coverage file.

    Parameters
    ----------
    input_coverage_file : str
        The file path of the coverage file that should be read. The file should
        contain the following, tab-separated values (see also description of
        class 'Peak'):

        - chromosome name
        - start coordinate
        - end coordinate
        - name
        - score
        - strand
        - [...]
        - nucleotide number (one-based)
        - coverage

        Additional columns between columns 'strand' and 'nucleotide' are
        ignored.
    verbose : bool (default: False)
        Print information to console when set to True.


    Returns
    -------
    peaks : dict
        Dictionary containing the peak data.
    """

    if verbose:
        print("[NOTE] Read the coverage file from '{}'."
              .format(input_coverage_file))

    peaks = {}
    peak_index = -1
    peak = None
    previous_nt = -1

    coverage_file = open(input_coverage_file, "r")
    for line in coverage_file:
        data = line.strip("\n").split("\t")

        chrom = data[0]
        chrom_start = int(data[1])
        chrom_end = int(data[2])
        name = data[3]
        score = data[4]
        strand = data[5]
        # Some additional values which might be stored are ignored here,
        # continuing with nucleotide and coverage.
        nt = int(data[-2])  # Nucleotide of the peak.
        coverage = int(data[-1])  # Coverage at that nt.

        peak_length = chrom_end - chrom_start

        if nt == 1:
            # New peak section was found.
            peak_index += 1
            peak = Peak(chrom, chrom_start, chrom_end, name, score, strand,
                        peak_length, np.empty(peak_length, dtype=int),
                        (peak_index + 1))
            peaks[peak_index] = peak
            previous_nt = 1
        else:
            # Peak section is continued, ensure that the data is consistent.
            assert chrom == peak.chrom
            assert chrom_start == peak.chrom_start
            assert chrom_end == peak.chrom_end
            assert name == peak.name
            assert score == peak.score
            assert strand == peak.strand
            assert peak_length == peak.peak_length
            assert nt == (previous_nt + 1)

            previous_nt = nt

        peak.coverage[nt-1] = coverage
    coverage_file.close()
    return peaks
