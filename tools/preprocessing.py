
from collections import OrderedDict
import gzip
import os
import pathlib
import subprocess as sb
import sys

import numpy as np


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
    with open(input_bed, "r") as bed_file:
        first_line = bed_file.readline()
        if (len(first_line.strip("\n").split("\t")) < 6):
            sys.exit("[ERROR] Peak file has to be in bed6 format!")

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
        cmd_sort = "sort -V -k1,1 -k2,2n -k3,3n  {} > {}".format(
            input_bed, input_bed_sorted
            )
        if verbose:
            print("[NOTE] Create sorted peak file. Cmd: {}".format(cmd_sort))
        sb.call(cmd_sort, shell=True)

        cmd_coverage = \
            "bedtools coverage -sorted -a {} -b {} -g {} -d -s > {}".format(
                input_bed_sorted, input_bam, input_bed_sorted,
                coverage_file_path
                )

    if verbose:
        print("[NOTE] Create coverage file. Cmd: {}".format(cmd_coverage))
    sb.call(cmd_coverage, shell=True)

    return coverage_file_path
