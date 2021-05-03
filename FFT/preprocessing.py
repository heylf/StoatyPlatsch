
import numpy as np

from .data import Peak


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

    with open(input_coverage_file, "r") as coverage_file:
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
                # Peak section is continued, ensure the data is consistent.
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

    return peaks
