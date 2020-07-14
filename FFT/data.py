
class Peak(object):
    """ Contains the data of a peak.

    Attributes
    ----------
    chrom : str
        The name of the chromosome.
    chrom_start : int
        The start coordinate on the chromosome (zero-based).
    chrom_end : int
        The end coordinate on the chromosome  (zero-based, non-inclusive).
    name : str
        The name of the line in the BED file.
    score : int
        The score, a value between 0 and 1000.
    strand : str
        The DNA strand orientation, "+" for positive, "-" for negative.
    peak_length : int
        The length of the peak, i.e. the number of nucleotides.
    coverage : numpy.ndarray
        The coverage for each nucleotide.
    peak_number : int
        The number of the peak.

    References
    ----------
    For the description of the the attributes see also
    .. [1] https://en.wikipedia.org/wiki/BED_(file_format)
    .. [2] https://bedtools.readthedocs.io/en/latest/content/tools/coverage.html    # @IgnorePep8
    """

    def __init__(self, chrom, chrom_start, chrom_end, name, score, strand,
                 peak_length, coverage, peak_number):
        """ Constructor

        Parameters
        ----------
        See attributes description of the class.
        """
        self.chrom = chrom
        self.chrom_start = chrom_start
        self.chrom_end = chrom_end
        self.name = name
        self.score = score
        self.strand = strand
        self.peak_length = peak_length
        self.coverage = coverage
        self.peak_number = peak_number
