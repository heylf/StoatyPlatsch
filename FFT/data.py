
import numpy as np


class Mapping(object):
    """ Contains the data of the mapping between profile and FFT frequencies.

    Attributes
    ----------
    idx_max_profile : int
        The index of the maximum within the profiles maxima.
    max_pos_profile : int
        The position of the profile maximum.
    idx_freq : int
        The index of the mapped frequency.
    max_pos_freq : int
        The position of the maximum of the mapped frequency.
    distance : int
        The distance between the mapped maximum of the profile and the mapped
        maximum of the frequency.
    """

    def __init__(self, idx_max_profile=-1, max_pos_profile=-1,
                 idx_freq=-1, max_pos_freq=-1, distance=np.Inf):
        """ Constructor

        Parameters
        ----------
        See attributes description of the class.
        """
        self.idx_max_profile = idx_max_profile
        self.max_pos_profile = max_pos_profile
        self.idx_freq = idx_freq
        self.max_pos_freq = max_pos_freq
        self.distance = distance


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
    fft : FFT.processing.FFT
        Dummy object, that bundles multiple information, after FFT has applied
        to the peak, e.g.

        - peak.fft.f : numpy.ndarray
              Contains the padded profile, that was used for FFT.
        - peak.fft.fhat : numpy.ndarray
            The result of the FFT.
        - ...

        See processing routines for detailed information.

    References
    ----------
    For the description of the the attributes (except for 'fft') see also:
    .. [1] https://en.wikipedia.org/wiki/BED_(file_format)
    .. [2] https://bedtools.readthedocs.io/en/latest/content/tools/coverage.html    # @IgnorePep8
    .. [3] https://genome.ucsc.edu/FAQ/FAQformat#format1
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

    def __repr__(self):
        """ Returns a representation of the peak.

        Returns
        -------
        result : str
            The representation of the peak.
        """
        return '{}:{}-{}'.format(self.chrom, self.chrom_start, self.chrom_end)
