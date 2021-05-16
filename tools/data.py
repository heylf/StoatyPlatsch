
import numpy as np


class OriginalPeak(object):
    """ Contains the unmodified data of a peak.

    Attributes
    ----------
    chrom : str
        The name of the chromosome.
    chrom_start : int
        The start coordinate on the chromosome (zero-based).
    chrom_end : int
        The end coordinate on the chromosome (zero-based, non-inclusive).
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
    peak_id : int
        The id of the peak, an internal number.
    transcripts : list
        List of associated transcripts.
    exons : list
        List of associated exons.

    References
    ----------
    For the description of some of the attributes see also:
    .. [1] https://en.wikipedia.org/wiki/BED_(file_format)
    .. [2] https://bedtools.readthedocs.io/en/latest/content/tools/coverage.html    # @IgnorePep8
    .. [3] https://genome.ucsc.edu/FAQ/FAQformat#format1
    """

    def __init__(self, chrom, chrom_start, chrom_end, name, score, strand,
                 peak_length, coverage, peak_id):
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
        self.peak_id = peak_id

        self.transcripts = []
        self.exons = []

    def __repr__(self):
        """ Returns a representation of the peak.

        Returns
        -------
        result : str
            The representation of the peak.
        """
        return 'Original Peak ID {} ({}:{}-{})'.format(
            self.peak_id, self.chrom, self.chrom_start+1, self.chrom_end)

    def deconv_peaks_to_bed_entries(self, bed_peak_id):
        """ Returns the deconvoluted peaks as BED entries.

        Parameters
        ----------
        bed_peak_id : int
            The unique id that should be used for the next bed entry.

        Returns
        -------
        result : str
            The BED entries of the deconvoluted peaks.
        bed_peak_id : int
            The unique id that should be used for the next bed entry.
        """
        result = ""
        for subpeak_id, deconv_peak_rel in enumerate(self.deconv_peaks_rel):
            result += "{0}\t{1}\t{2}\t{3}\t{4}\t{5}\t{6}\t{7}\n".format(
                self.chrom,
                self.chrom_start + deconv_peak_rel[0],
                self.chrom_start + deconv_peak_rel[2],
                "{}__Peak_ID_{}{}".format(
                    self.name, self.peak_id,
                    "" if len(self.deconv_peaks_rel) == 1
                    else "__{}".format(subpeak_id)
                    ),
                self.score,
                self.strand,
                bed_peak_id + subpeak_id,
                0
                )
        return result, bed_peak_id + subpeak_id + 1


class UnrefinedPeak(object):
    """ Contains the data of a peak that has not been refined.

    Attributes
    ----------
    peak_id : int
        The id of the peak, an internal number.
    seqname : str
        The seqname of the peak (e.g. chromosome name).
    start : int
        The start coordinate (zero-based).
    end : int
        The end coordinate (zero-based, non-inclusive).
    coverage : numpy.ndarray
        The coverage for each nucleotide.
    orig_peak : OriginalPeak
        The original, unmodified peak as it was read from the input file.
    orig_peaks : list
        List of the original peaks, here only containing attribute 'orig_peak'
        as it is an unrefined peak.
    """

    def __init__(self, peak_id, orig_peak):
        """ Constructor

        Parameters
        ----------
        peak_id : int
            The id of the peak, an internal number.
        orig_peak : OriginalPeak
            The original peak, that should be used for creating the unrefined
            peak.
        """

        self.peak_id = peak_id

        self.seqname = orig_peak.chrom
        self.start = orig_peak.chrom_start
        self.end = orig_peak.chrom_end
        self.coverage = orig_peak.coverage
        self.orig_peak = orig_peak
        self.orig_peaks = [self.orig_peak]

    def __repr__(self):
        """ Returns a representation of the unrefined peak.

        Returns
        -------
        result : str
            The representation of the unrefined peak.
        """
        return 'Unrefined Peak ID {} ({}:{}-{})'.format(
            self.peak_id, self.seqname, self.start+1, self.end)

    @property
    def peak_length(self):
        """ Returns the length of the peak.

        Returns
        -------
        result : int
            The length of the peak .
        """
        return len(self.coverage)

    def deconv_peaks_to_bed_entries(self, bed_peak_id):
        """ Returns the deconvoluted peaks as BED entries.

        Parameters
        ----------
        bed_peak_id : int
            The unique id that should be used for the next bed entry.

        Returns
        -------
        result : str
            The BED entries of the deconvoluted peaks.
        bed_peak_id : int
            The unique id that should be used for the next bed entry.
        """
        result = ""
        for subpeak_id, deconv_peak_rel in enumerate(self.deconv_peaks_rel):
            result += "{0}\t{1}\t{2}\t{3}\t{4}\t{5}\t{6}\t{7}\n".format(
                self.seqname,
                self.start + deconv_peak_rel[0],
                self.start + deconv_peak_rel[2],
                "{}__Peak_ID_{}{}".format(
                    self.orig_peak.name, self.peak_id,
                    "" if len(self.deconv_peaks_rel) == 1
                    else "__{}".format(subpeak_id)
                    ),
                self.orig_peak.score,
                self.orig_peak.strand,
                bed_peak_id + subpeak_id,
                0
                )
        return result, bed_peak_id + subpeak_id + 1


class RefinedPeak(object):
    """ Contains the data of a refined and possible split peak.

    Attributes
    ----------
    peak_id : int
        The id of the peak, an internal number.
    seqnames : list
        List containing the  seqnames of the peak (e.g. chromosome name)
        for each segment.
    starts : list
        List containing the start coordinates (zero-based) for each segment.
    ends : list
        List containing the end coordinates (zero-based, non-inclusive) for
        each segment.
    coverages : list
        List containing the coverage for each segment.
    """

    def _add_segment(self, seqname, start, end, coverage, orig_peak,
                     direction):
        """ Add the peak segment with the given information.

        Parameters
        ----------
        seqname : str
            The seqname of the peak segment.
        start : int
            The start coordinate (zero-based) of the peak segment.
        end : int
            The end coordinate (zero-based, non-inclusive) of the peak segment.
        coverage : numpy.ndarray
            The coverage for each nucleotide of the peak segment.
        orig_peak : OriginalPeak
            The original peak that was used to create this peak segment.
        direction : int
            >=0: Append segment at the end.
            <0: Insert segment at the beginning.
        """
        if direction >= 0:
            self.seqnames.append(seqname)
            self.starts.append(start)
            self.ends.append(end)
            self.coverages.append(coverage)
            self.orig_peaks.append(orig_peak)
        else:
            self.seqnames.insert(0, seqname)
            self.starts.insert(0, start)
            self.ends.insert(0, end)
            self.coverages.insert(0, coverage)
            self.orig_peaks.insert(0, orig_peak)

    def _refine_peak(self, peak, exon_peak_boundary_distance, direction=0):
        """ Handles the refinement for the given peak.

        Parameters
        ----------
        peak : OriginalPeak
            The next peak, that should be used for creating or extending
            the refined peak.
        exon_peak_boundary_distance : int
            Maximum distance between an exon and a peak boundary, for which the
            peak should still be considered at the border of the exon and
            therefore considered for combining with other peaks on neighboring
            exons of the same transcript.
        direction : int (default: 0)
            0: The peak is the root peak.
            <0: The peak should be inserted before the existing peak segments.
            >0: The peak should be inserted after the existing peak segments.
        """

        if peak.refined_peak:
            raise ValueError("Peak is already refined!")

        peak_start = peak.chrom_start
        peak_end = peak.chrom_end
        peak_coverage = peak.coverage
        if (peak.mapped_exon.start
                > peak.chrom_start + 1 - exon_peak_boundary_distance):
            # Peak is crossing the left exon boundary, or is close enough to be
            # considered at the boundary.
            peak_start = peak.mapped_exon.start - 1
            if peak.mapped_exon.start > peak.chrom_start + 1:
                # If the peak is actually crossing the exon boundary, cut it.
                peak_coverage = \
                    peak_coverage[(peak_start-peak.chrom_start):].copy()
            else:
                # Peak is not crossing the exon boundary, but close enough to
                # be considered at the boundary. Fill the gap with zeros.
                peak_coverage = np.pad(peak_coverage,
                                       (peak.chrom_start-peak_start, 0),
                                       'constant', constant_values=0)
        if peak.mapped_exon.end < peak.chrom_end + exon_peak_boundary_distance:
            # Peak is crossing the right exon boundary, or is close enough to
            # be considered at the boundary.
            peak_end = peak.mapped_exon.end
            if peak.mapped_exon.end < peak.chrom_end:
                # If the peak is actually crossing the exon boundary, cut it.
                peak_coverage = \
                    peak_coverage[:-(peak.chrom_end-peak_end)].copy()
            else:
                # Peak is not crossing the exon boundary, but close enough to
                # be considered at the boundary. Fill the gap with zeros.
                peak_coverage = np.pad(peak_coverage,
                                       (0, peak_end-peak.chrom_end),
                                       'constant', constant_values=0)

        peak.refined_peak = self
        self._add_segment(seqname=peak.mapped_exon.seqname, start=peak_start,
                          end=peak_end, coverage=peak_coverage, orig_peak=peak,
                          direction=direction)

        exon_index_on_transcript = \
            peak.mapped_transcript.exons.index(peak.mapped_exon)
        if ((peak_start == peak.mapped_exon.start - 1)
                and (exon_index_on_transcript > 0)):
            # Peak is on left exon boundary and there are more exons before the
            # current exon on the same transcript => Check if on the previous
            # exon is a peak that is on the right boundary and should be
            # therefore merged with the current peak.

            # First ensure that there is no previous peak on the same exon.
            peaks_on_current_exon = sorted(peak.mapped_exon.peaks,
                                           key=lambda p: p.chrom_start)
            peak_index_on_current_exon = \
                peaks_on_current_exon.index(peak) - 1
            can_proceed_with_prev_exon = True
            while peak_index_on_current_exon >= 0:
                prev_peak_on_current_exon = \
                    peaks_on_current_exon[peak_index_on_current_exon]
                if prev_peak_on_current_exon.mapped_exon == peak.mapped_exon:
                    can_proceed_with_prev_exon = False
                    break
                peak_index_on_current_exon -= 1

            # Next, if there is no previous peak on the same exon, we can
            # search for a peak on the previous exon.
            prev_exon = \
                peak.mapped_transcript.exons[exon_index_on_transcript - 1]
            prev_peak = None
            if prev_exon.peaks and can_proceed_with_prev_exon:
                for tmp_prev_peak in sorted(prev_exon.peaks,
                                            key=lambda p: p.chrom_end,
                                            reverse=True
                                            ):
                    if ((tmp_prev_peak.mapped_exon == prev_exon)
                            and (tmp_prev_peak.mapped_transcript
                                 == peak.mapped_transcript)):
                        prev_peak = tmp_prev_peak
                        break

            # Check if the previous peak should be embedded in the current
            # refined peak.
            if (prev_peak and (not prev_peak.refined_peak)
                    and (prev_peak.mapped_exon.end
                         <= prev_peak.chrom_end + exon_peak_boundary_distance
                         )):
                self._refine_peak(
                    peak=prev_peak,
                    exon_peak_boundary_distance=exon_peak_boundary_distance,
                    direction=-1)

        if ((peak_end == peak.mapped_exon.end)
                and ((exon_index_on_transcript + 1)
                     < len(peak.mapped_transcript.exons))):
            # Peak is on right exon boundary and there are more exons after the
            # current exon on the same transcript => Check if on the next
            # exon is a peak that is on the left boundary and should be
            # therefore merged with the current peak.

            # First ensure that there is no following peak on the same exon.
            peaks_on_current_exon = sorted(peak.mapped_exon.peaks,
                                           key=lambda p: p.chrom_end)
            peak_index_on_current_exon = \
                peaks_on_current_exon.index(peak) + 1
            can_proceed_with_next_exon = True
            while peak_index_on_current_exon < len(peaks_on_current_exon):
                next_peak_on_current_exon = \
                    peaks_on_current_exon[peak_index_on_current_exon]
                if next_peak_on_current_exon.mapped_exon == peak.mapped_exon:
                    can_proceed_with_next_exon = False
                    break
                peak_index_on_current_exon += 1

            # Next, if there is no following peak on the same exon, we can
            # search for a peak on the next exon.
            next_exon = \
                peak.mapped_transcript.exons[exon_index_on_transcript + 1]
            next_peak = None
            if next_exon.peaks and can_proceed_with_next_exon:
                for tmp_next_peak in sorted(next_exon.peaks,
                                            key=lambda p: p.chrom_start):
                    if ((tmp_next_peak.mapped_exon == next_exon)
                            and (tmp_next_peak.mapped_transcript
                                 == peak.mapped_transcript)):
                        next_peak = tmp_next_peak
                        break

            # Check if the next peak should be embedded in the current refined
            # peak.
            if (next_peak and (not next_peak.refined_peak)
                and (next_peak.mapped_exon.start
                     >= next_peak.chrom_start + 1 - exon_peak_boundary_distance
                     )):
                self._refine_peak(
                    peak=next_peak,
                    exon_peak_boundary_distance=exon_peak_boundary_distance,
                    direction=1)

    def __init__(self, peak_id, orig_peak, exon_peak_boundary_distance):
        """ Constructor

        Parameters
        ----------
        peak_id : int
            The id of the peak, an internal number.
        orig_peak : OriginalPeak
            The original peak, that should used for creating a refined peak.
        exon_peak_boundary_distance : int
            Maximum distance between an exon and a peak boundary, for which the
            peak should still be considered at the border of the exon and
            therefore considered for combining with other peaks on neighboring
            exons of the same transcript.
        """

        self.peak_id = peak_id

        self.seqnames = []
        self.starts = []
        self.ends = []
        self.coverages = []
        self.orig_peaks = []

        self._refine_peak(orig_peak, exon_peak_boundary_distance)

    def __repr__(self):
        """ Returns a representation of the refined peak.

        Returns
        -------
        result : str
            The representation of the unrefined peak.
        """
        inner_part = ''
        for seqname, start, end in zip(self.seqnames, self.starts, self.ends):
            inner_part += '{}:{}-{} | '.format(seqname, start+1, end)
        inner_part = inner_part[:-3]
        result = 'Refined Peak ID {} ({})'.format(self.peak_id, inner_part)
        return result

    @property
    def peak_length(self):
        """ Returns the length of the whole refined peak.

        Returns
        -------
        result : int
            The length of the peak after separated segments have been
            concatenated.
        """
        return len(self.coverage)

    @property
    def coverage(self):
        """ Returns the coverage of the whole refined peak.

        Returns
        -------
        result : numpy.ndarray
            The coverage of the peak. Separated segments are concatenated.
        """
        return np.concatenate(self.coverages)

    def deconv_peaks_to_bed_entries(self, bed_peak_id):
        """ Returns the deconvoluted peaks as BED entries.

        Parameters
        ----------
        bed_peak_id : int
            The unique id that should be used for the next bed entry.

        Returns
        -------
        result : str
            The BED entries of the deconvoluted peaks.
        bed_peak_id : int
            The unique id that should be used for the next bed entry.
        """
        result = ""
        for subpeak_id, deconv_peak_rel in enumerate(self.deconv_peaks_rel):
            ranges = np.array(self.ends) - np.array(self.starts)
            peak_segment_end_pos = np.cumsum(ranges)
            peak_segment_index_start = \
                (peak_segment_end_pos >= deconv_peak_rel[0]).argmax()
            peak_segment_index_end = \
                (peak_segment_end_pos >= deconv_peak_rel[2]).argmax()
            for peak_segment_index in range(peak_segment_index_start,
                                            peak_segment_index_end + 1):

                if peak_segment_index > peak_segment_index_start:
                    start_pos_in_cur_segment = 0
                else:
                    start_pos_in_cur_segment = deconv_peak_rel[0]
                    if peak_segment_index > 0:
                        start_pos_in_cur_segment -= \
                            peak_segment_end_pos[peak_segment_index-1]

                if peak_segment_index < peak_segment_index_end:
                    end_pos_in_cur_segment = ranges[peak_segment_index]
                else:
                    end_pos_in_cur_segment = deconv_peak_rel[2]
                    if peak_segment_index > 0:
                        end_pos_in_cur_segment -= \
                            peak_segment_end_pos[peak_segment_index-1]

                result += "{0}\t{1}\t{2}\t{3}\t{4}\t{5}\t{6}\t{7}\n".format(
                    self.seqnames[peak_segment_index],
                    self.starts[peak_segment_index] + start_pos_in_cur_segment,
                    self.starts[peak_segment_index] + end_pos_in_cur_segment,
                    "{}__Peak_ID_{}{}{}".format(
                        self.orig_peaks[peak_segment_index].name, self.peak_id,
                        "" if len(self.deconv_peaks_rel) == 1
                        else "__{}".format(subpeak_id),
                        "" if (peak_segment_index_start
                               == peak_segment_index_end)
                        else "__Gr_{}".format(peak_segment_index
                                              - peak_segment_index_start)
                        ),
                    self.orig_peaks[peak_segment_index].score,
                    self.orig_peaks[peak_segment_index].strand,
                    bed_peak_id + subpeak_id,
                    peak_segment_index - peak_segment_index_start
                    )

        return result, bed_peak_id + subpeak_id + 1


class GtfFeature(object):
    """ Contains the data of a feature from a GTF file.

    Attributes
    ----------
    feature_id : str
        An unique string for identifying the feature.
    seqname : str
        The name of the chromosome or scaffold.
    feature : str
        The type of the feature, e. g. 'transcript' or 'exon'.
    start : int
        The start coordinate of the feature (one-based).
    end : int
        The end coordinate of the feature (one-based, inclusive).
    score : str
        The score of the feature.
    strand : str
        The strand orientation, "+" for forward, "-" for reverse.

    peaks : list
        List of peaks associated with this feature.

    References
    ----------
    For the description of some of the attributes see also:
    .. [1] https://www.ensembl.org/info/website/upload/gff.html
    """

    def __init__(self, feature_id, seqname, feature, start, end, score,
                 strand):
        """ Constructor

        Parameters
        ----------
        See attributes description of the class.
        """
        self.feature_id = feature_id
        self.seqname = seqname
        self.feature = feature
        self.start = start
        self.end = end
        self.score = score
        self.strand = strand

        self.peaks = []

    def __repr__(self):
        """ Returns a representation of the feature.

        Returns
        -------
        result : str
            The representation of the feature.
        """
        return '{} ({}:{}-{})'.format(self.feature_id, self.seqname,
                                      self.start, self.end)

    def __len__(self):
        """ Returns the length of the feature.

        Returns
        -------
        result : int
            The length of the feature (i.e. the number of spanned nucleotides).
        """
        return self.end - self.start + 1


class Transcript(GtfFeature):
    """ Contains the data of a transcript from a GTF file.

    Attributes
    ----------
    exons : list
        A list of the exons associated with this transcript. The order of
        the list defines the concatenation order of the exons.
    """

    def __init__(self, **kwargs):
        """ Constructor

        Parameters
        ----------
        See attributes description of parent class 'GtfFeature'.

        Raises
        ------
        ValueError
            When parameter 'feature' has the wrong value.
        """
        if kwargs['feature'] != 'transcript':
            raise ValueError("Wrong value for parameter 'feature'. For class "
                             "'Transcript' the value must be 'transcript.'")
        super().__init__(**kwargs)
        self.exons = []


class Exon(GtfFeature):
    """ Contains the data of an exon from a GTF file.

    Attributes
    ----------
    transcripts : list
        A list of transcripts that are associated with this exon.
    """

    def __init__(self, **kwargs):
        """ Constructor

        Parameters
        ----------
        See attributes description of parent class 'GtfFeature'.

        Raises
        ------
        ValueError
            When parameter 'feature' has the wrong value.
        """
        if kwargs['feature'] != 'exon':
            raise ValueError("Wrong value for parameter 'feature'. For class "
                             "'Exon' the value must be 'exon.'")
        super().__init__(**kwargs)
        self.transcripts = []

    def __eq__(self, other):
        """ Compares two Exon objects.

        Parameters
        ----------
        other : Exon
            The Exon object self should be compared to.

        Returns
        -------
        result : bool
            True, if the two Exon objects are equal, otherwise False.
        """
        result = ((self.feature_id == other.feature_id)
                  and (self.seqname == other.seqname)
                  and (self.feature == other.feature)
                  and (self.start == other.start)
                  and (self.end == other.end)
                  and (self.score == other.score)
                  and (self.strand == other.strand))
        return result
