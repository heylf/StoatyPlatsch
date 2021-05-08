
from collections import OrderedDict
import gzip
import os
import pathlib
import subprocess as sb
import sys

import numpy as np

from .data import Exon, OriginalPeak, RefinedPeak, Transcript, UnrefinedPeak


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


def read_coverage_file(input_coverage_file, verbose=False):
    """ Reads and processes the given coverage file.

    Parameters
    ----------
    input_coverage_file : str
        The file path of the coverage file that should be read. The file should
        contain the following, tab-separated values (see also description of
        class 'OriginalPeak'):

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
    peaks : OrderedDict
        Dictionary containing the peak data.
    """

    if verbose:
        print("[NOTE] Read the coverage file from '{}'."
              .format(input_coverage_file))

    peaks = OrderedDict()
    peak_id = -1
    peak = None
    previous_nt = -1

    with open(input_coverage_file, "r") as coverage_file:
        for line_number, line in enumerate(coverage_file):

            error_message = 'Error in line {}.'.format(line_number+1)

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

                if previous_nt != -1:
                    # Before proceeding with the next peak, ensure that enough
                    # coverage values were provided for the previous peak.
                    assert previous_nt == peaks[peak_id].peak_length, \
                        error_message
                peak_id += 1
                peak = OriginalPeak(chrom, chrom_start, chrom_end, name, score,
                                    strand, peak_length,
                                    np.empty(peak_length, dtype=int),
                                    peak_id)
                peaks[peak_id] = peak
                previous_nt = 1
            else:
                # Peak section is continued, ensure the data is consistent.
                assert chrom == peak.chrom, error_message
                assert chrom_start == peak.chrom_start, error_message
                assert chrom_end == peak.chrom_end, error_message
                assert name == peak.name, error_message
                assert score == peak.score, error_message
                assert strand == peak.strand, error_message
                assert peak_length == peak.peak_length, error_message
                assert nt == (previous_nt + 1), error_message

                previous_nt = nt

            peak.coverage[nt-1] = coverage

    # Final check for last peak entry, too ensure that enough data points
    # were provided.
    assert previous_nt == peaks[peak_id].peak_length, error_message

    return peaks


def read_transcript_annotations(transcript_file, verbose=False):
    """ Reads and processes the given transcript annotation file.

    Parameters
    ----------
    transcript_file : str
        The file path of the transcript annotation file in GTF format that
        should be read. Only the information regarding transcripts and exons
        are extracted.
    verbose : bool (default: False)
        Print information to console when set to True.

    Returns
    -------
    transcripts : OrderedDict
        Dictionary containing the transcript data.
    exons : OrderedDict
        Dictionary containing the exon data.
    """
    if verbose:
        print("[NOTE] ... Read the transcript annotations from '{}'."
              .format(transcript_file))

    transcripts = OrderedDict()
    exons = OrderedDict()

    transcript_exon_mapping = {}    # Temporary mapping dictionary, so that
    # data can be read first and then mapped afterwards using this dictionary.
    # Necessary, as in a not well formatted GTF file, transcripts and exons
    # could have a random order. Here, only the transcript_id as string is
    # stored, not the final transcript object.

    with gzip.open(transcript_file) as fh_gtf:
        for linenumber, line in enumerate(fh_gtf):
            line = line.decode()
            if line.startswith('#'):
                # Skip comment lines.
                continue
            data = line.strip("\n").split("\t")

            seqname = data[0]
            feature = data[2]
            if feature not in ['transcript', 'exon']:
                # Handle only transcript and exon data.
                continue
            if seqname.isdigit() or seqname in ['X', 'Y']:
                seqname = 'chr' + seqname
            elif not seqname.startswith('chr'):
                continue
            start = int(data[3])
            end = int(data[4])
            score = data[5]
            strand = data[6]
            attribute = data[8]

            transcript_id = None
            exon_id = None
            exon_number = None

            attributes = attribute.strip(";").split(";")

            for attribute in attributes:
                attr_name, attr_val = attribute.split(maxsplit=1)
                attr_val = attr_val[1:-1]
                if attr_name == 'transcript_id':
                    transcript_id = attr_val
                elif attr_name == 'exon_id':
                    exon_id = attr_val
                elif attr_name == 'exon_number':
                    exon_number = int(attr_val)

            if transcript_id is None:
                raise ValueError(
                    "No transcript_id found for feature in line {}!"
                    .format(linenumber + 1)
                    )
            if (feature == 'exon') and (exon_id is None):
                raise ValueError(
                    "No exon_id found for exon feature in line {}!"
                    .format(linenumber + 1)
                    )
            if feature == 'transcript':
                if transcript_id in transcripts:
                    raise ValueError(
                        "Multiple definitions for transcript '{}'. Either"
                        " there is an error in the input file or the input"
                        " data is not supported."
                        .format(transcript_id)
                        )
                transcript = Transcript(feature_id=transcript_id,
                                        seqname=seqname, feature=feature,
                                        start=start, end=end, score=score,
                                        strand=strand)
                transcripts[transcript_id] = transcript
            elif feature == 'exon':
                exon = Exon(feature_id=exon_id, seqname=seqname,
                            feature=feature, start=start, end=end, score=score,
                            strand=strand)
                if exon_id in exons:
                    if exon != exons[exon_id]:
                        raise ValueError(
                            "Multiple definitions for exon '{}'. Either"
                            " there is an error in the input file or the input"
                            " data is not supported."
                            .format(exon_id)
                            )
                    exon = exons[exon_id]
                else:
                    exons[exon_id] = exon

                if transcript_id not in transcript_exon_mapping:
                    transcript_exon_mapping[transcript_id] = {}
                if exon_number is None:
                    exon_number = \
                        len(transcript_exon_mapping[transcript_id]) + 1
                transcript_exon_mapping[transcript_id][exon_number] = exon

    for transcript in transcripts.values():
        for exon_number in sorted(
                transcript_exon_mapping[transcript.feature_id]
                ):
            exon = transcript_exon_mapping[transcript.feature_id][exon_number]
            exon.transcripts.append(transcript)
            transcript.exons.append(exon)

    return transcripts, exons


def peaks_to_bed(peaks, output_file_path):
    """ Save the given peaks in BED format.

    Parameters
    ----------
    peaks : OrderedDict
        Dictionary containing the peak data that should be saved.
    output_file_path : str
        The file path that is used for saving the peak data.
    """
    with open(output_file_path, 'w') as output_fh:
        for peak in peaks.values():
            output_fh.write(
                "{0}\t{1}\t{2}\t{3}\t{4}\t{5}\n"
                .format(peak.chrom, peak.chrom_start, peak.chrom_end,
                        "{}__peak_id_{}".format(peak.name, peak.peak_id),
                        peak.score, peak.strand)
                )


def gtf_features_to_bed(features, output_file_path):
    """ Save the given GTF features in BED format.

    Parameters
    ----------
    features : OrderedDict
        Dictionary containing the feature data that should be saved.
    output_file_path : str
        The file path that is used for saving the feature data.
    """
    with open(output_file_path, 'w') as output_fh:
        for feature in features.values():
            output_fh.write(
                "{0}\t{1}\t{2}\t{3}\t{4}\t{5}\n"
                .format(feature.seqname,
                        feature.start - 1,    # Start for feature is one-based.
                        feature.end,    # End for feature is also one-based,
                        # but inclusive in contrast to end value for bed, which
                        # is zero-based but non-inclusive.
                        feature.feature_id, feature.score, feature.strand)
                )


def add_transcript_annotations(peaks, transcript_file, output_path=os.getcwd(),
                               exon_peak_boundary_distance=10,
                               verbose=False):
    """ Embeds transcript information to the given peaks and refine them.

    Parameters
    ----------
    peaks : OrderedDict
        Dictionary containing the peak data that should be refined with
        transcript information.
    transcript_file : str
        The file path of the transcript annotation file in GTF format that
        should be read. Only the information regarding transcripts and exons
        are extracted.
    output_path : str (default: path of the current working directory)
        The folder path where generated files should be saved. Non existing
        folders will be created.
    exon_peak_boundary_distance : int
        Maximum distance between an exon and a peak boundary, for which the
        peak should still be considered at the border of the exon and therefore
        considered for combining with other peaks on neighboring exons of the
        same transcript.
    verbose : bool (default: False)
        Print information to console when set to True.

    Returns
    -------
    transcripts : OrderedDict
        Dictionary containing the transcript data.
    exons : OrderedDict
        Dictionary containing the exon data.
    refined_peaks : OrderedDict
        Dictionary containing the refined peaks.
    """

    if verbose:
        print("[NOTE] Process transcript annotations ...")
    transcripts, exons = read_transcript_annotations(transcript_file, verbose)

    os.makedirs(output_path, exist_ok=True)

    file_path_peaks_bed = os.path.join(output_path, '01_peaks.bed')
    peaks_to_bed(peaks=peaks, output_file_path=file_path_peaks_bed)

    file_path_transcipts_bed = os.path.join(output_path,
                                            '02_transcript_annotations.bed')
    gtf_features_to_bed(features=transcripts,
                        output_file_path=file_path_transcipts_bed)

    file_path_intersection_peaks_transcripts = \
        os.path.join(output_path,
                     "03_intersection_peaks_transcripts.bed")
    cmd = "bedtools intersect -s -wb -a {} -b {} > {}".format(
               file_path_peaks_bed, file_path_transcipts_bed,
               file_path_intersection_peaks_transcripts
           )
    if verbose:
        print("[NOTE] ... Create intersection peaks <-> transcripts. Cmd: {}"
              .format(cmd))
    sb.call(cmd, shell=True)

    with open(file_path_intersection_peaks_transcripts,
              "r") as fh_intersection:
        for line in fh_intersection:
            data = line.strip("\n").split("\t")
            name = data[3]
            p_id = int(name.split('__peak_id_')[-1])
            transcipt_id = data[9]
            peaks[p_id].transcripts.append(transcripts[transcipt_id])
            transcripts[transcipt_id].peaks.append(peaks[p_id])

    file_path_exons_bed = os.path.join(output_path, '04_exons_annotations.bed')
    gtf_features_to_bed(features=exons, output_file_path=file_path_exons_bed)

    file_path_intersection_peaks_exons = \
        os.path.join(output_path, "05_intersection_peaks_exons.bed")
    cmd = "bedtools intersect -s -wb -f 0.9 -a {} -b {} > {}".format(
               file_path_peaks_bed, file_path_exons_bed,
               file_path_intersection_peaks_exons
           )
    if verbose:
        print("[NOTE] ... Create intersection peaks <-> exons. Cmd: {}"
              .format(cmd))
    sb.call(cmd, shell=True)

    with open(file_path_intersection_peaks_exons,
              "r") as fh_intersection:
        for line in fh_intersection:
            data = line.strip("\n").split("\t")
            name = data[3]
            p_id = int(name.split('__peak_id_')[-1])
            exon_id = data[9]
            peaks[p_id].exons.append(exons[exon_id])
            exons[exon_id].peaks.append(peaks[p_id])

    if verbose:
        print("[NOTE] ... Refine peaks with transcript annotations.")

    for peak in peaks.values():

        peak.refined_peak = None

        # Map each peak to a single exon of a single transcript. Use the
        # transcript that has overall the most peaks assigned to it. For
        # calculating the number of peaks assigned to a transcript, only
        # peaks that also are assigned to exons on that transcript are
        # considered.
        peak.mapped_transcript = None
        peak.mapped_exon = None

        if not peak.exons:
            continue

        max_number_of_peaks = 0
        for exon in peak.exons:    # Iterate over peak.exons to ensure that
            # only transcripts are considered, where the peak is also on an
            # exon.
            for transcript in exon.transcripts:
                number_of_peaks = 0
                for transcript_exon in transcript.exons:
                    number_of_peaks += len(transcript_exon.peaks)
                if number_of_peaks > max_number_of_peaks:
                    max_number_of_peaks = number_of_peaks
                    peak.mapped_transcript = transcript
                    peak.mapped_exon = exon

    refined_peaks = OrderedDict()
    # We have to iterate over the peaks again, as the mapping above has to be
    # applied to all peaks first.
    for peak in peaks.values():
        if peak.refined_peak is not None:
            # This is the case when the peak was merged with another peak.
            continue

        new_peak_id = len(refined_peaks)
        if ((peak.mapped_exon is None)
                or ((peak.mapped_exon.start
                     < peak.chrom_start + 1 - exon_peak_boundary_distance)
                    and (peak.mapped_exon.end
                         > peak.chrom_end + exon_peak_boundary_distance)
                    )):
            # Existing peak can be reused, if no exon is mapped to it (then
            # peak is (at least mostly) on intron), or it is not on the
            # boundaries of the mapped extron. But for storing, a different
            # class is used, to have a similar interface as refined peaks.
            refined_peaks[new_peak_id] = UnrefinedPeak(new_peak_id, peak)
        else:
            refined_peaks[new_peak_id] = \
                RefinedPeak(new_peak_id, peak, exon_peak_boundary_distance)

    return transcripts, exons, refined_peaks
