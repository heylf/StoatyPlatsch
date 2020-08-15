
import os
import subprocess as sb
import sys

import numpy as np


def create_output_files(peaks, output_path, verbose=False):
    """ Creates the output files for the given peaks.

    Three output files are created and saved to 'output_path'.

    File "final_tab_overview.tsv":
        Contains an overview of the deconvoluted peaks. For each peak of the
        original peaks provided as input, the following information are stored
        (tab-separated):

        - name (the name of the line from the input BED file)
        - number of subpeaks
        - list of start coordinates for the subpeaks or a single start
          coordinate, if the peak could not be deconvoluted into multiple
          subpeaks

    File "final_tab_summits.bed":
        Contains an entry for each peak or the corresponding deconvoluted
        peaks, with the following information (tab-separated):

        - chromosome name (from input BED file)
        - coordinate on the chromosome of the summit
        - coordinate on the chromosome of the summit (again due to BED format)
        - name (the name of the line from the input BED file, added by a
          number of the subpeak, if the peak was deconvoluted into multiple
          subpeaks)
        - score (from input BED file)
        - strand (from input BED file)

    File "final_tab_all_peaks.bed":
        Contains an entry for each peak or the corresponding deconvoluted
        peaks, with the following information (tab-separated):

        - chromosome name (from input BED file)
        - start coordinate on the chromosome of the (deconvoluted) peak
        - end coordinate on the chromosome of the (deconvoluted) peak
        - name (the name of the line from the input BED file, added by a
          number of the subpeak, if the peak was deconvoluted into multiple
          subpeaks)
        - score (from input BED file)
        - strand (from input BED file)

    Parameters
    ----------
    peaks : dict
        The dictionary containing the peaks for which the output should be
        created.
    output_path : str
        The folder path where the coverage file should be saved. Non existing
        folders will be created.
    verbose : bool (default: False)
        Print information to console when set to True.

    Returns
    -------
    file_path_overview : str
        The file path of the output file 'final_tab_overview.tsv'.
    file_path_summits : str
        The file path of the output file 'final_tab_summits.tsv'.
    file_path_all_peaks : str
        The file path of the output file 'final_tab_all_peaks.tsv'.
    """

    if verbose:
        print("[NOTE] Generate output files.")

    os.makedirs(output_path, exist_ok=True)

    file_path_overview = os.path.join(output_path, "final_tab_overview.tsv")
    file_overview = open(file_path_overview, "w")
    file_path_summits = os.path.join(output_path, "final_tab_summits.bed")
    file_summits = open(file_path_summits, "w")
    file_path_all_peaks = os.path.join(output_path, "final_tab_all_peaks.bed")
    file_all_peaks = open(file_path_all_peaks, "w")

    for p_id in sorted(peaks.keys()):
        peak = peaks[p_id]

        starts = np.array([], dtype=int)

        for p in peak.fft.new_peaks:
            starts = np.append(starts, p[0])
        file_overview.write(
            "{0}\t{1}\t{2}\n".format(peak.name, len(peak.fft.new_peaks),
                                     starts)
            )

        for p_i, p in enumerate(peak.fft.new_peaks):
            file_summits.write(
                "{0}\t{1}\t{1}\t{2}\t{3}\t{4}\n"
                .format(peak.chrom, p[1],
                        "{}{}".format(peak.name,
                                      "" if len(peak.fft.new_peaks) == 1
                                      else "_{}".format(p_i)),
                        peak.score, peak.strand)
                )

            file_all_peaks.write(
                "{0}\t{1}\t{2}\t{3}\t{4}\t{5}\n"
                .format(peak.chrom, p[0], p[2],
                        "{}{}".format(peak.name,
                                      "" if len(peak.fft.new_peaks) == 1
                                      else "_{}".format(p_i)),
                        peak.score, peak.strand)
                )

    file_overview.close()
    file_summits.close()
    file_all_peaks.close()

    return file_path_overview, file_path_summits, file_path_all_peaks


def refine_peaks_with_annotations(file_path_peaks, gene_file, exon_file,
                                  output_path, remove_tmp_files=True,
                                  verbose=False):
    """ Refine peaks with the given annotation files using bedtools intersect.

    file_path_peaks : str
        The file path of the file containing the peaks that should be refined.
    gene_file : str
        Path to the gene annotation file.
    exon_file : str
        Path to the exon boundary file.
    output_path : str
        The folder path where the coverage file should be saved. Non existing
        folders will be created.
    remove_tmp_files : bool (default: True)
        When set to True temporary files will be deleted, otherwise they are
        kept.
    verbose : bool (default: False)
        Print information to console when set to True.
    """

    if (gene_file is None) and (exon_file is None):
        return

    if verbose:
        print("[NOTE] Refine peaks with annotations.")

    output_file_new_peaks_gene = None

    # Gene Annotation
    if gene_file:

        if verbose:
            print("[NOTE] Use gene annotation ...")

        # Check gene file
        annotation_file = open(gene_file, "r")
        first_line = annotation_file.readline()
        if (len(first_line.strip("\n").split("\t")) < 6):
            sys.exit("[ERROR] Gene annotation has to be in bed6 format!")

        output_file_new_peaks_gene = os.path.join(
            output_path,
            "final_tab_all_peaks_annotation_refined_gene.bed"
            )

        tmp_file_1 = os.path.join(output_path, "tmp_gene_1.bed")
        cmd = "bedtools intersect -s -a {} -b {} > {}".format(
            file_path_peaks, gene_file, tmp_file_1
            )
        if verbose:
            print("[NOTE] ... Create 1st intersection. Cmd: {}".format(cmd))
        sb.Popen(cmd, shell=True).wait()

        tmp_file_2 = os.path.join(output_path, "tmp_gene_2.bed")
        cmd = "bedtools intersect -s -v -a {} -b {} > {}".format(
            file_path_peaks, gene_file, tmp_file_2
            )
        if verbose:
            print("[NOTE] ... Create 2nd intersection. Cmd: {}".format(cmd))
        sb.Popen(cmd, shell=True).wait()

        cmd = "cat {} {} > {}".format(tmp_file_1, tmp_file_2,
                                      output_file_new_peaks_gene)
        if verbose:
            print("[NOTE] ... Concatenate files. Cmd: {}".format(cmd))
        sb.Popen(cmd, shell=True).wait()

        output_file_new_peaks_sorted = os.path.join(
            output_path,
            "final_tab_all_peaks_annotation_refined_gene_sorted.bed"
            )
        cmd = "sort -V -k1,1 -k2,2n -k3,3n {} > {}".format(
            output_file_new_peaks_gene, output_file_new_peaks_sorted
            )
        if verbose:
            print("[NOTE] ... Create sorted result. Cmd: {}".format(cmd))
        sb.Popen(cmd, shell=True).wait()

        if remove_tmp_files:
            if verbose:
                print("[NOTE] ... Remove temporary files.")
            os.remove(tmp_file_1)
            os.remove(tmp_file_2)

    # Exon Annotation
    if exon_file:

        if verbose:
            print("[NOTE] Use exon annotation ...")

        # Check exon file
        annotation_file = open(exon_file, "r")
        first_line = annotation_file.readline()
        if (len(first_line.strip("\n").split("\t")) < 6):
            sys.exit("[ERROR] Exon annotation has to be in bed6 format!")

        output_file_new_peaks = os.path.join(
            output_path,
            "final_tab_all_peaks_annotation_refined_{}exon.bed".format(
                "gene_and_" if gene_file else ""
                )
            )

        tmp_file_1 = os.path.join(output_path, "tmp_exon_1.bed")
        cmd = "bedtools intersect -s -a {} -b {} > {}".format(
            output_file_new_peaks_gene if gene_file else file_path_peaks,
            exon_file, tmp_file_1
            )
        if verbose:
            print("[NOTE] ... Create 1st intersection. Cmd: {}".format(cmd))
        sb.Popen(cmd, shell=True).wait()

        tmp_file_2 = os.path.join(output_path, "tmp_exon_2.bed")
        cmd = "bedtools intersect -s -v -a {} -b {} > {}".format(
            output_file_new_peaks_gene if gene_file else file_path_peaks,
            exon_file, tmp_file_2
            )
        if verbose:
            print("[NOTE] ... Create 2nd intersection. Cmd: {}".format(cmd))
        sb.Popen(cmd, shell=True).wait()

        cmd = "cat {} {} > {}".format(tmp_file_1, tmp_file_2,
                                      output_file_new_peaks)
        if verbose:
            print("[NOTE] ... Concatenate files. Cmd: {}".format(cmd))
        sb.Popen(cmd, shell=True).wait()

        output_file_new_peaks_sorted = os.path.join(
            output_path,
            "final_tab_all_peaks_annotation_refined_{}exon_sorted.bed".format(
                "gene_and_" if gene_file else ""
                )
            )
        cmd = "sort -V -k1,1 -k2,2n -k3,3n {} > {}".format(
            output_file_new_peaks, output_file_new_peaks_sorted
            )
        if verbose:
            print("[NOTE] ... Create sorted result. Cmd: {}".format(cmd))
        sb.Popen(cmd, shell=True).wait()

        if remove_tmp_files:
            if verbose:
                print("[NOTE] ... Remove temporary files.")
            os.remove(tmp_file_1)
            os.remove(tmp_file_2)
