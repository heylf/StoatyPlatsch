
import os
import subprocess as sb
import sys


def create_output_files(peaks, output_path, verbose=False):
    """ Creates the output files for the given peaks.

    File "final_tab_all_peaks.bed":
        Contains an entry for each peak or the corresponding deconvoluted
        peaks, with the following information (tab-separated):

        - chromosome name (from input BED file)
        - start coordinate on the chromosome of the (deconvoluted) peak
        - end coordinate on the chromosome of the (deconvoluted) peak
        - name, consisting of:

            - The name of the line from the input BED file
            - The unique id assigned to this peak after refinement
            - The number of the subpeak
            - Group number, if the subpeak consists of further peaks that
              all together form the overall peak.

        - score (from input BED file)
        - strand (from input BED file)

        - The unique id assigned to this peak after refinement. Can then be
          used in follow up steps to group and combine split peak back again.
        - A zero-based, consecutive number, giving the position of the BED
          entry within a split peak. Therefore, non split peaks have only a 0
          entry here with no relevant information.

    Parameters
    ----------
    peaks : OrderedDict
        The dictionary containing the peaks for which the output should be
        created.
    output_path : str
        The folder path where the coverage file should be saved. Non existing
        folders will be created.
    verbose : bool (default: False)
        Print information to console when set to True.

    Returns
    -------
    file_path_all_peaks : str
        The file path of the output file 'final_tab_all_peaks.tsv'.
    """

    if verbose:
        print("[NOTE] Generate output files.")

    os.makedirs(output_path, exist_ok=True)

    file_path_all_peaks = os.path.join(output_path, "final_tab_all_peaks.bed")

    with open(file_path_all_peaks, "w") as file_all_peaks:
        bed_peak_id = 0
        for peak in peaks.values():
            deconv_peaks_as_bed, bed_peak_id = \
                peak.deconv_peaks_to_bed_entries(bed_peak_id)
            file_all_peaks.write(deconv_peaks_as_bed)
    return file_path_all_peaks


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
        sb.call(cmd, shell=True)

        tmp_file_2 = os.path.join(output_path, "tmp_gene_2.bed")
        cmd = "bedtools intersect -s -v -a {} -b {} > {}".format(
            file_path_peaks, gene_file, tmp_file_2
            )
        if verbose:
            print("[NOTE] ... Create 2nd intersection. Cmd: {}".format(cmd))
        sb.call(cmd, shell=True)

        cmd = "cat {} {} > {}".format(tmp_file_1, tmp_file_2,
                                      output_file_new_peaks_gene)
        if verbose:
            print("[NOTE] ... Concatenate files. Cmd: {}".format(cmd))
        sb.call(cmd, shell=True)

        output_file_new_peaks_sorted = os.path.join(
            output_path,
            "final_tab_all_peaks_annotation_refined_gene_sorted.bed"
            )
        cmd = "sort -V -k1,1 -k2,2n -k3,3n {} > {}".format(
            output_file_new_peaks_gene, output_file_new_peaks_sorted
            )
        if verbose:
            print("[NOTE] ... Create sorted result. Cmd: {}".format(cmd))
        sb.call(cmd, shell=True)

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
        sb.call(cmd, shell=True)

        tmp_file_2 = os.path.join(output_path, "tmp_exon_2.bed")
        cmd = "bedtools intersect -s -v -a {} -b {} > {}".format(
            output_file_new_peaks_gene if gene_file else file_path_peaks,
            exon_file, tmp_file_2
            )
        if verbose:
            print("[NOTE] ... Create 2nd intersection. Cmd: {}".format(cmd))
        sb.call(cmd, shell=True)

        cmd = "cat {} {} > {}".format(tmp_file_1, tmp_file_2,
                                      output_file_new_peaks)
        if verbose:
            print("[NOTE] ... Concatenate files. Cmd: {}".format(cmd))
        sb.call(cmd, shell=True)

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
        sb.call(cmd, shell=True)

        if remove_tmp_files:
            if verbose:
                print("[NOTE] ... Remove temporary files.")
            os.remove(tmp_file_1)
            os.remove(tmp_file_2)
