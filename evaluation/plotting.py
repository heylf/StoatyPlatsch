
from collections import OrderedDict
import itertools
import re

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.widgets import CheckButtons, RadioButtons, Slider, TextBox
import numpy as np

from FFT.processing import deconvolute_with_FFT
from STFT.processing import deconvolute_peaks_with_STFT
from tools.data import OriginalPeak, RefinedPeak, UnrefinedPeak
from tools.plotting import draw_profile
from tools.postprocessing import read_fasta_file
from tools.preprocessing import add_transcript_annotations, read_coverage_file


class PeakViewer(object):
    """ Visualizes peaks with genomic data and deconvolution results.

    Attributes
    ----------

    Attributes defined by given initialization parameters:

    output_folder : str
        The folder path where generated files should be saved. Non existing
        folders will be created.
    transcript_file : str
        The file path of the transcript annotation file in GTF format that
        should be read. Only the information regarding transcripts and exons
        are extracted.
    motifs : list
        List of motifs that should be highlighted when found in the genomic
        data.
    fasta_files : list
        List of file paths to fasta files used for extracting and adding
        genomic data.
    verbose : bool
        Print information to console when set to True.
    paper_plots : bool
        Use specific plot settings when set to True to improve quality when
        embedding the plots into a Latex document.

    peaks_orig : list
        List of OriginalPeaks as read from the given coverage file.
    peaks_with_tr_info : list
        List of UnrefinedPeaks and RefinedPeaks, yielded by embedding
        transcript information to the OriginalPeaks list.
    peak : OriginalPeak, RefinedPeak, UnrefinedPeak
        The peak currently selected and plotted.
    peaks : list
        The list of peaks currently activated, either 'self.peaks_orig' or
        'self.peaks_with_tr_info' or a filtered version of
        'self.peaks_with_tr_info'.
    fasta_data : list
        List containing the data of the fasta files given by
        'self.fasta_files'.
    transcripts : OrderedDict
        Dictionary containing the transcript data.
    exons : OrderedDict
        Dictionary containing the exon data.
    peak_fasta_keys : list
        List with the keys used to select the genomic data from the fasta data.
    peak_genomic_labels : list
        List of labels that is used for plotting the genomic data.
    peak_genomic_data : list
        List of the genomic data that should be plotted.
    peak_genomic_pos_rel : list
        List of the relative positions of the genomic data.
    refined_peak_axis_label : str
        The label used for plotting the genomic data of a refined peak.
    refined_peak_genomic_data : str
        The genomic data of a refined peak.


    Attributes for the control figure:

    fig_controls : Figure
        The Figure containing the control elements.
    ax_r_transcripts : AxesSubplot
        The axes for plotting the RadioButton to change transcript embedding.
    ax_c_peak_filter : AxesSubplot
        The axes for plotting the RadioButton to filter the peaks.
    ax_s_peakid : AxesSubplot
        The axes for plotting the Slider to change the selected peak.
    ax_tb_peakid : AxesSubplot
        The axes for plotting the TextBox to change the selected peak.
    self.ax_text_info : AxesSubplot
        The axes for plotting the Text with some information of the selected
        peak.
    ax_plot_options_c : AxesSubplot
        The axes for plotting the CheckButtons to switch plotting options.
    r_transcripts : RadioButtons
        The RadioButton for changing transcript embedding.
    r_peak_filter : RadioButtons
        The RadioButton for filtering the peak list for refined peaks.
    s_peakid : Slider
        The Slider for changing the selected peak.
    tb_peakid : TextBox
        The TextBox for changing the selected peak.
    text_info : Text
        The Text with information of the selected peak.
    c_plot_options : CheckButtons
        The CheckButtons for switching the plotting options.

    Attributes for the peak figure:
    fig_peak : Figure
        The Figure used for plotting the peak.
    ax_peak : AxesSubplot
        The aces for plotting the peak profile.
    ax_genomic_data : list
        List with axes for plotting the genomic data. These axes here are only
        used for defining the general position and synchronising the xaxis with
        the xaxis of 'self.ax_peak'. For actually plotting the genomic data
        / nucleotides, the axes in 'self.ax_nts' are used.
    ax_subpeaks : dict
        Dictionary containing the axes for plotting the subpeaks of the
        deconvolution approaches.
    ax_nts : list
        List with secondary axes that are actually used to plot the genomic
        data.
    ax_nts_labels : list
        List with labels that are used for denoting the origin of the genomic
        data.


    Notes
    -----
    Currently the plots are optimized for positive strands. The plots for
    negative should be further improved, especially for refined peaks! Flipping
    the genomic data might be necessary, but then highlighting must be adapted,
    too.
    """

    def __init__(self, input_coverage_file, output_folder,
                 transcript_file, motifs=None, fasta_files=None,
                 init_peak_ID=None, verbose=False, paper_plots=False,
                 postpone_show=False):
        """ Constructor

        Parameters
        ----------
        input_coverage_file : str
            The file path of the coverage file that should be read.
        output_folder : str
            The folder path where generated files should be saved. Non existing
            folders will be created.
        transcript_file : str
            The file path of the transcript annotation file in GTF format that
            should be read. Only the information regarding transcripts and
            exons are extracted.
        motifs : list (default: None)
            List of motifs that should be highlighted when found in the genomic
            data.
        fasta_files : list (default: None)
            List of file paths to fasta files used for extracting and adding
            genomic data.
        init_peak_ID : int (default: None)
            Sets the peak that should be plotted initially.
        verbose : bool (default: False)
            Print information to console when set to True.
        paper_plots : bool (default: False)
            Use specific plot settings when set to True to improve quality when
            embedding the plots into a Latex document.
        postpone_show : bool (default: False)
            Do not show plots directly, if set to True.

        See attributes description of the class for other parameters.
        """
        self.output_folder = output_folder
        self.transcript_file = transcript_file
        self.motifs = motifs
        self.fasta_files = fasta_files
        self.verbose = verbose
        self.paper_plots = paper_plots

        self.fasta_data = []
        if self.paper_plots:
            plt.rcParams.update({'font.size': 15})
        matplotlib.rcParams["figure.figsize"] = [12, 8]

        self.peaks_orig = read_coverage_file(input_coverage_file, verbose)
        self.peaks_with_tr_info = None

        self.peak = None
        self.peaks = self.peaks_orig

        if self.fasta_files:
            for f in self.fasta_files:
                self.fasta_data.append(read_fasta_file(f, verbose))

        if verbose:
            print("[NOTE] Motifs to highlight: {}".format(self.motifs))

        self.create_fig_controls()
        self.fig_peak = None

        self.r_transcripts_on_clicked(None)
        self.s_peakid_changed(init_peak_ID if init_peak_ID
                              else self.s_peakid.val)
        if not postpone_show:
            plt.show()

    def reset_textbox_events(self, tb):
        """ Resets events for the given TextBox, workaround for a bug.

        Parameters
        ----------
        tb : TextBox
            The TextBox for which the events should be reset.
        """

        def custom_keypress(x):
            """ Wrapper function to prevent exceptions. """
            if x.key is not None:
                tb._keypress(x)

        tb.disconnect_events()
        tb.connect_event('button_press_event', tb._click)
        tb.connect_event('button_release_event', tb._release)
        tb.connect_event('motion_notify_event', tb._motion)
        tb.connect_event('resize_event', tb._resize)
        # That's the event that causes exceptions when using 'Tab' key. Use
        # custom key press event as a wrapper.
        # tb.connect_event('key_press_event', tb._keypress)
        tb.connect_event('key_press_event', custom_keypress)

    def create_fig_controls(self):
        """ Creates the control figure and control elements. """
        self.fig_controls = plt.figure("Controls", figsize=[10.0, 6.0])
        self.fig_controls.subplots_adjust(left=0.1, bottom=0.0, right=1.0,
                                          top=0.94, wspace=0.0, hspace=0.03)
        gs = self.fig_controls.add_gridspec(9, 10)

        # Create axes for control elements.
        self.ax_r_transcripts = self.fig_controls.add_subplot(gs[0, :3])
        self.ax_r_peak_filter = self.fig_controls.add_subplot(gs[0:2, 4:-1])
        self.ax_s_peakid = self.fig_controls.add_subplot(gs[3, :6])
        self.ax_tb_peakid = self.fig_controls.add_subplot(gs[3, 7:-1])
        self.ax_text_info = self.fig_controls.add_subplot(gs[4, 0])
        self.ax_plot_options_c = self.fig_controls.add_subplot(gs[5:8, :9])

        # Create control elements.
        self.r_transcripts = RadioButtons(
            self.ax_r_transcripts, ('Original / Raw', 'Transcripts'), active=0)
        self.r_transcripts.on_clicked(self.r_transcripts_on_clicked)
        self.r_peak_filter = RadioButtons(
            self.ax_r_peak_filter,
            ('All', 'Unrefined', 'Refined', 'Refined with multiple segments'))
        self.r_peak_filter.on_clicked(self.r_peak_filter_on_clicked)
        self.r_peak_filter.ax.set_visible(False)
        self.s_peakid = Slider(self.ax_s_peakid, 'Peak ID',
                               0, 1, valinit=16, valstep=1)
        self.s_peakid.on_changed(self.s_peakid_changed)
        self.tb_peakid = TextBox(self.ax_tb_peakid, '', self.s_peakid.val)
        self.reset_textbox_events(self.tb_peakid)
        self.tb_peakid.on_submit(self.tb_peakid_on_submit)
        self.ax_text_info.set_axis_off()
        self.text_info = self.ax_text_info.text(0, 0.5, 'Selected peak: -')
        self.c_plot_options = CheckButtons(
            self.ax_plot_options_c,
            ['Show FFT deconvolution results',
             'Show STFT deconvolution results (without transcript)',
             'Show STFT deconvolution results (with transcript)',
             'Show original peaks (only for refined peaks)',
             'Show genomic data for original peak(s)',
             'Show genomic data for refined peak',
             'Show center of subpeaks'],
            [False, False, False, True, True, False, False]
            )
        self.c_plot_options.on_clicked(self.c_plot_options_clicked)

    def r_transcripts_on_clicked(self, val):
        """ Handles the radio button event for changed transcript embedding.

        Parameters
        ----------
        val : str
            The new selected transcript embedding.
        """

        if val == 'Transcripts':
            if not self.peaks_with_tr_info:
                self.transcripts, self.exons, self.peaks_with_tr_info = \
                    add_transcript_annotations(
                        peaks=self.peaks_orig,
                        transcript_file=self.transcript_file,
                        output_path=self.output_folder,
                        verbose=self.verbose)
                if self.verbose:
                    self.print_transcript_statistics()

            self.r_peak_filter.ax.set_visible(True)
        else:
            self.r_peak_filter.ax.set_visible(False)

        self.select_peaks()

    def r_peak_filter_on_clicked(self, _val):
        """ Handles the radio button event for changing the peak filter.

        Parameters
        ----------
        _val : str
            The new selected peak filter.
        """
        self.select_peaks()

    def select_peaks(self):
        """ Sets and filters the peaks according to the selected options. """

        if self.r_transcripts.value_selected == 'Transcripts':
            if self.r_peak_filter.value_selected == 'All':
                self.peaks = self.peaks_with_tr_info
            else:
                filtered_peaks = OrderedDict()
                new_id = 0
                for p in self.peaks_with_tr_info.values():
                    if self.r_peak_filter.value_selected == 'Unrefined':
                        if isinstance(p, UnrefinedPeak):
                            filtered_peaks[new_id] = p
                            new_id += 1
                    elif isinstance(p, RefinedPeak):
                        if self.r_peak_filter.value_selected == 'Refined':
                            filtered_peaks[new_id] = p
                            new_id += 1
                        elif len(p.orig_peaks) > 1:
                            filtered_peaks[new_id] = p
                            new_id += 1
                self.peaks = filtered_peaks
        else:
            self.peaks = self.peaks_orig

        current_peakid = self.s_peakid.val

        # Update possible values for peak ID.
        self.s_peakid.valmax = len(self.peaks)-1
        self.s_peakid.set_val(current_peakid)
        self.s_peakid.ax.set_xlim(self.s_peakid.valmin, self.s_peakid.valmax)

        self.fig_controls.canvas.draw()

    def s_peakid_changed(self, val):
        """ Handles the slider event for a changed peak_id.

        Parameters
        ----------
        val : int
            The new peak_id value.
        """
        if len(self.peaks) == 0:
            if self.fig_peak and plt.fignum_exists(self.fig_peak.number):
                self.fig_peak.clear()
                self.fig_peak.canvas.draw()
            self.text_info.set_text('Selected peak: -')
            self.peak = None
            return
        if val < 0:
            self.s_peakid.set_val(0)
            return
        elif val > self.s_peakid.valmax:
            self.s_peakid.set_val(self.s_peakid.valmax)
            return
        elif self.peak != self.peaks[val]:
            self.peak = self.peaks[val]
            self.tb_peakid.set_val(val)
            self.refresh_peak()

    def tb_peakid_on_submit(self, val):
        """ Handles the text box event for a changed peak_id.

        Parameters
        ----------
        val : str
            The new peak_id value.
        """
        try:
            peak_id = int(val)
        except Exception:
            self.tb_peakid.set_val(self.s_peakid.val)
            return
        self.s_peakid.set_val(peak_id)

    def c_plot_options_clicked(self, _val):
        """ Handles the check button event for changed options.

        Parameters
        ----------
        _val : str
            The option that has been changed.
        """
        self.refresh_peak()

    def refresh_peak(self):
        """ Refreshes all values and plots for a changed peak or option. """

        if self.verbose:
            print("[NOTE] Selected peak: {}".format(self.peak))

        self.text_info.set_text('Selected peak: {}'.format(self.peak))

        if self.fig_peak and plt.fignum_exists(self.fig_peak.number):
            self.fig_peak.clear()

        self.deconv_peak()

        self.load_genomic_data_for_peak()

        self.create_fig_peak_layout()

        lines = draw_profile(self.peak, self.fig_peak, self.ax_peak,
                             paper_plots=self.paper_plots)

        # Plot original peaks for a refined peak and activated plot option.
        if (self.c_plot_options.get_status()[3] and
                isinstance(self.peak, RefinedPeak)):
            if len(self.peak.orig_peaks) > 1:
                legend_label = 'Original peaks'
            else:
                legend_label = 'Original peak'
            prev_lengths = 0
            for pi, orig_peak in enumerate(self.peak.orig_peaks):
                shift = orig_peak.chrom_start - self.peak.starts[pi]
                x_values = range(
                    prev_lengths + shift,
                    prev_lengths + shift + len(orig_peak.coverage)
                    )
                y_values = orig_peak.coverage
                self.ax_peak.plot(
                    x_values, y_values,
                    label=legend_label if pi == 0 else '',
                    color='grey', linestyle='--'
                    )
                prev_lengths += len(self.peak.coverages[pi])
            lines[0].set_label('Refined peak')
            self.ax_peak.legend()

        self.add_genomic_data_to_profile()

        self.highlight_genomic_data()

        self.add_subpeaks()

        self.adjust_peakplot()

    def deconv_peak(self):
        """ Applies the deconvolution approaches to the current peak.

        The different deconvolution approaches are only applied when the
        corresponding options are selected. The results are only calculated
        once per peak and reused afterwards.
        """
        if self.c_plot_options.get_status()[0]:
            peaks_to_deconv = {}
            if isinstance(self.peak, OriginalPeak):
                if not hasattr(self.peak, 'fft'):
                    peaks_to_deconv[self.peak.peak_id] = self.peak
            else:
                for orig_peak in self.peak.orig_peaks:
                    if not hasattr(orig_peak, 'fft'):
                        peaks_to_deconv[orig_peak.peak_id] = orig_peak
            deconvolute_with_FFT(peaks=peaks_to_deconv, num_padding=[10, 10])
            for p in peaks_to_deconv.values():
                p.fft.deconv_peaks_rel = []
                for new_peak in p.fft.new_peaks:
                    p.fft.deconv_peaks_rel.append(
                        [new_peak[0] - p.chrom_start,
                         new_peak[1] - p.chrom_start,
                         new_peak[2] - p.chrom_start
                         ])
            if self.verbose:
                print("[NOTE] FFT deconvolution ...")
                if isinstance(self.peak, OriginalPeak):
                    print("[NOTE] ... self.peak.fft.deconv_peaks_rel:",
                          self.peak.fft.deconv_peaks_rel)
                else:
                    for orig_peak in self.peak.orig_peaks:
                        print("[NOTE] ... orig_peak.fft.deconv_peaks_rel:",
                              orig_peak.fft.deconv_peaks_rel)
        if self.c_plot_options.get_status()[1]:
            peaks_to_deconv = OrderedDict()
            if isinstance(self.peak, OriginalPeak):
                if not hasattr(self.peak, 'deconv_peaks_rel'):
                    peaks_to_deconv[self.peak.peak_id] = self.peak
            else:
                for orig_peak in self.peak.orig_peaks:
                    if not hasattr(orig_peak, 'deconv_peaks_rel'):
                        peaks_to_deconv[orig_peak.peak_id] = orig_peak
            deconvolute_peaks_with_STFT(peaks=peaks_to_deconv)
            if self.verbose:
                print("[NOTE] STFT deconvolution ...")
                if isinstance(self.peak, OriginalPeak):
                    print("[NOTE] ... self.peak.deconv_peaks_rel:",
                          self.peak.deconv_peaks_rel)
                else:
                    for orig_peak in self.peak.orig_peaks:
                        print("[NOTE] ... orig_peak.deconv_peaks_rel:",
                              orig_peak.deconv_peaks_rel)
        if (self.c_plot_options.get_status()[2]
                and not isinstance(self.peak, OriginalPeak)):
            peaks_to_deconv = OrderedDict()
            if not hasattr(self.peak, 'deconv_peaks_rel'):
                peaks_to_deconv[self.peak.peak_id] = self.peak
            deconvolute_peaks_with_STFT(peaks=peaks_to_deconv)
            if self.verbose:
                print("[NOTE] STFT deconvolution with transcript info ...")
                print("[NOTE] ... self.peak.deconv_peaks_rel:",
                      self.peak.deconv_peaks_rel)

    def load_genomic_data_for_peak(self):
        """ Loads the fasta data for the current selected peak. """
        self.peak_fasta_keys = []
        self.peak_genomic_labels = []
        self.peak_genomic_data = []
        self.peak_genomic_pos_rel = []
        if isinstance(self.peak, (OriginalPeak, UnrefinedPeak)):
            # Selected peak is an OriginalPeak or UnrefinedPeak peak, so only
            # on segment with genomic data has to be plotted.
            if isinstance(self.peak, OriginalPeak):
                fasta_key = "{}:{}-{}({})".format(
                    self.peak.chrom, self.peak.chrom_start,
                    self.peak.chrom_end, self.peak.strand)
                axis_label = "{}:{}-{}({})".format(
                    self.peak.chrom, self.peak.chrom_start+1,
                    self.peak.chrom_end, self.peak.strand)
                data_length = self.peak.chrom_end - self.peak.chrom_start
            else:
                if self.verbose:
                    print("[NOTE] ... {}".format(self.peak.orig_peak))
                fasta_key = "{}:{}-{}({})".format(
                    self.peak.seqname, self.peak.start,
                    self.peak.end, self.peak.orig_peak.strand)
                axis_label = "{}:{}-{}({})".format(
                    self.peak.seqname, self.peak.start+1,
                    self.peak.end, self.peak.orig_peak.strand)
                data_length = self.peak.end - self.peak.start
            genomic_data = self.get_genomic_data(fasta_key)
            if genomic_data is None:
                genomic_data = '?'*data_length
#           Negative strand causes currently problems for refined peaks.
#           Possible solution is following code, but then highlighting does
#           not work as intended.
#             if ((isinstance(self.peak, OriginalPeak)
#                     and (self.peak.strand == '-'))
#                 or (isinstance(self.peak, UnrefinedPeak)
#                     and (self.peak.orig_peak.strand == '-'))):
#                 genomic_data = genomic_data[::-1]

            self.peak_fasta_keys.append(fasta_key)
            self.peak_genomic_labels.append(axis_label)
            self.peak_genomic_data.append(genomic_data)
            self.peak_genomic_pos_rel.append(range(len(genomic_data)))
        else:
            # Selected peak is a RefinedPeak, so it can consist of multiple
            # segments of genomic data on different position.
            prev_lengths = 0
            for pi, p in enumerate(self.peak.orig_peaks):
                if self.verbose:
                    print("[NOTE] ... {}".format(p))
                fasta_key = "{}:{}-{}({})".format(
                    p.chrom, p.chrom_start, p.chrom_end, p.strand)
                axis_label = "{}:{}-{}({})".format(
                    p.chrom, p.chrom_start+1, p.chrom_end, p.strand)
                shift = p.chrom_start - self.peak.starts[pi]
                genomic_data = self.get_genomic_data(fasta_key)
                data_length = p.chrom_end - p.chrom_start
                if genomic_data is None:
                    genomic_data = '?'*data_length
#               Negative strand causes currently problems for refined peaks.
#               Possible solution is following code, but then highlighting does
#               not work as intended.
#                 if p.strand == '-':
#                     genomic_data = genomic_data[::-1]

                self.peak_fasta_keys.append(fasta_key)
                self.peak_genomic_labels.append(axis_label)
                self.peak_genomic_data.append(genomic_data)
                self.peak_genomic_pos_rel.append(
                    range(prev_lengths + shift,
                          prev_lengths + shift + len(genomic_data))
                    )

                prev_lengths += len(self.peak.coverages[pi])

            # In addition, the combined genomic data of the final refined peak
            # needs to be loaded.
            self.refined_peak_axis_label = ''
            self.refined_peak_genomic_data = ''
            for i, (seq, start, end) in enumerate(
                    zip(self.peak.seqnames, self.peak.starts, self.peak.ends)
                    ):
                fasta_key = "{}:{}-{}({})".format(
                    seq, start, end, self.peak.orig_peaks[i].strand)
                axis_label = "{}:{}-{}({})".format(
                    seq, start+1, end, self.peak.orig_peaks[i].strand)
                genomic_data = self.get_genomic_data(fasta_key)
                data_length = end - start
                if genomic_data is None:
                    genomic_data = '?'*data_length
#               Negative strand causes currently problems for refined peaks.
#               Possible solution is following code, but then highlighting does
#               not work as intended.
#                 if self.peak.orig_peaks[i].strand == '-':
#                     genomic_data = genomic_data[::-1]
                self.refined_peak_genomic_data += genomic_data
                if self.refined_peak_axis_label != '':
                    self.refined_peak_axis_label += '\n$\cup$ '
                self.refined_peak_axis_label += axis_label

        if self.verbose:
            print("[NOTE] ... self.peak_fasta_keys: ", self.peak_fasta_keys)
            print("[NOTE] ... self.peak_genomic_labels: ",
                  self.peak_genomic_labels)
            print("[NOTE] ... self.peak_genomic_data: ",
                  self.peak_genomic_data)
            print("[NOTE] ... self.peak_genomic_pos_rel: ",
                  self.peak_genomic_pos_rel)
            if isinstance(self.peak, RefinedPeak):
                print("[NOTE] ... self.refined_peak_axis_label: ",
                      self.refined_peak_axis_label.replace('\n', ' '))
                print("[NOTE] ... self.refined_peak_genomic_data: ",
                      self.refined_peak_genomic_data)

    def get_genomic_data(self, fasta_key):
        """ Searches and returns the genomic data for the given fasta key.

        Parameters
        ----------
        fasta_key : str
            The key that should be used to select the genomic data from the
            fasta data.

        Returns
        -------
        genomic_data : str, None
            The genomic data for the given fasta key, if the key was found.
            Otherwise a warning is issued and None is returned.
        """
        for f in self.fasta_data:
            genomic_data = f.get(fasta_key)
            if genomic_data:
                return genomic_data
        print("[WARNING] Fasta key '{}' not found!".format(fasta_key))
        return None

    def create_fig_peak_layout(self):
        """ Creates the figure and layout for plotting the peak. """

        # Contains the height ratio of the different element and defines how
        # they should be plotted in relation to each other.
        height_ratios = [12,    # Profile
                         3      # Additional space
                         ]

        # Determine values for plotting the genomic data.
        genome_data_height_ratio = 2
        if self.c_plot_options.get_status()[4]:
            row_count_genomic_data = len(self.peak_genomic_labels)
            height_ratios.extend(
                [genome_data_height_ratio]*row_count_genomic_data)
        else:
            row_count_genomic_data = 0
        if (self.c_plot_options.get_status()[5]
                and isinstance(self.peak, RefinedPeak)):
            row_count_genomic_data += 1
            height_ratios.extend([genome_data_height_ratio])

        # Determine values for plotting the deconvolution results.
        row_counts_deconv = []
        if self.c_plot_options.get_status()[0]:
            if isinstance(self.peak, OriginalPeak):
                row_count_fft = len(self.peak.fft.deconv_peaks_rel)
            else:
                row_count_fft = 0
                for orig_peak in self.peak.orig_peaks:
                    row_count_fft = max(row_count_fft,
                                        len(orig_peak.fft.deconv_peaks_rel))
            row_counts_deconv.append(row_count_fft)
        else:
            row_counts_deconv.append(0)
        if self.c_plot_options.get_status()[1]:
            if isinstance(self.peak, OriginalPeak):
                row_count_stft_wo_tr = len(self.peak.deconv_peaks_rel)
            else:
                row_count_stft_wo_tr = 0
                for orig_peak in self.peak.orig_peaks:
                    row_count_stft_wo_tr = max(row_count_stft_wo_tr,
                                               len(orig_peak.deconv_peaks_rel))
            row_counts_deconv.append(row_count_stft_wo_tr)
        else:
            row_counts_deconv.append(0)
        if (self.c_plot_options.get_status()[2]
                and not isinstance(self.peak, OriginalPeak)):
            row_count_stft_w_tr = len(self.peak.deconv_peaks_rel)
            row_counts_deconv.append(row_count_stft_w_tr)
        else:
            row_counts_deconv.append(0)
        dist = 2
        add_final_row = []
        for r in row_counts_deconv:
            if r > 0:
                height_ratios.extend([dist,    # Additional space
                                      r])
                dist = 3.5
                add_final_row = [1]
        height_ratios.extend(add_final_row)

        # For layout development.
        develop = False
        if develop:
            # Note: Activating this messes up the final margin adjustments on
            # the right side.
            width_ratios = [18, 1, 1]
            height_ratios.extend([1, 1])
        else:
            width_ratios = [1]

        col_count = len(width_ratios)
        row_count = len(height_ratios)

        figsize = matplotlib.rcParams["figure.figsize"].copy()
        height_per_row = figsize[1] / 30
        figsize[1] = height_per_row * sum(height_ratios)

        # Create the figure with basic layout for plotting the peak profile.

        self.fig_peak = plt.figure("Peak")
        self.fig_peak.set_size_inches(figsize)    # Sets here separately and
        # not when figure is created to enable resizing.

        gs = self.fig_peak.add_gridspec(
            row_count, col_count,
            height_ratios=height_ratios,
            width_ratios=width_ratios,
            hspace=0.0, wspace=0.0
            )

        if develop:
            for ri in range(row_count):
                _ax_tmp = self.fig_peak.add_subplot(gs[ri, -1])
            for ci in range(col_count-1):
                _ax_tmp = self.fig_peak.add_subplot(gs[-1, ci])

        self.ax_peak = self.fig_peak.add_subplot(gs[0, 0])

        # Create aces for plotting genomic data.
        self.ax_genomic_data = []
        for i in range(row_count_genomic_data):
            self.ax_genomic_data.append(self.fig_peak.add_subplot(
                gs[2+i, 0],
                sharex=self.ax_peak))

        # Create axes for plotting the results of the deconvolution approaches.
        self.ax_subpeaks = {}
        dist = 0
        for i, r in enumerate(row_counts_deconv):
            if r > 0:
                self.ax_subpeaks[i] = self.fig_peak.add_subplot(
                    gs[2+row_count_genomic_data+1+dist, 0],
                    sharex=self.ax_peak)
                dist += 2
                if i == 0:
                    subpeak_title = 'Deconvolution with FFT'
                elif i == 1:
                    subpeak_title = 'Deconvolution with STFT'
                else:
                    subpeak_title = \
                        'Deconvolution with STFT and transcript information'
                self.ax_subpeaks[i].set_title(subpeak_title)
            else:
                self.ax_subpeaks[i] = None

    def add_genomic_data_to_profile(self):
        """ Adds the genomic data to the profile plot. """

        self.ax_nts = []
        self.ax_nts_labels = []
        # Handles the plotting of the genomic data for original peak(s).
        if self.c_plot_options.get_status()[4]:
            for i, (label, pos_rel, data) in enumerate(itertools.zip_longest(
                    self.peak_genomic_labels, self.peak_genomic_pos_rel,
                    self.peak_genomic_data
                    )):
                self.add_genomic_data_to_axis(self.ax_genomic_data[i],
                                              label, pos_rel, data)
        # Handles the plotting of the genomic data for a refined peak.
        if (self.c_plot_options.get_status()[5]
                and isinstance(self.peak, RefinedPeak)):
            self.add_genomic_data_to_axis(
                self.ax_genomic_data[-1], self.refined_peak_axis_label,
                range(len(self.refined_peak_genomic_data)),
                self.refined_peak_genomic_data)
            ranges = np.array(self.peak.ends) - np.array(self.peak.starts)
            if len(ranges) > 1:
                # Mark different segments by plotting vertical lines between
                # nucleotides of different segments.
                peak_segment_end_pos = np.cumsum(ranges)
                for p in peak_segment_end_pos[:-1]:
                    self.ax_genomic_data[-1].axvline(
                        x=p-0.5, ymin=0.5, color='grey', lw=1)

    def add_genomic_data_to_axis(self, ax_gd, label, pos_rel, data):
        """ Add the given genomic data to the given axes.

        Parameters
        ----------
        ax_gd : AxesSubplot
            The axes (from 'self.ax_genomic_data') that should be used to add
            a secondary axis to plot the genomic data.
        label : str
            The label for the secondary axis.
        pos_rel : range
            The ranege with the relative nucleotide positions.
        data : str
        """

        # Hide the axes of ax_gd as they are only used for the basic layouting.
        for p in ['top', 'right', 'bottom', 'left']:
            ax_gd.spines[p].set_visible(False)
        ax_gd.tick_params(bottom=False, labelbottom=False)
        ax_gd.yaxis.set_visible(False)

        # Create the secondary axis with the data and apply basic formatting of
        # nucleotide data.

        # If location 'top' is used some issues with formatting occur.
        # Setting location to 1.0 fist and the switching the ticks to
        # bottom ticks seems to circumvent this issues. Probably there
        # is difference between ticks that are considered top and
        # bottom ticks, but formatting options seem to not handle this
        # in all cases.
        ax_nt = ax_gd.secondary_xaxis(location=1.0)
        self.ax_nts.append(ax_nt)

        ax_nt.xaxis.tick_bottom()
        ax_nt.set_xticks(pos_rel)
        ax_nt.set_xticklabels(data)

        self.color_nt_labels(ax_nt.get_xticklabels())

        axislabel = ax_gd.set_xlabel(label)
        axislabel.set_horizontalalignment('left')
        ax_gd.xaxis.set_label_coords(1.01, 1.0)
        self.ax_nts_labels.append(axislabel)

    def color_nt_labels(self, labels):
        """ Colors the given labels when they correspond to nucleotides.

        Parameters
        ----------
        labels : list of `~matplotlib.text.Text`
            List of nucleotide labels that should be colored.
        """
        for l in labels:
            if l.get_text() == 'A':
                l.set_color('red')
            elif l.get_text() == 'C':
                l.set_color('blue')
            elif l.get_text() == 'G':
                l.set_color('yellow')
            elif l.get_text() in ['T', 'U']:
                l.set_color('green')

    def highlight_genomic_data(self):
        """ Highlight occurrences of defined motifs in the genomic data. """
        if self.c_plot_options.get_status()[4]:
            for i, genomic_data in enumerate(self.peak_genomic_data):
                highlight_nts = self.find_motif_indices(genomic_data)
                for nt in highlight_nts:
                    self.highlight_nt_tick(self.ax_nts[i], nt)
        if (self.c_plot_options.get_status()[5]
                and isinstance(self.peak, RefinedPeak)):
            highlight_nts = self.find_motif_indices(
                self.refined_peak_genomic_data)
            for nt in highlight_nts:
                self.highlight_nt_tick(self.ax_nts[-1], nt)
        if plt.fignum_exists(self.fig_peak.number):
            self.fig_peak.canvas.draw()

    def find_motif_indices(self, genomic_data):
        """ Searches for occurrences of motifs in the given data.

        Parameters
        ----------
        genomic_data : str
            The genomic data that should be searched for the defines motifs.

        Returns
        -------
        match_indices : set
            A set with the indices that are part of a motif.
        """
        if self.motifs is None:
            return set()
        match_indices = []
        for motif in self.motifs:
            matches = re.finditer(motif, genomic_data)
            for m in matches:
                for mi in range(m.start(), m.end()):
                    match_indices.append(mi)
        if self.verbose:
            print("[NOTE] Highlight: ", match_indices)
        return set(match_indices)

    def highlight_nt_tick(self, ax_nt, nt_pos):
        """ Searches and returns the genomic data for the given fasta key.

        Parameters
        ----------
        ax_nt : SecondaryAxis
            The secondary axis that should be used for highlighting.
        nt_pos : int
            The position that should be highlighted.
        """
        tl = ax_nt.get_xticklabels()[nt_pos]
        if tl.get_fontweight() == 'bold':
            return

        tl.set_fontweight('bold')
        tl.set_fontsize(tl.get_fontsize() + 2)
        mt = ax_nt.xaxis.get_major_ticks()[nt_pos]

        scale = 3 if not self.paper_plots else 4
        mt.tick1line.set_markeredgewidth(
            scale * mt.tick1line.get_markeredgewidth()
            )

    def add_subpeaks(self):
        """ Plots the subpeaks of the different deconvolution approaches. """

        xlim = self.ax_peak.get_xlim()

        # Plot the FFT results.
        if self.c_plot_options.get_status()[0]:
            if isinstance(self.peak, OriginalPeak):
                subpeaks_all_sections = [self.peak.fft.deconv_peaks_rel]
            else:
                subpeaks_all_sections = []
                for orig_peak in self.peak.orig_peaks:
                    subpeaks_all_sections.append(
                        orig_peak.fft.deconv_peaks_rel)
            subpeaks_names = []
            # Start and end for the different bars to plot (so not only start
            #     of peak, but also center of peak)
            subpeaks_starts = []
            subpeaks_widths = []
            prev_lengths = 0
            for si, subpeaks_section in enumerate(subpeaks_all_sections):
                if isinstance(self.peak, RefinedPeak):
                    shift = (self.peak.orig_peaks[si].chrom_start
                             - self.peak.starts[si])
                else:
                    shift = 0
                for i, (l, c, r) in enumerate(subpeaks_section):
                    if self.c_plot_options.get_status()[6]:
                        subpeaks_names.insert(0, "Subpeak {}".format(i))
                        subpeaks_names.insert(0, "Subpeak {}".format(i))
                        subpeaks_starts.insert(0, prev_lengths + shift + l)
                        subpeaks_starts.insert(0, prev_lengths + shift + c)
                        subpeaks_widths.insert(0, c-l)
                        subpeaks_widths.insert(0, r-c-1)
                    else:
                        subpeaks_names.insert(0, "Subpeak {}".format(i))
                        subpeaks_starts.insert(0, prev_lengths + shift + l)
                        subpeaks_widths.insert(0, r-l-1)
                if isinstance(self.peak, RefinedPeak):
                    prev_lengths += len(self.peak.coverages[si])
            ax_subpeak = self.ax_subpeaks[0]
            ax_subpeak.barh(
                y=subpeaks_names,
                left=subpeaks_starts,
                width=subpeaks_widths,
                linewidth=1, edgecolor='black', alpha=0.5
                )
        # Plot the STFT results.
        if self.c_plot_options.get_status()[1]:
            if isinstance(self.peak, OriginalPeak):
                subpeaks_all_sections = [self.peak.deconv_peaks_rel]
            else:
                subpeaks_all_sections = []
                for orig_peak in self.peak.orig_peaks:
                    subpeaks_all_sections.append(orig_peak.deconv_peaks_rel)
            subpeaks_names = []
            # Start and end for the different bars to plot (so not only start
            #     of peak, but also center of peak)
            subpeaks_starts = []
            subpeaks_widths = []
            prev_lengths = 0
            for si, subpeaks_section in enumerate(subpeaks_all_sections):
                if isinstance(self.peak, RefinedPeak):
                    shift = (self.peak.orig_peaks[si].chrom_start
                             - self.peak.starts[si])
                else:
                    shift = 0
                for i, (l, c, r) in enumerate(subpeaks_section):
                    if self.c_plot_options.get_status()[6]:
                        subpeaks_names.insert(0, "Subpeak {}".format(i))
                        subpeaks_names.insert(0, "Subpeak {}".format(i))
                        subpeaks_starts.insert(0, prev_lengths + shift + l)
                        subpeaks_starts.insert(0, prev_lengths + shift + c)
                        subpeaks_widths.insert(0, c-l)
                        subpeaks_widths.insert(0, r-c-1)
                    else:
                        subpeaks_names.insert(0, "Subpeak {}".format(i))
                        subpeaks_starts.insert(0, prev_lengths + shift + l)
                        subpeaks_widths.insert(0, r-l-1)
                if isinstance(self.peak, RefinedPeak):
                    prev_lengths += len(self.peak.coverages[si])
            ax_subpeak = self.ax_subpeaks[1]
            ax_subpeak.barh(
                y=subpeaks_names,
                left=subpeaks_starts,
                width=subpeaks_widths,
                linewidth=1, edgecolor='black', alpha=0.5
                )
        # Plot the STFT results with transcript information.
        if (self.c_plot_options.get_status()[2]
                and not isinstance(self.peak, OriginalPeak)):
            subpeaks_names = []
            # Start and end for the different bars to plot (so not only start
            #     of peak, but also center of peak)
            subpeaks_starts = []
            subpeaks_widths = []
            for i, (l, c, r) in enumerate(self.peak.deconv_peaks_rel):
                if self.c_plot_options.get_status()[6]:
                    subpeaks_names.insert(0, "Subpeak {}".format(i))
                    subpeaks_names.insert(0, "Subpeak {}".format(i))
                    subpeaks_starts.insert(0, l)
                    subpeaks_starts.insert(0, c)
                    subpeaks_widths.insert(0, c-l)
                    subpeaks_widths.insert(0, r-c-1)
                else:
                    subpeaks_names.insert(0, "Subpeak {}".format(i))
                    subpeaks_starts.insert(0, l)
                    subpeaks_widths.insert(0, r-l-1)
            ax_subpeak = self.ax_subpeaks[2]
            ax_subpeak.barh(
                y=subpeaks_names,
                left=subpeaks_starts,
                width=subpeaks_widths,
                linewidth=1, edgecolor='black', alpha=0.5
                )

        if self.ax_peak.get_xlim()[0] > xlim[0]:
            self.ax_peak.set_xlim(left=xlim[0])
        if self.ax_peak.get_xlim()[1] < xlim[1]:
            self.ax_peak.set_xlim(right=xlim[1])

    def adjust_peakplot(self):
        """ Adjust the subplots to minimize the margins around the plots. """

        self.fig_peak.tight_layout(pad=0.1)

        width = 0
        if self.c_plot_options.get_status()[4]:
            for l in self.ax_nts_labels:
                width = max(width, l.get_window_extent().width)

        if width > 0:
            self.fig_peak.subplots_adjust(
                right=(1.0
                       - width / (0.9*self.fig_peak.get_window_extent().width)
                       )
                )

        if plt.fignum_exists(self.fig_peak.number):
            self.fig_peak.canvas.draw()

    def print_transcript_statistics(self):
        """ Prints transcript statistics to console. """
        peaks_refined = OrderedDict()
        peaks_unrefined = OrderedDict()
        peaks_unkown = OrderedDict()  # Should be empty
        for p_id, peak in self.peaks_with_tr_info.items():
            if isinstance(peak, RefinedPeak):
                peaks_refined[p_id] = peak
            elif isinstance(peak, UnrefinedPeak):
                peaks_unrefined[p_id] = peak
            else:
                peaks_unkown[p_id] = peak

        orig_peaks_with_transcripts = OrderedDict()
        orig_peaks_with_exons = OrderedDict()
        orig_peaks_refined = OrderedDict()
        for p_id, peak in self.peaks_orig.items():
            if peak.transcripts:
                orig_peaks_with_transcripts[p_id] = peak
            if peak.exons:
                orig_peaks_with_exons[p_id] = peak
            if peak.refined_peak:
                orig_peaks_refined[p_id] = peak

        print("[NOTE] Transcript statistics ... ")
        print("[NOTE] ... len(peaks_orig): ", len(self.peaks_orig))
        print("[NOTE] ... len(orig_peaks_with_transcripts): ",
              len(orig_peaks_with_transcripts))
        print("[NOTE] ... len(orig_peaks_with_exons): ",
              len(orig_peaks_with_exons))
        print("[NOTE] ... len(orig_peaks_refined): ",
              len(orig_peaks_refined))
        print("[NOTE] ... len(orig_peaks_with_transcripts)/len(peaks_orig): ",
              len(orig_peaks_with_transcripts)/len(self.peaks_orig))
        print("[NOTE] ... len(orig_peaks_with_exons)/len(peaks_orig): ",
              len(orig_peaks_with_exons)/len(self.peaks_orig))
        print("[NOTE] ... len(orig_peaks_refined)/len(peaks_orig): ",
              len(orig_peaks_refined)/len(self.peaks_orig))
