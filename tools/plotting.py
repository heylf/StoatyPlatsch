
from distutils.version import StrictVersion
import os

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

import numpy as np

from tools.data import OriginalPeak, RefinedPeak, UnrefinedPeak

# Default size: [6.4, 4.8]
matplotlib.rcParams["figure.figsize"] = [12, 8]


def draw_profile(peak, _fig, ax_rel, paper_plots=False):
    """ Draws the profile plot for the peak on the given figure and axis.

    Parameters
    ----------
    peak : Peak
        The peak data.
    _fig : Figure
        The figure which should be used for drawing the plot. Currently not
        used.
    ax_rel : AxesSubplot
        The axis which should be used for drawing the plot.
    paper_plots : bool (default: False)
        Use specific plot settings when set to True to improve quality when
        embedding the plots into a Latex document.

    Returns
    -------
    lines : list
        The list containing the plot lines created by the plot method.
    """
    x_values = np.arange(len(peak.coverage))
    y_values = peak.coverage
    lines = ax_rel.plot(x_values, y_values,
                        label='{}'.format(peak), linewidth=1, marker='.')
    ax_rel.set_xlabel('Relative Nucleotide Position')
    ax_rel.set_ylabel('Coverage')
    for ax in [ax_rel.xaxis, ax_rel.yaxis]:
        # Ensure that only integers are used for labeling the axes.
        ax.get_major_locator().set_params(integer=True)

    if isinstance(peak, UnrefinedPeak):
        ax_abs = \
            ax_rel.secondary_xaxis(
                location='top',
                functions=(lambda x: x + peak.start + 1,
                           lambda x: x - peak.start - 1)
                )
        ax_abs.set_xticks([peak.start + 1, peak.end])
    elif isinstance(peak, RefinedPeak):
        ax_abs = None
        ranges = np.array(peak.ends) - np.array(peak.starts)
        if len(ranges) > 1:
            peak_segment_end_pos = np.cumsum(ranges)
            for p in peak_segment_end_pos[:-1]:
                ax_rel.axvline(p-0.5, color='grey', lw=1)
    else:
        ax_abs = \
            ax_rel.secondary_xaxis(
                location='top',
                functions=(lambda x: x + peak.chrom_start + 1,
                           lambda x: x - peak.chrom_start - 1)
                )
        ax_abs.set_xticks([peak.chrom_start + 1, peak.chrom_end])

    if ax_abs is not None:
        ax_abs.set_xlabel('Absolute Nucleotide Position',)
        #  Disable scientific notation.
        ax_abs.ticklabel_format(useOffset=False, style='plain')

    if not paper_plots:
        ax_rel.legend()

    return lines


def create_profile_plots(peaks, output_path, output_format='svg',
                         verbose=False, paper_plots=False):
    """ Creates and save the peak profiles for the given peaks.

    Parameters
    ----------
    peaks : OrderedDict
        The dictionary containing the peaks that should be plotted.
    output_path : str
        The folder path where the plots should be saved. Non existing folders
        will be created.
    output_format : str (default: 'svg')
        The file format which should be used for saving the figures. See
        matplotlib documentation for supported values.
    verbose : bool (default: False)
        Print information to console when set to True.
    paper_plots : bool (default: False)
        Use specific plot settings when set to True to improve quality when
        embedding the plots into a Latex document.
    """

    if len(peaks) == 0:
        return

    if verbose:
        print("[NOTE] Create profile plots.")

    os.makedirs(output_path, exist_ok=True)

    for peak in peaks.values():

        if verbose:
            print("[NOTE] ... Create profile plot for peak {}."
                  .format(peak.peak_id))

        fig, ax_rel = plt.subplots()

        draw_profile(peak, fig, ax_rel, paper_plots=paper_plots)

        fig.tight_layout()

        width = len("{}".format(max(peaks)))
        file_path = os.path.join(
            output_path,
            'peak__id_{:0{}d}.{}'.format(peak.peak_id, width, output_format)
            )
        fig.savefig(file_path, format=output_format)

        plt.close(fig)


def create_deconv_profile_figure(peak, paper_plots=False, hide_subpeaks=False):
    """ Creates figure of the peak profile and deconvoluted peaks.

    Parameters
    ----------
    peak : Peak
        The peak data.
    paper_plots : bool (default: False)
        Use specific plot settings when set to True to improve quality when
        embedding the plots into a Latex document.
    hide_subpeaks : bool
        Hide the subpeaks by drawing all in 'white', thus preserving the
        layout. Used for creating the presentation plots.

    Returns
    -------
    fig : Figure
        The figure which was created for drawing the plot.
    axes : list
        List containing the different axes: [ax_rel, ax_subpeaks]
    """
    row_count_subfig1 = 10
    row_count_subfig2 = max(1, len(peak.deconv_peaks_rel))
    row_count = row_count_subfig1 + row_count_subfig2

    figsize = matplotlib.rcParams["figure.figsize"].copy()
    height_per_row = 2/3 * figsize[1] / row_count_subfig1
    figsize[1] = row_count * height_per_row

    fig = plt.figure(figsize=figsize, constrained_layout=True)

    gs = fig.add_gridspec(row_count+1, 1)    # +1 for additional space
    if StrictVersion(matplotlib.__version__) >= StrictVersion('3.3.0'):
        # Workaround for a bug in newer matplotlib version (only tested with
        # version 3.3.3; workaround was not necessary in version 3.2.2).
        # Creating a subplot on multiple rows of the GridSpec as first subplot
        # resulted in errors when applying constrained_layout. Creating an
        # invisible and unused Axes on a single row first solves this issue.
        ax_tmp = fig.add_subplot(gs[0, :])
        ax_tmp.set_visible(False)

    ax_rel = fig.add_subplot(gs[:row_count_subfig1, :])
    ax_subpeaks = fig.add_subplot(gs[-row_count_subfig2:, :])

    if hide_subpeaks:
        color = 'white'
        for spine in ['bottom', 'left', 'top', 'right']:
            ax_subpeaks.spines[spine].set_color('white')
            ax_subpeaks.tick_params(axis='x', colors='white')
            ax_subpeaks.tick_params(axis='y', colors='white')
            ax_subpeaks.xaxis.label.set_color('white')
            ax_subpeaks.yaxis.label.set_color('white')
    else:
        color = 'black'

    draw_profile(peak, fig, ax_rel, paper_plots=paper_plots)

    subpeaks_names = np.empty(len(peak.deconv_peaks_rel), dtype=object)
    subpeaks_left = np.empty(len(peak.deconv_peaks_rel), dtype=int)
    subpeaks_center = np.empty(len(peak.deconv_peaks_rel), dtype=int)
    subpeaks_right = np.empty(len(peak.deconv_peaks_rel), dtype=int)

    # Mark the found deconvoluted peaks below plot. Transform the data
    # into a format that can be used by the matplotlibs bar plot.
    for subpeak, (left, center, right) in enumerate(peak.deconv_peaks_rel):

        subpeaks_names[-(subpeak+1)] = "Subpeak {}".format(subpeak)
        subpeaks_left[-(subpeak+1)] = left
        subpeaks_center[-(subpeak+1)] = center
        subpeaks_right[-(subpeak+1)] = right - 1  # Chrom end is
        #                                         # non-inclusive

    if isinstance(peak, (OriginalPeak, UnrefinedPeak)):
        ax_subpeaks.set_xlabel('Absolute Nucleotide Position', color=color)
        if isinstance(peak, OriginalPeak):
            start = peak.chrom_start
        else:
            start = peak.start
        start += 1
    else:
        start = 0
        ax_subpeaks.set_xlabel('Relative Nucleotide Position', color=color)

#     # Extend range of x-axis if peak is outside current axis.
    if (len(subpeaks_right) > 0  # Ensure that list is not empty to prevent
        # errors.
            and max(subpeaks_right) > ax_rel.get_xlim()[1]):
        ax_rel.set_xlim(ax_rel.get_xlim()[0], max(subpeaks_right) + 1)
    if (len(subpeaks_left) > 0  # Ensure that list is not empty to prevent
        # errors.
            and min(subpeaks_left) < ax_rel.get_xlim()[0]):
        ax_rel.set_xlim(min(subpeaks_left) - 1, ax_rel.get_xlim()[1])

    ax_subpeaks.set_xlim((ax_rel.get_xlim()[0] + start,
                          ax_rel.get_xlim()[1] + start))
    ax_subpeaks.barh(y=subpeaks_names,
                     width=subpeaks_center - subpeaks_left,
                     left=subpeaks_left + start,
                     color="C0" if not hide_subpeaks else 'white',
                     linewidth=1, edgecolor=color)
    ax_subpeaks.barh(y=subpeaks_names,
                     width=subpeaks_right - subpeaks_center,
                     left=subpeaks_center + start,
                     color="C0" if not hide_subpeaks else 'white',
                     linewidth=1, edgecolor=color)

    # Disable scientific notation. Has to be done here differently as the
    # previous approach does not work with bar plots.
    def formatter_disable_scientific(x, _pos):
        return "{:.0f}".format(x)
    ax_subpeaks.xaxis.set_major_formatter(
        FuncFormatter(formatter_disable_scientific)
        )

    return fig, [ax_rel, ax_subpeaks]


def create_deconv_profile_plots(peaks, output_path, output_format='svg',
                                verbose=False, paper_plots=False):
    """ Creates and saves the peak profiles and deconvoluted peaks.

    Parameters
    ----------
    peaks : dict
        The dictionary containing the peaks that should be plotted.
    output_path : str
        The folder path where the plots should be saved. Non existing folders
        will be created.
    output_format : str (default: 'svg')
        The file format which should be used for saving the figures. See
        matplotlib documentation for supported values.
    verbose : bool (default: False)
        Print information to console when set to True.
    paper_plots : bool (default: False)
        Use specific plot settings when set to True to improve quality when
        embedding the plots into a Latex document.
    """

    if len(peaks) == 0:
        return

    if verbose:
        print("[NOTE] Create profile plots with deconvoluted peaks.")

    os.makedirs(output_path, exist_ok=True)

    for peak in peaks.values():

        if verbose:
            print("[NOTE] ... Create plots for peak {}.".format(peak.peak_id))

        fig, [_ax_rel, _ax_subpeaks] = \
            create_deconv_profile_figure(peak, paper_plots=paper_plots)
        width = len("{}".format(max(peaks)))
        file_path = os.path.join(
            output_path,
            'deconv__peak_id_{:0{}d}.{}'.format(peak.peak_id,
                                                width, output_format)
            )
        fig.savefig(file_path, format=output_format)
        plt.close(fig)
