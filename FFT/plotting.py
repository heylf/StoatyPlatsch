
import os

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

import numpy as np

# Default size: [6.4, 4.8]
matplotlib.rcParams["figure.figsize"] = [12, 8]


def draw_profile(peak, _fig, ax_rel, fft_applied=False, paper_plots=False):
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
    fft_applied : bool (default: False)
        True when the profile plot should be drawn for a peak that was already
        processed with FFT.
    paper_plots : bool (default: False)
        Use specific plot settings when set to True to improve quality when
        embedding the plots into a Latex document.

    Returns
    -------
    lines : list
        The list containing the plot lines created by the plot method.
    """
    if fft_applied:
        x_values = np.arange(-peak.fft.num_padding[0],
                             len(peak.fft.f) - peak.fft.num_padding[0])
        y_values = peak.fft.f
    else:
        x_values = np.arange(len(peak.coverage))
        y_values = peak.coverage
    lines = ax_rel.plot(x_values, y_values,
                        label='{} - Peak #{}'.format(peak.chrom,
                                                     peak.peak_number),
                        linewidth=1, marker='.')
    ax_rel.set_xlabel('Relative Nucleotide Position')
    ax_rel.set_ylabel('Intensity')
    for ax in [ax_rel.xaxis, ax_rel.yaxis]:
        # Ensure that only integers are used for labeling the axes.
        ax.get_major_locator().set_params(integer=True)

    if fft_applied:
        ax_rel.axvline(0, color='grey', lw=1)
        ax_rel.axvline(len(peak.fft.f) - peak.fft.num_padding[0]
                                       - peak.fft.num_padding[1] - 1,
                       color='grey', lw=1)

    ax_abs = \
        ax_rel.secondary_xaxis(location='top',
                               functions=(lambda x: x + peak.chrom_start,
                                          lambda x: x - peak.chrom_start)
                               )
    ax_abs.set_xlabel('Absolute Nucleotide Position',)
    ax_abs.set_xticks([peak.chrom_start, peak.chrom_end - 1])
    #  Disable scientific notation.
    ax_abs.ticklabel_format(useOffset=False, style='plain')

    if not paper_plots:
        ax_rel.legend()

    return lines


def create_deconv_profile_figure(peak, paper_plots=False):
    """ Creates figure of the peak profile and deconvoluted peaks.

    Parameters
    ----------
    peak : Peak
        The peak data.
    paper_plots : bool (default: False)
        Use specific plot settings when set to True to improve quality when
        embedding the plots into a Latex document.

    Returns
    -------
    fig : Figure
        The figure which was created for drawing the plot.
    axes : list
        List containing the different axes: [ax_rel, ax_subpeaks]
    """
    row_count_subfig1 = 10
    row_count_subfig2 = max(1, len(peak.fft.new_peaks))
    row_count = row_count_subfig1 + row_count_subfig2

    figsize = matplotlib.rcParams["figure.figsize"].copy()
    height_per_row = 2/3 * figsize[1] / row_count_subfig1
    figsize[1] = row_count * height_per_row

    fig = plt.figure(figsize=figsize, constrained_layout=True)

    gs = fig.add_gridspec(row_count+1, 1)    # +1 for additional space
    ax_rel = fig.add_subplot(gs[:row_count_subfig1, :])
    ax_subpeaks = fig.add_subplot(gs[-row_count_subfig2:, :])

    draw_profile(peak, fig, ax_rel, fft_applied=True, paper_plots=paper_plots)

    subpeaks_names = np.empty(len(peak.fft.new_peaks), dtype=object)
    subpeaks_left = np.empty(len(peak.fft.new_peaks), dtype=int)
    subpeaks_center = np.empty(len(peak.fft.new_peaks), dtype=int)
    subpeaks_right = np.empty(len(peak.fft.new_peaks), dtype=int)

    # Mark the found deconvoluted peaks below plot. Transform the data
    # into a format that can be used by the matplotlibs bar plot.
    for subpeak, (left, center, right) in enumerate(peak.fft.new_peaks):

        subpeaks_names[-(subpeak+1)] = "Subpeak {}".format(subpeak)
        subpeaks_left[-(subpeak+1)] = left
        subpeaks_center[-(subpeak+1)] = center
        subpeaks_right[-(subpeak+1)] = right - 1  # Chrom end is non-inclusive

    # Extend range of x-axis if peak is outside current axis.
    if (len(subpeaks_right) > 0  # Ensure that list is not empty to prevent
        # errors.
            and max(subpeaks_right) > ax_rel.get_xlim()[1] + peak.chrom_start):
        ax_rel.set_xlim(ax_rel.get_xlim()[0],
                        max(subpeaks_right) + 1 - peak.chrom_start)
    if (len(subpeaks_left) > 0  # Ensure that list is not empty to prevent
        # errors.
            and min(subpeaks_left) < ax_rel.get_xlim()[0] + peak.chrom_start):
        ax_rel.set_xlim(min(subpeaks_left) - 1 - peak.chrom_start,
                        ax_rel.get_xlim()[1])

    ax_subpeaks.set_xlim((ax_rel.get_xlim()[0] + peak.chrom_start,
                          ax_rel.get_xlim()[1] + peak.chrom_start))
    ax_subpeaks.barh(y=subpeaks_names,
                     width=subpeaks_center - subpeaks_left,
                     left=subpeaks_left,
                     color="C0", linewidth=1, edgecolor="black")
    ax_subpeaks.barh(y=subpeaks_names,
                     width=subpeaks_right - subpeaks_center,
                     left=subpeaks_center,
                     color="C0", linewidth=1, edgecolor="black")
    ax_subpeaks.set_xlabel('Absolute Nucleotide Position',)

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

    for p_id in sorted(peaks.keys()):
        peak = peaks[p_id]

        if verbose:
            print("[NOTE] ... Create plots for peak {}.".format(p_id))

        fig, [_ax_rel, _ax_subpeaks] = \
            create_deconv_profile_figure(peak, paper_plots=paper_plots)
        width = len("{}".format(max(peaks)))
        file_path = os.path.join(
            output_path,
            'deconv__peak_id_{:0{}d}.{}'.format(p_id, width, output_format)
            )
        fig.savefig(file_path, format=output_format)
        plt.close(fig)

        # Create plot with additional information, depending on the used
        # FFT approach.
        fig, [ax_rel, _ax_subpeaks] = \
            create_deconv_profile_figure(peak, paper_plots=paper_plots)
        xrange = np.arange(-peak.fft.num_padding[0],
                           len(peak.fft.f) - peak.fft.num_padding[0])
        if peak.fft.approach == 'smooth':
            # Add the smoothed profile.
            ax_rel.plot(xrange, peak.fft.filtered_f,
                        label='smoothed profile', linewidth=1, marker='.')

            # Mark the maxima of the smoothed profile which were used for
            # defining the peak centers.
            ax_rel.plot(peak.fft.local_maxs - peak.fft.num_padding[0],
                        peak.fft.filtered_f[peak.fft.local_maxs],
                        "o", markersize=8, fillstyle='none', color="red")
        else:
            if peak.fft.disable_frequency_shift:
                # Add the considered frequencies of the FFT.
                for idx_freq, frequency in sorted(
                            peak.fft.frequencies.items()
                        ):
                    ax_rel.plot(
                        np.arange(-peak.fft.num_padding[0],
                                  len(peak.fft.f) - peak.fft.num_padding[0]
                                  ),
                        frequency,
                        label='frequency {}'.format(idx_freq)
                        )
            else:
                # Add the actual used, shifted frequencies.
                for mapping in peak.fft.mappings:
                    ax_rel.plot(xrange, mapping.shifted_freq,
                                label='frequency {} (shifted)'
                                      .format(mapping.idx_freq))
                # Add the considered (original) frequencies of the FFT.
                for idx_freq, frequency in sorted(
                            peak.fft.frequencies.items()
                        ):
                    ax_rel.plot(
                        xrange,  frequency,
                        label='frequency {} (original)'.format(idx_freq),
                        linestyle='--')

            # Mark the maxima of the original profile which were used for
            # mapping them to the signal or vice versa.
            ax_rel.plot(peak.fft.local_maxs - peak.fft.num_padding[0],
                        peak.fft.f[peak.fft.local_maxs],
                        "o", markersize=8, fillstyle='none', color="red")

        if not paper_plots:
            ax_rel.legend()
        width = len("{}".format(max(peaks)))
        file_path = os.path.join(
            output_path,
            'deconv__with_FFT_info__peak_id_{:0{}d}.{}'.format(p_id, width,
                                                               output_format)
            )
        fig.savefig(file_path, format=output_format)
        plt.close(fig)


def create_FFT_analysis_plots(peaks, output_path, output_format='svg',
                              plot_fft_values=True,
                              plot_fft_transformations=True,
                              verbose=False, paper_plots=False):
    """ Creates and saves plots for analyzing the FFT approach.

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
    plot_fft_values : bool (default: True)
        Enables plotting different FFT values.
    plot_fft_transformations : bool (default: True)
        Enables plotting different FFT transformation and filter results.
    verbose : bool (default: False)
        Print information to console when set to True.
    paper_plots : bool (default: False)
        Use specific plot settings when set to True to improve quality when
        embedding the plots into a Latex document.
    """

    if ((len(peaks) == 0)
            or ((not plot_fft_values) and (not plot_fft_transformations))):
        return

    if verbose:
        print("[NOTE] Create FFT analysis plots.")

    os.makedirs(output_path, exist_ok=True)

    for p_id in sorted(peaks.keys()):
        peak = peaks[p_id]

        if verbose:
            print("[NOTE] ... Create plots for peak {}.".format(p_id))

        print("[NOTE] ... ... Create preprocessed profile plot.")

        fig, ax_rel = plt.subplots()
        draw_profile(peak, fig, ax_rel, fft_applied=True,
                     paper_plots=paper_plots)
        fig.tight_layout()
        width = len("{}".format(max(peaks)))
        file_path = os.path.join(
            output_path,
            'preprocessed_peak__id_{:0{}d}.{}'.format(p_id, width,
                                                      output_format)
            )
        fig.savefig(file_path, format=output_format)
        plt.close(fig)

        if plot_fft_values:
            if verbose:
                print("[NOTE] ... ... Create FFT value plots.")

            output_path_sub = os.path.join(output_path, 'FFT_values')
            os.makedirs(output_path_sub, exist_ok=True)

            for f, label, output_label, log_scale in \
                [  # FFT without norm
                 (peak.fft.fhat_abs, 'fhat - Abs',
                  'fft_fhat_01_abs__default', False),
                 (peak.fft.fhat_abs, 'fhat - Abs (log)',
                  'fft_fhat_02_abs__log', True),
                 (peak.fft.fhat_re, 'fhat - Re',
                  'fft_fhat_03_re__default', False),
                 (peak.fft.fhat_re, 'fhat - Re (log)',
                  'fft_fhat_04_re__log', True),
                 (peak.fft.fhat_im, 'fhat - Im',
                  'fft_fhat_05_im__default', False),
                 (peak.fft.fhat_im, 'fhat - Im (log)',
                  'fft_fhat_06_im__log', True),
                 (peak.fft.fhat_power_spectrum, 'power spectrum',
                  'fft_fhat_07_power_spectrum__default', False),
                 (peak.fft.fhat_power_spectrum, 'power spectrum (log)',
                  'fft_fhat_08_power_spectrum__log', True),
                 (peak.fft.fhat_phase_spectrum, 'phase spectrum',
                  'fft_fhat_09_phase_spectrum__default', False),
                 (peak.fft.fhat_phase_spectrum, 'phase spectrum (log)',
                  'fft_fhat_10_phase_spectrum__log', True),
                 # FFT with norm 'ortho'
                 (peak.fft.fhat_norm_abs, 'normed fhat - Abs',
                  'fft_fhatnorm_01_abs__default', False),
                 (peak.fft.fhat_norm_abs, 'normed fhat - Abs (log)',
                  'fft_fhatnorm_02_abs__log', True),
                 (peak.fft.fhat_norm_re, 'normed fhat - Re',
                  'fft_fhatnorm_03_re__default', False),
                 (peak.fft.fhat_norm_re, 'normed fhat - Re (log)',
                  'fft_fhatnorm_04_re__log', True),
                 (peak.fft.fhat_norm_im, 'normed fhat - Im',
                  'fft_fhatnorm_05_im__default', False),
                 (peak.fft.fhat_norm_im, 'normed fhat - Im (log)',
                  'fft_fhatnorm_06_im__log', True),
                 (peak.fft.fhat_norm_power_spectrum, 'normed power spectrum',
                  'fft_fhatnorm_07_power_spectrum__default', False),
                 (peak.fft.fhat_norm_power_spectrum,
                  'normed power spectrum (log)',
                  'fft_fhatnorm_08_power_spectrum__log', True),
                 (peak.fft.fhat_norm_phase_spectrum, 'normed phase spectrum',
                  'fft_fhatnorm_09_phase_spectrum__default', False),
                 (peak.fft.fhat_norm_phase_spectrum,
                  'normed phase spectrum (log)',
                  'fft_fhatnorm_10_phase_spectrum__log', True),
                 ]:

                fig, ax = plt.subplots()
                ax.plot(peak.fft.freq, f, linewidth=1, marker='.', label=label)
                if log_scale:
                    ax.set_yscale('symlog')
                ax.set_title('{} - Peak #{}'.format(peak.chrom,
                                                    peak.peak_number))
                ax.legend()
                fig.tight_layout()

                width = len("{}".format(max(peaks)))
                file_path = os.path.join(
                    output_path_sub,
                    '{}__peak__id_{:0{}d}.{}'
                    .format(output_label, p_id, width,
                            output_format)
                    )
                fig.savefig(file_path, format=output_format)
                plt.close(fig)

        if plot_fft_transformations:
            if verbose:
                print("[NOTE] ... ... Create FFT transformation and filter"
                      " plots.")

            output_path_sub = os.path.join(output_path, 'FFT_transforms')
            os.makedirs(output_path_sub, exist_ok=True)

            filter_ids_to_plot = [0, 1, 2, 3, 4, 5, 10]
            # filter_ids_to_plot = sorted(peak.fft.f_filtered.keys())
            for filter_index in filter_ids_to_plot:
                if filter_index not in peak.fft.filtered_f:
                    break

                fig, ax_rel = plt.subplots()

                draw_profile(peak, fig, ax_rel, fft_applied=True,
                             paper_plots=paper_plots)

                ax_rel.plot(np.arange(-peak.fft.num_padding[0],
                                      len(peak.fft.f) - peak.fft.num_padding[0]
                                      ),
                            peak.fft.filtered_f[filter_index],
                            label='filter index {}'.format(filter_index),
                            linewidth=1, marker='.')
                ax_rel.legend()
                fig.tight_layout()

                width = len("{}".format(max(peaks)))
                file_path = os.path.join(
                    output_path_sub,
                    'peak__id_{:0{}d}__filter_index_{}.{}'
                    .format(p_id, width, filter_index, output_format)
                    )
                fig.savefig(file_path, format=output_format)
                plt.close(fig)

            output_path_sub = os.path.join(output_path, 'FFT_frequencies')
            os.makedirs(output_path_sub, exist_ok=True)

            fig, axis = plt.subplots()
            line, = draw_profile(peak, fig, axis, fft_applied=True,
                                 paper_plots=paper_plots)
            line.set_label(None)
            # for i in np.arange(len(peak.fft.fhat)):
            for i in np.arange(min(len(peak.fft.fhat),
                                   10 if paper_plots else 20)):
                indices = np.zeros(len(peak.fft.fhat))
                indices[i] = 1
                f_reverse_filtered = np.fft.irfft(indices * peak.fft.fhat,
                                                  peak.fft.n)
                axis.plot(np.arange(-peak.fft.num_padding[0],
                                    len(peak.fft.f) - peak.fft.num_padding[0]
                                    ),
                          f_reverse_filtered,
                          label='frequency {}'.format(i))
            fig.legend()
            plt.tight_layout(rect=[0, 0, 0.85, 1])

            width = len("{}".format(max(peaks)))
            file_path = os.path.join(
                output_path_sub,
                'fft__frequencies__peak__id_{:0{}d}.{}'
                .format(p_id, width, output_format)
                )
            fig.savefig(file_path, format=output_format)
            plt.close(fig)


def create_profile_plots(peaks, output_path, output_format='svg',
                         verbose=False, paper_plots=False):
    """ Creates and save the peak profiles for the given peaks.

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
        print("[NOTE] Create profile plots.")

    os.makedirs(output_path, exist_ok=True)

    for p_id in sorted(peaks.keys()):
        peak = peaks[p_id]

        if verbose:
            print("[NOTE] ... Create profile plot for peak {}.".format(p_id))

        fig, ax_rel = plt.subplots()

        draw_profile(peak, fig, ax_rel, paper_plots=paper_plots)

        fig.tight_layout()

        width = len("{}".format(max(peaks)))
        file_path = os.path.join(
            output_path,
            'peak__id_{:0{}d}.{}'.format(p_id, width, output_format)
            )
        fig.savefig(file_path, format=output_format)

        plt.close(fig)
