
import os

import matplotlib.pyplot as plt
import numpy as np


def create_FFT_plots(peaks, output_path, output_format='svg'):
    """ TODO: Add text

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
    """

    if len(peaks) == 0:
        return

    os.makedirs(output_path, exist_ok=True)

    for p_id in sorted(peaks.keys()):
        peak = peaks[p_id]

        for filter_index in [0, 1, 2, 3, 4, 5, 10]:
            # sorted(peak.fft.f_filtered.keys()):

            if filter_index not in peak.fft.f_filtered:
                break

            fig, ax_rel = plt.subplots()
            ax_rel.plot(np.arange(-peak.fft.num_padding[0],
                                  len(peak.fft.f) - peak.fft.num_padding[0]
                                  ),
                        peak.fft.f,
                        label='{} - Peak #{}'.format(peak.chrom,
                                                     peak.peak_number),
                        linewidth=1, marker='.')
            # print("p_id: ", p_id, " ; filter_index: ", filter_index)
            # print("len(x): ", len(x))
            # print("len(y): ", len(y))
            ax_rel.plot(np.arange(-peak.fft.num_padding[0],
                                  len(peak.fft.f) - peak.fft.num_padding[0]
                                  ),
                        peak.fft.f_filtered[filter_index],
                        label='filter index {}'.format(filter_index),
                        linewidth=1, marker='.')

            ax_rel.set_xlabel('Relative Nucleotide Position')
            ax_rel.set_ylabel('Intensity')
            for ax in [ax_rel.xaxis, ax_rel.yaxis]:
                # Ensure that only integers are used for labeling the axes.
                ax.get_major_locator().set_params(integer=True)

            ax_rel.axvline(0, color='grey', lw=1)
            ax_rel.axvline(len(peak.fft.f) - peak.fft.num_padding[0]
                                           - peak.fft.num_padding[1] - 1,
                           color='grey', lw=1)

            ax_abs = \
                ax_rel.secondary_xaxis(
                    location='top',
                    functions=(lambda x: x + peak.chrom_start,
                               lambda x: x - peak.chrom_start)
                    )
            ax_abs.set_xlabel('Absolute Nucleotide Position',)
            ax_abs.set_xticks([peak.chrom_start, peak.chrom_end - 1])
            #  Disable scientific notation.
            ax_abs.ticklabel_format(useOffset=False, style='plain')

            fig.legend()

            width = len("{}".format(max(peaks)))
            file_path = os.path.join(
                output_path,
                'peak__id_{:0{}d}__filter_index_{}.{}'
                .format(p_id, width, filter_index, output_format)
                )
            fig.savefig(file_path, format=output_format)

            plt.close(fig)


def create_profile_plots(peaks, output_path, output_format='svg'):
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
    """

    if len(peaks) == 0:
        return

    os.makedirs(output_path, exist_ok=True)

    for p_id in sorted(peaks.keys()):
        peak = peaks[p_id]

        fig, ax_rel = plt.subplots()
        ax_rel.plot(np.arange(len(peak.coverage)), peak.coverage,
                    label='{} - Peak #{}'.format(peak.chrom, peak.peak_number),
                    linewidth=1, marker='.')
        ax_rel.set_xlabel('Relative Nucleotide Position')
        ax_rel.set_ylabel('Intensity')
        for ax in [ax_rel.xaxis, ax_rel.yaxis]:
            # Ensure that only integers are used for labeling the axes.
            ax.get_major_locator().set_params(integer=True)

        ax_abs = \
            ax_rel.secondary_xaxis(location='top',
                                   functions=(lambda x: x + peak.chrom_start,
                                              lambda x: x - peak.chrom_start)
                                   )
        ax_abs.set_xlabel('Absolute Nucleotide Position',)
        ax_abs.set_xticks([peak.chrom_start, peak.chrom_end - 1])
        #  Disable scientific notation.
        ax_abs.ticklabel_format(useOffset=False, style='plain')

        fig.legend()

        width = len("{}".format(max(peaks)))
        file_path = os.path.join(
            output_path,
            'peak__id_{:0{}d}.{}'.format(p_id, width, output_format)
            )
        fig.savefig(file_path, format=output_format)

        plt.close(fig)
