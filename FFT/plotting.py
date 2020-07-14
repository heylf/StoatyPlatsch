
import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np


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

    os.makedirs(output_path, exist_ok=True)

    for p in sorted(peaks.keys()):
        peak = peaks[p]

        fig, ax_rel = plt.subplots()
        ax_rel.plot(np.arange(len(peak.coverage)), peak.coverage,
                    label='Peak #{}'.format(peak.peak_number),
                    linewidth=1,
                    marker='.')
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

        width = len("{}".format(len(peaks)))
        file_path = '{}/peak_{:0{}d}.{}'.format(output_path, peak.peak_number,
                                                width, output_format)
        fig.savefig(file_path, format=output_format)

        plt.close(fig)
