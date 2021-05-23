
import matplotlib.pyplot as plt

from STFT.plotting import PeakAnalyzer
from tools.preprocessing import read_coverage_file


if __name__ == '__main__':
    input_coverage_file = '../../../Data/02_Preprocessing/01__HepG2_against_RBFOX2/RBFOX2__02_01__full_7477_peaks__Rep1__coverage.tsv'    # @IgnorePep8
    # input_coverage_file = '../../Data/02_Preprocessing/01__HepG2_against_RBFOX2/RBFOX2__02_01__full_7477_peaks__Rep1__coverage.tsv'    # @IgnorePep8
    verbose = True
    peak_ID = 3171
    peaks = read_coverage_file(input_coverage_file=input_coverage_file,
                               verbose=verbose)
    pa = PeakAnalyzer(peaks=peaks, init_peak_ID=peak_ID,
                      verbose=verbose, create_additional_plots=False,
                      postpone_show=True)
    pa.r_detrend.set_active(0)
    pa.s_nperseg.set_val(20)
    pa.r_plot_overview.set_active(1)
    pa.s_plot_overview_val.set_val(1)

    for i_segment in range(4, pa.stft_result.Zxx.shape[1]-1, 3):
        for i_freq in range(pa.stft_result.Zxx.shape[0]):
                pa.plot_istft_result(i_freq, i_segment)

    # import IPython
    # IPython.embed()
    # print("pa.fig_freq.get_figheight(): ", pa.fig_freq.get_figheight())
    # print("pa.fig_freq.get_figwidth(): ", pa.fig_freq.get_figwidth())
    pa.fig_freq.set_figwidth(9)
    # print("pa.fig_freq.get_figheight(): ", pa.fig_freq.get_figheight())
    # print("pa.fig_freq.get_figwidth(): ", pa.fig_freq.get_figwidth())

    plt.show()
