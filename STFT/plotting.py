
import copy

import matplotlib.pyplot as plt
from matplotlib.widgets import CheckButtons, RadioButtons, Slider, TextBox
import numpy as np
import scipy.fft
import scipy.signal

from FFT.processing import deconvolute_with_FFT
from STFT.processing import deconvolute_peak_with_STFT
from tools.data import OriginalPeak
from tools.plotting import draw_profile


class Result(object):
    """ Dummy class for storing result values. """
    pass


class PeakAnalyzer(object):
    """ Visualizes the STFT for given peaks under different parameters.

    Attributes
    ----------
    peaks : OrderedDict
        The dictionary containing the peaks that should be deconvoluted.
    peak : None, OriginalPeak, RefinedPeak, UnrefinedPeak
        The current selected and plotted peak.
    verbose : bool (default: False)
        Print information to console when set to True.
    resetting_stft_values : bool
        True if the STFT parameter values are currently changed. Used to
        prevent recalculations before all parameters have been changed.
    resetting_plot_values : bool
        True if the plot parameter values are currently changed. Used to
        prevent plotting before all parameters have been changed.
    padding : tuple
        Contains the padding values for the left and the right side.
    stft_args : dict
        Holds the parameter used for the calculation of the STFT.

    Various other attributes as 'fig_controls' or 'ax_stft_text', all storing
    the different figures and axes.
    """

    def __init__(self, peaks, init_peak_ID=None, verbose=False,
                 create_additional_plots=False, postpone_show=False):
        """ Constructor

        Parameters
        ----------
        init_peak_ID : int (default: None)
            Sets the peak that should be plotted initially.
        create_additional_plots : bool (default: False)
            If True, create additional plots.
        postpone_show : bool (default: False)
            Do not show plots directly, if set to True.

        See attributes description of the class for other parameters.
        """
        self.peaks = peaks
        self.peak = None
        self.verbose = verbose

        self._create_fig_controls()
        self._create_fig_STFT_freq_3D()
        if create_additional_plots:
            self._create_additional_fig()

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

    def _create_fig_controls(self):
        """ Creates the control figure and control elements. """
        self.fig_controls = plt.figure("Controls", figsize=[7.0, 8.0])
        self.fig_controls.subplots_adjust(left=0.1, bottom=0.0, right=1.0,
                                          top=0.94, wspace=0.0, hspace=0.03)
        row_count = 29
        col_count = 16
        col_count_controls = 4
        gs = self.fig_controls.add_gridspec(row_count, col_count)

        # Create axes for STFT parameters.
        self.ax_stft_text = \
            self.fig_controls.add_subplot(gs[0, :col_count_controls])
        self.ax_s_peakid = \
            self.fig_controls.add_subplot(gs[1, :col_count_controls-2])
        self.ax_tb_peakid = \
            self.fig_controls.add_subplot(
                gs[1, col_count_controls-1:col_count_controls]
                )
        self.ax_window = \
            self.fig_controls.add_subplot(gs[3, :col_count_controls])
        self.ax_nperseg = \
            self.fig_controls.add_subplot(gs[5, :col_count_controls])
        self.ax_padding = \
            self.fig_controls.add_subplot(gs[7, :col_count_controls])
        self.ax_noverlap = \
            self.fig_controls.add_subplot(gs[9, :col_count_controls])
        self.ax_detrend = \
            self.fig_controls.add_subplot(gs[11:13, :col_count_controls-1])

        # Create axes for plotting parameters.
        self.ax_plot_text = \
            self.fig_controls.add_subplot(gs[15, :col_count_controls])
        self.ax_plot_overview = \
            self.fig_controls.add_subplot(gs[16:18, :col_count_controls-1])
        self.ax_freq_option_val = \
            self.fig_controls.add_subplot(gs[19, :col_count_controls-1])
        self.ax_plot_options_r = self.fig_controls.add_subplot(
                gs[21:24, :np.ceil(col_count_controls*2.5).astype(int)])
        self.ax_plot_options_c = self.fig_controls.add_subplot(
                gs[25:28, :np.ceil(col_count_controls*3.5).astype(int)]
                )

        # Create axis for plotting the peak profile.
        profile_end_row = row_count//2
        self.ax_profile = \
            self.fig_controls.add_subplot(gs[1:profile_end_row,
                                             col_count_controls+2:-1])
        self.ax_text_info = self.fig_controls.add_subplot(
            gs[profile_end_row+4:profile_end_row+6, col_count_controls+4:]
            )

        # Create widgets for STFT parameters.
        self.ax_stft_text.set_axis_off()
        self.ax_stft_text.text(-0.3, 1.0, 'STFT parameters', weight='bold')
        self.s_peakid = Slider(self.ax_s_peakid, 'Peak ID',
                               0, len(self.peaks)-1,
                               valinit=16,
                               valstep=1)
        self.s_peakid.on_changed(self.s_peakid_changed)
        self.tb_peakid = TextBox(self.ax_tb_peakid, '', self.s_peakid.val)
        self.reset_textbox_events(self.tb_peakid)
        self.tb_peakid.on_submit(self.tb_peakid_on_submit)

        self.tb_window = TextBox(self.ax_window, 'window', 'boxcar')
        self.reset_textbox_events(self.tb_window)
        self.tb_window.on_submit(self.tb_window_on_submit)
        self.s_nperseg = Slider(self.ax_nperseg, 'nperseg',
                                1, 2, valinit=1, valstep=1)
        self.s_nperseg.on_changed(self.s_nperseg_changed)
        self.s_padding = Slider(self.ax_padding, 'padding',
                                0, 2, valinit=1, valstep=1)
        self.s_padding.on_changed(self.s_padding_changed)
        self.s_noverlap = Slider(self.ax_noverlap, 'noverlap',
                                 0, 1, valinit=1, valstep=1)
        self.s_noverlap.on_changed(self.s_noverlap_changed)
        self.r_detrend = RadioButtons(self.ax_detrend,
                                      (False, 'constant', 'linear'),
                                      active=1)
        self.r_detrend.on_clicked(self.r_detrend_on_clicked)

        # Create widgets for plotting parameters.
        self.ax_plot_text.set_axis_off()
        self.ax_plot_text.text(-0.3, 1.0, 'Plotting parameters', weight='bold')
        self.r_plot_overview = RadioButtons(self.ax_plot_overview,
                                            ('All', 'Segment', 'Frequency'))
        self.r_plot_overview.on_clicked(self.r_plot_overview_changed)
        self.s_plot_overview_val = Slider(self.ax_freq_option_val,
                                          '',
                                          0, 1, valinit=0, valstep=1)
        self.s_plot_overview_val.on_changed(self.s_plot_overview_val_changed)
        self.s_plot_overview_val.set_active(False)
        self.s_plot_overview_val.ax.set_visible(False)

        self.r_plot_options = RadioButtons(
            self.ax_plot_options_r,
            ('Default',
             'Propagate segment values',
             'Include neighboring segment values',
             'Include all segment values')
            )
        self.r_plot_options.on_clicked(self.r_plot_options_changed)

        self.c_plot_options = CheckButtons(
            self.ax_plot_options_c,
            ['Limit plot range to segment',
             'Plot FFT frequencies (only for plot option Segment)',
             'Plot FFT deconv results (only w/o transcript information)',
             'Plot STFT deconv results'],
            [True, False, False, False]
            )
        self.c_plot_options.on_clicked(self.c_plot_options_clicked)

        self.ax_text_info.set_axis_off()
        self.text_info = self.ax_text_info.text(-0.3, 0.5, 'Info: -')

        self.resetting_stft_values = False
        self.resetting_plot_values = False

    def _create_fig_STFT_freq_3D(self):
        """ Creates the figure for plotting the STFT frequencies in 3D. """
        self.fig_freq = plt.figure("STFT Frequencies")
        self.ax_freq = self.fig_freq.add_subplot(projection='3d')
        self.fig_freq.subplots_adjust(left=0, bottom=0.0, right=1.0, top=1.0)

    def _create_additional_fig(self):
        self.fig_colormesh_abs_Zxx = plt.figure("Colormesh abs(Zxx)")
        self.ax_colormesh_abs_Zxx = self.fig_colormesh_abs_Zxx.add_subplot()

        self.fig_surface_abs_Zxx = plt.figure("Surface abs(Zxx)")
        self.ax_surface_abs_Zxx = \
            self.fig_surface_abs_Zxx.add_subplot(projection='3d')

        self.fig_spectogram = plt.figure("Spectogram")
        self.ax_spectogram = self.fig_spectogram.add_subplot()

    def s_peakid_changed(self, val):
        """ Handles the slider event for a changed peak_id.

        Parameters
        ----------
        val : int
            The new peak_id value.
        """
        if val < 0:
            self.s_peakid.set_val(0)
            return
        elif val > self.s_peakid.valmax:
            self.s_peakid.set_val(self.s_peakid.valmax)
            return
        elif self.peak != self.peaks[val]:
            self.peak = self.peaks[val]
            self.tb_peakid.set_val(val)
            self.peakid_changed()

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

    def peakid_changed(self):
        """ Refreshes all values and plots for a changed peak_id. """

        if self.verbose:
            print("[NOTE] Selected peak: {}".format(self.peak))

        # Update profile plot.
        self.ax_profile.clear()
        draw_profile(self.peak, self.fig_controls, self.ax_profile)

        # Update possible values for parameter 'nperseg'.
        self.s_nperseg.valmax = self.peak.peak_length

        distance = 10
        height = 5
        prominence = 3
        find_peaks_result = scipy.signal.find_peaks(
            self.peak.coverage, distance=distance,
            height=[max(self.peak.coverage) if max(self.peak.coverage) < height
                    else height],
            prominence=prominence)
        self.peak.num_peaks_estimated = max(1, len(find_peaks_result[0]))

        self.s_nperseg.set_val(min(np.ceil(self.peak.peak_length
                                           / self.peak.num_peaks_estimated
                                           ).astype(int),
                                   self.s_nperseg.valmax))

        self.s_nperseg.valinit = self.s_nperseg.val
        self.s_nperseg.vline.set_data([self.s_nperseg.val, self.s_nperseg.val],
                                      [0, 1])
        self.s_nperseg.ax.set_xlim(self.s_nperseg.valmin,
                                   self.s_nperseg.valmax)

    def tb_window_on_submit(self, _val):
        """ Handles the text box event for a changed window type.

        Parameters
        ----------
        _val : str
            The new window type value.
        """
        if self.tb_window.text not in scipy.signal.windows.__all__:
            self.tb_window.set_val('boxcar')
            return

        self.calculate_stft()

    def s_nperseg_changed(self, _val):
        """ Handles the slider event for a changed nperseg value.

        Parameters
        ----------
        _val : int
            The new nperseg value.
        """

        self.resetting_stft_values = True

        # Update possible values for padding.
        self.s_padding.valmax = 2 * self.s_nperseg.val
        self.s_padding.set_val(self.s_nperseg.val)
        self.s_padding.valinit = self.s_padding.val
        self.s_padding.vline.set_data([self.s_padding.val, self.s_padding.val],
                                      [0, 1])
        self.s_padding.ax.set_xlim(self.s_padding.valmin,
                                   self.s_padding.valmax)

        # Update possible values for parameter 'noverlap'.
        self.s_noverlap.valmax = self.s_nperseg.val - 1
        self.s_noverlap.set_val(self.s_nperseg.val // 2)
        self.s_noverlap.valinit = self.s_noverlap.val
        self.s_noverlap.vline.set_data([self.s_noverlap.val,
                                        self.s_noverlap.val],
                                       [0, 1])
        self.s_noverlap.ax.set_xlim(self.s_noverlap.valmin,
                                    self.s_noverlap.valmax)

        self.resetting_stft_values = False
        self.calculate_stft()

    def s_padding_changed(self, _val):
        """ Handles the slider event for a changed padding value.

        Parameters
        ----------
        _val : int
            The new padding value.
        """
        if not self.resetting_stft_values:
            self.calculate_stft()

    def s_noverlap_changed(self, _val):
        """ Handles the slider event for a changed noverlap value.

        Parameters
        ----------
        _val : int
            The new padding value.
        """
        if not self.resetting_stft_values:
            self.calculate_stft()

    def r_detrend_on_clicked(self, _val):
        """ Handles the radio button event for a changed detrend value.

        Parameters
        ----------
        _val : str
            The new detrend value.
        """
        self.calculate_stft()

    def calculate_stft(self):
        """ Calculates the STFT, using the defined parameters. """

        self.clear_plots()

        self.padding = (self.s_padding.val, self.s_padding.val)

        # Sets the parameters for function 'stft'.
        self.stft_args = {}
        self.stft_args['x'] = np.pad(self.peak.coverage, self.padding)
        self.stft_args['fs'] = 1    # Should always be 1, as the sampling
        # frequency is always 1 value per nucleotide.
        self.stft_args['window'] = self.tb_window.text
        self.stft_args['nperseg'] = self.s_nperseg.val
        self.stft_args['noverlap'] = self.s_noverlap.val
        self.stft_args['nfft'] = None
        self.stft_args['detrend'] = self.r_detrend.value_selected
        if self.stft_args['detrend'] == 'False':
            self.stft_args['detrend'] = False
        self.stft_args['return_onesided'] = True    # Should always be True, as
        # input data is always real, therefore two-sided spectrum is symmetric
        # and one-sided result is sufficient.
        self.stft_args['boundary'] = None
        self.stft_args['padded'] = True
        self.stft_args['axis'] = -1

        if self.verbose:
            print("[NOTE] Function 'stft' is executed with the following"
                  " parameters (omitting values for 'x') :\n\t{}".format(
                      {k: v for k, v in self.stft_args.items() if k != 'x'}
                      )
                  )
        self.stft_result = Result()
        self.stft_result.f, self.stft_result.t, self.stft_result.Zxx = \
            deconvolute_peak_with_STFT(self.peak, self.stft_args)
        self.stft_result.stft_args = copy.deepcopy(self.stft_args)

        info_text = ("Info: Shape of Zxx: {}\n"
                     "              # of frequencies: {}\n"
                     "              # of segments: {}\n"
                     "      Estimated # of peaks: {}\n"
                     .format(self.stft_result.Zxx.shape,
                             self.stft_result.Zxx.shape[0],
                             self.stft_result.Zxx.shape[1],
                             self.peak.num_peaks_estimated)
                     )
        self.text_info.set_text(info_text)

        # Update possible values for plotting parameter and plot the result.
        self.update_plot_overview_vals()

    def r_plot_overview_changed(self, _val):
        """ Handles the radio button event for a changed plot overview.

        Parameters
        ----------
        _val : str
            The new plot overview.
        """
        self.update_plot_overview_vals()

    def update_plot_overview_vals(self):
        """ Resets plot parameters after plot values have been changed. """

        self.resetting_plot_values = True

        if self.r_plot_overview.value_selected != 'All':
            if self.r_plot_overview.value_selected == 'Segment':
                self.s_plot_overview_val.valmax = \
                    self.stft_result.Zxx.shape[1] - 1
                self.s_plot_overview_val.label = 'i_segment'
            elif self.r_plot_overview.value_selected == 'Frequency':
                self.s_plot_overview_val.valmax = \
                    self.stft_result.Zxx.shape[0] - 1
                self.s_plot_overview_val.label = 'i_freq'

            current_overview_val = self.s_plot_overview_val.val

            self.s_plot_overview_val.set_val(0)
            self.s_plot_overview_val.valinit = 0
            self.s_plot_overview_val.vline.set_data([0, 0], [0, 1])
            if self.s_plot_overview_val.valmax != 0:
                self.s_plot_overview_val.ax.set_xlim(
                    self.s_plot_overview_val.valmin,
                    self.s_plot_overview_val.valmax)
                if current_overview_val < self.s_plot_overview_val.valmax:
                    self.s_plot_overview_val.set_val(current_overview_val)
                else:
                    self.s_plot_overview_val.set_val(
                        self.s_plot_overview_val.valmax
                        )

        if ((self.r_plot_overview.value_selected == 'All')
                or (self.s_plot_overview_val.valmax == 0)):
            self.s_plot_overview_val.set_active(False)
            self.s_plot_overview_val.ax.set_visible(False)
        else:
            self.s_plot_overview_val.ax.set_visible(True)
            self.s_plot_overview_val.set_active(True)
        self.fig_controls.canvas.draw()

        self.resetting_plot_values = False
        self.plot_stft_result()

    def s_plot_overview_val_changed(self, _val):
        """ Handles the slider event for a changed plot overview value.

        Parameters
        ----------
        _val : int
            The new plot overview value.
        """
        if not self.resetting_plot_values:
            self.plot_stft_result()

    def r_plot_options_changed(self, _val):
        """ Handles the radio button event for a changed plot options.

        Parameters
        ----------
        _val : str
            The plot option that has been changed.
        """
        self.plot_stft_result()

    def c_plot_options_clicked(self, _val):
        """ Handles the check button event for a changed plot options.

        Parameters
        ----------
        _val : str
            The plot option that has been changed.
        """
        self.plot_stft_result()

    def clear_plots(self):
        """ Clears the plots. """
        self.ax_freq.clear()
        if hasattr(self, 'ax_colormesh_abs_Zxx'):
            self.ax_colormesh_abs_Zxx.clear()
        if hasattr(self, 'ax_surface_abs_Zxx'):
            self.ax_surface_abs_Zxx.clear()
        if hasattr(self, 'ax_spectogram'):
            self.ax_spectogram.clear()

    def plot_istft_result(self, i_freq, i_segment):
        """ Plots the inverse STFT for the given frequency and segment.

        Parameters
        ----------
        i_freq : int
            The index of the frequency that should be plotted.
        i_segemnt : int
            The index of the segment that should be plotted.
        """

        z = np.zeros(self.stft_result.Zxx.shape,
                     dtype=self.stft_result.Zxx.dtype)
        z[i_freq, i_segment] = self.stft_result.Zxx[i_freq, i_segment]

        if self.r_plot_options.value_selected.startswith("Include nei"):
            if i_segment > 0:
                z[i_freq, i_segment-1] = \
                    self.stft_result.Zxx[i_freq, i_segment-1]
            if i_segment < self.stft_result.Zxx.shape[1]-1:
                z[i_freq, i_segment+1] = \
                    self.stft_result.Zxx[i_freq, i_segment+1]
        elif self.r_plot_options.value_selected.startswith("Include all"):
            z[i_freq, :] = self.stft_result.Zxx[i_freq, :]
        elif self.r_plot_options.value_selected.startswith("Propagate"):
            z[i_freq, :] = self.stft_result.Zxx[i_freq, i_segment]

        self.istft_args['Zxx'] = z

        t, x = scipy.signal.istft(**self.istft_args)
        if self.r_plot_overview.value_selected == 'Frequency':
            ys = [-1-i_segment]*len(t)
        else:
            ys = [-1-i_freq]*len(t)

        i_seg_start = i_segment * (self.istft_args['nperseg']
                                   - self.istft_args['noverlap'])

        t = t - self.padding[0]

        if self.c_plot_options.get_status()[0]:
            t = t[i_seg_start:i_seg_start + self.istft_args['nperseg']]
            ys = ys[i_seg_start:i_seg_start + self.istft_args['nperseg']]
            x = x[i_seg_start:i_seg_start + self.istft_args['nperseg']]

        self.ax_freq.plot(
            xs=t, ys=ys, zs=x, zdir='z',
            linewidth=1, marker='.', markersize=3.0
            )

    def plot_fft_requencies(self, yvalue):
        """ Plots the frequencies of the FFT if possible.

        Parameters
        ----------
        yvalue : int
            The y position where the results should be plotted.
        """
        if (self.c_plot_options.get_status()[1]
                and (self.r_plot_overview.value_selected == 'Segment')):
            i_segment = self.s_plot_overview_val.val
            i_seg_start = i_segment * (self.istft_args['nperseg']
                                       - self.istft_args['noverlap'])
            i_seg_end = i_seg_start + self.istft_args['nperseg']

            padded_coverage = np.pad(self.peak.coverage, self.padding)

            segment_coverage = padded_coverage[i_seg_start:i_seg_end]
            fhat = scipy.fft.rfft(segment_coverage, len(segment_coverage))
            for i in range(len(fhat)):
                # Note: The results of function stft differ to results of
                # function fft by a different scale factor!

                fback = np.zeros(len(fhat), dtype=fhat.dtype)
                fback[i] = fhat[i]
                fback = scipy.fft.irfft(fback, len(segment_coverage))

                # t = range(i_seg_start, i_seg_end)    # This does not work in
                # all cases. E.g. when the last segment is shorter than the
                # default length, segment_coverage is cropped on the right end,
                # thus len(segment_coverage) is smaller than
                # i_seg_end - i_seg_start
                t = range(i_seg_start - self.padding[0],
                          i_seg_start + len(segment_coverage) - self.padding[0]
                          )
                ys = [-yvalue]*len(t)
                x = fback
                self.ax_freq.plot(
                    xs=t, ys=ys, zs=x, zdir='z',
                    linewidth=1, marker='.', markersize=3.0
                    )

    def plot_stft_result(self):
        """ Plots the STFT results. """

        self.clear_plots()

        if hasattr(self, 'ax_colormesh_abs_Zxx'):
            self.ax_colormesh_abs_Zxx.pcolormesh(
                self.stft_result.t, self.stft_result.f,
                np.abs(self.stft_result.Zxx), shading='auto')
            self.ax_colormesh_abs_Zxx.set_title('STFT Magnitude')
            self.ax_colormesh_abs_Zxx.set_xlabel('Time [sec]')
            self.ax_colormesh_abs_Zxx.set_ylabel('Frequency [Hz]')
        if hasattr(self, 'ax_surface_abs_Zxx'):
            # Plot the surface.
            X, Y = np.meshgrid(self.stft_result.f, self.stft_result.t)
            _surf = self.ax_surface_abs_Zxx.plot_surface(
                X, Y,
                np.abs(self.stft_result.Zxx).T
                )
        if hasattr(self, 'ax_spectogram'):
            spectrogram_args = copy.deepcopy(self.stft_args)
            spectrogram_args.pop('boundary')
            spectrogram_args.pop('padded')
            specto_stft_f, specto_stft_t, specto_Sxx = \
                scipy.signal.spectrogram(**spectrogram_args)
            self.ax_spectogram.pcolormesh(
                specto_stft_t, specto_stft_f, specto_Sxx, shading='gouraud')

        self.ax_freq.set_xlabel('Relative Nucleotide Position')
        self.ax_freq.set_ylabel('Frequency')
        self.ax_freq.set_zlabel('Coverage')
        x_values = np.arange(len(self.peak.coverage))
        z_values = self.peak.coverage
        self.ax_freq.plot(xs=x_values, ys=z_values, zdir='y',
                          label='{}'.format(self.peak), linewidth=1, marker='.'
                          )

        # Sets the parameters for function 'istft'.
        self.istft_args = {}
        self.istft_args['fs'] = self.stft_args['fs']
        self.istft_args['window'] = self.stft_args['window']
        self.istft_args['nperseg'] = self.stft_args['nperseg']
        self.istft_args['noverlap'] = self.stft_args['noverlap']
        self.istft_args['nfft'] = self.stft_args['nfft']
        self.istft_args['input_onesided'] = self.stft_args['return_onesided']
        self.istft_args['boundary'] = self.stft_args['boundary']
        self.istft_args['time_axis'] = -1
        self.istft_args['freq_axis'] = -2

        if self.r_plot_overview.value_selected == 'All':
            for i_freq in range(self.stft_result.Zxx.shape[0]):
                for i_segment in range(self.stft_result.Zxx.shape[1]):
                    self.plot_istft_result(i_freq, i_segment)
        elif self.r_plot_overview.value_selected == 'Segment':
            i_segment = self.s_plot_overview_val.val
            for i_freq in range(self.stft_result.Zxx.shape[0]):
                self.plot_istft_result(i_freq, i_segment)
        elif self.r_plot_overview.value_selected == 'Frequency':
            i_freq = self.s_plot_overview_val.val
            for i_segment in range(self.stft_result.Zxx.shape[1]):
                self.plot_istft_result(i_freq, i_segment)

        yvalue = self.stft_result.Zxx.shape[0] + 1

        yvalue += 1
        self.plot_fft_requencies(yvalue)

        yvalue += 5
        if (self.c_plot_options.get_status()[2]
                and isinstance(self.peak, OriginalPeak)):
            fft_peak = {self.peak.peak_id: copy.deepcopy(self.peak)}
            deconvolute_with_FFT(fft_peak, num_padding=(10, 10),
                                 verbose=self.verbose)
            for l, _c, r in fft_peak[self.peak.peak_id].fft.new_peaks:
                left = l - fft_peak[self.peak.peak_id].chrom_start
                right = r - fft_peak[self.peak.peak_id].chrom_start
                xvalues = range(left, right)
                yvalue += 1
                self.ax_freq.plot(
                    xs=xvalues,
                    ys=[-yvalue]*len(xvalues),
                    zs=[1.0]*len(xvalues),
                    zdir='z',
                    linewidth=1,
                    marker='.',
                    markersize=3.0
                )

        yvalue += 5
        if self.c_plot_options.get_status()[3]:
            for l, _c, r in self.peak.deconv_peaks_rel:
                xvalues = range(l, r)
                yvalue += 1
                self.ax_freq.plot(
                    xs=xvalues,
                    ys=[-yvalue]*len(xvalues),
                    zs=[1.0]*len(xvalues),
                    zdir='z',
                    linewidth=1,
                    marker='.',
                    markersize=3.0
                )

        if (hasattr(self, 'fig_colormesh_abs_Zxx')
                and plt.fignum_exists(self.fig_colormesh_abs_Zxx.number)):
            self.fig_colormesh_abs_Zxx.canvas.draw()
        if (hasattr(self, 'fig_surface_abs_Zxx')
                and plt.fignum_exists(self.fig_surface_abs_Zxx.number)):
            self.fig_surface_abs_Zxx.canvas.draw()
        if (hasattr(self, 'fig_spectogram')
                and plt.fignum_exists(self.fig_spectogram.number)):
            self.fig_spectogram.canvas.draw()
        if plt.fignum_exists(self.fig_freq.number):
            self.fig_freq.canvas.draw()


if __name__ == '__main__':

    from tools.preprocessing import read_coverage_file
    verbose = True

    try:
        input_coverage_file = '../../Data/02_Preprocessing/01__HepG2_against_RBFOX2/RBFOX2__02_02__26_peaks__Rep1__coverage.tsv'    # @IgnorePep8
        peaks = read_coverage_file(input_coverage_file, verbose)
    except FileNotFoundError:
        # input_coverage_file = '../../../Data/02_Preprocessing/01__HepG2_against_RBFOX2/RBFOX2__02_02__26_peaks__Rep1__coverage.tsv'    # @IgnorePep8
        input_coverage_file = '../../../Data/02_Preprocessing/01__HepG2_against_RBFOX2/RBFOX2__02_01__full_7477_peaks__Rep1__coverage.tsv'    # @IgnorePep8
        peaks = read_coverage_file(input_coverage_file, verbose)
    PeakAnalyzer(peaks=peaks, verbose=verbose, create_additional_plots=False)
