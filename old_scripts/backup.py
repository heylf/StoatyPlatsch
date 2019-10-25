# # data of the peak
        # data = data_dict[peak]
        # chr = data[0]
        # start = int(data[1])
        # end = int(data[2])
        # id = data[3]
        # score = data[4]
        # strand = data[5]
        #
        # # without x+1 because genome coordinates starts at zero (end-1, see info bedtools coverage)
        # pre_x = numpy.array([x for x in range(0, len(cov_matrix[peak]))])
        # pre_y = cov_matrix[peak]
        #
        # #y = cov_matrix[9]
        # #y = cov_matrix[15]
        # #y = cov_matrix[21]
        # #y = cov_matrix[18]
        # #y = cov_matrix[47]
        # #y = cov_matrix[50]
        #
        # # Padding with zero makes sure I will not screw up the fitting. Sometimes if a peak is too close to the border
        # # The Gaussian is too big to be fitted and a very borad Guassian will matched to the data.
        # num_padding = 40
        #
        # # without x+1 because genome coordinates starts at zero (end-1, see info bedtools coverage)
        # x = numpy.array([x for x in range(0, len(pre_x) + num_padding)])
        # y = numpy.pad(pre_y, (int(num_padding/2), int(num_padding/2)), 'constant', constant_values=(0, 0))
        # inv_y = y * -1
        #
        # models_dict_array = [{'type': args.peak_model} for i in range(0,args.max_peaks)]
        #
        # spec = {
        #     'x': x,
        #     'y': y,
        #     'model': models_dict_array
        # }
        #
        # peaks_indices_array = [i for i in range(0, args.max_peaks)]
        #
        # # Peak Detection Plot
        # list_of_update_spec_from_peaks = update_spec_from_peaks(args.output_folder, possible_dist, spec,
        #                                                         peaks_indices_array, minimal_height=args.min_height,
        #                                                         distance=args.distance, std=2)
        # peaks_found = list_of_update_spec_from_peaks[0]
        # found_local_minima = list_of_update_spec_from_peaks[1]
        #
        # new_start = start - int(num_padding/2)
        # new_end = end + int(num_padding/2)
        # real_coordinates_list = numpy.array([x for x in range(new_start, new_end)])
        #
        # # Check number of potential local maxima
        # if( len(peaks_found) != 0 ):
        #
        #     # Check for distributions to be deleted
        #     dist_index = 0
        #     while dist_index < len(spec['model']):
        #         if 'params' not in spec['model'][dist_index]:
        #             del spec['model'][dist_index]
        #         else:
        #             dist_index += 1
        #
        #     # Fitting Plot
        #     #bic = 100000
        #
        #     first_model = True
        #     for m in spec['model']:
        #         dist_not_to_ckeck = ""
        #
        #         if ( not first_model ):
        #             dist_not_to_ckeck = args.peak_model
        #
        #         return_dict = dict()
        #
        #         start_time = time.time()
        #         for d in possible_dist:
        #             if ( d != dist_not_to_ckeck ):
        #                 m['type'] = d
        #                 #model, params = generate_model(spec, possible_dist, min_peak_width=args.min_width, max_peak_width=args.max_width)
        #                 #output = model.fit(spec['y'], params, x=spec['x'], nan_policy='propagate')
        #
        #                 fitting(spec, possible_dist, args.min_width, args.max_width, return_dict, d)
        #
        #         m['type'] = min(return_dict, key=return_dict.get)
        #
        #         if ( estimation_count < number_of_peaks_for_estimation ):
        #             estimated_dists.append(m['type'])
        #
        #         end_time = time.time()
        #         print('function took {} s'.format( end_time - start_time) )
        #
        #     print("[NOTE] Fitting Model")
        #
        #     if ( estimation_count == number_of_peaks_for_estimation ):
        #         possible_dist = list(set(estimated_dists))
        #     else:
        #         estimation_count += 1
        #
        #     print(possible_dist)
        #
        #     model, params = generate_model(spec, possible_dist, min_peak_width=args.min_width, max_peak_width=args.max_width)
        #
        #     output = model.fit(spec['y'], params, x=spec['x'], nan_policy='propagate')
        #     #output.plot(data_kws={'markersize': 1})
        #     #plt.savefig('{}/profile_fit.pdf'.format(args.output_folder))
        #
        #     # Get new peaks
        #     peaks_in_profile = get_best_values(spec, output)
        #     num_deconvoluted_peaks = len(peaks_in_profile[0])
        #     components = output.eval_components(x=spec['x'])
        #
        #     peak_start_list = [start] * num_deconvoluted_peaks
        #     peak_end_list = [end] * num_deconvoluted_peaks
        #     peak_center_list = [-1] * num_deconvoluted_peaks
        #
        #     rectangle_start_list = [-1] * num_deconvoluted_peaks
        #     rectangle_end_list = [-1] * num_deconvoluted_peaks
        #
        #     if (num_deconvoluted_peaks > 1):
        #         for i in range(0, num_deconvoluted_peaks):
        #             peak_center = int(peaks_in_profile[0][i])
        #             peak_center_list[i] = real_coordinates_list[peak_center]
        #             peak_sigma = peaks_in_profile[1][i]
        #
        #             # Change Coordinates
        #             left_right_extension = numpy.floor((peak_sigma * args.std))
        #
        #             peak_start_list[i] = real_coordinates_list[peak_center] - left_right_extension
        #             peak_end_list[i] = real_coordinates_list[peak_center] + left_right_extension + 1
        #             # end+1 because genome coordinates starts at zero (end-1, see info bedtools coverage)
        #
        #             rectangle_start_list[i] = peak_center - left_right_extension
        #             rectangle_end_list[i] = peak_center + left_right_extension
        #
        #             if ( peak_start_list[i] < 0 ):
        #                 peak_start_list[i] = 0
        #
        #             if ( peak_end_list[i] > chr_sizes_dict[chr] ):
        #                peak_end_list[i] = chr_sizes_dict[chr]
        #
        #             # Write Output tables
        #             # Check number of potential found peaks
        #             output_table_summits.write("{0}\t{1}\t{1}\t{2}\t{3}\t{4}\n".format(chr, peak_center_list[i], id + "_" + str(i),
        #                                                                                 score, strand))
        #             output_table_new_peaks.write("{0}\t{1}\t{2}\t{3}\t{4}\t{5}\n".format(chr, peak_start_list[i], peak_end_list[i],
        #                                                                                  id + "_" + str(i), score, strand))
        #         output_table_overview.write("{0}\t{1}\t{2}\n".format(id, len(peak_center_list), peak_center_list))
        #     else:
        #         output_table_overview.write("{0}\t{1}\t{2}\n".format(id, "1", start))
        #         summit = real_coordinates_list[numpy.argmax(pre_y)]
        #         output_table_summits.write("{0}\t{1}\t{2}\t{3}\t{4}\t{5}\n".format(chr, summit, summit, id, score, strand))
        #
        #     # Plot Area
        #     if (len(peaks_found) > 1 and plot_counter <= 10):
        #         ax = fig_profile.add_subplot(2, 5, plot_counter)
        #         ax.plot(pre_x, pre_y)
        #         ax.set_xlabel('Relative Nucleotide Position')
        #         ax.set_ylabel('Intensity')
        #         ax.axes.get_xaxis().set_ticks([])
        #
        #         ax2 = fig_extremas.add_subplot(2, 5, plot_counter)
        #         ax2.plot(x, y)
        #         ax2.plot(x, inv_y)
        #         ax2.plot(peaks_found, y[peaks_found], "o")
        #         ax2.plot(found_local_minima, inv_y[found_local_minima], "x")
        #         ax2.set_xlabel('Relative Nucleotide Position')
        #         ax2.set_ylabel('Intensity')
        #
        #         # Deconvolution Plot
        #         # get maximum value of components
        #         max_fitted_y = 0
        #         for i, model in enumerate(spec['model']):
        #             if (max_fitted_y < numpy.max(components[f'm{i}_'])):
        #                 max_fitted_y = numpy.max(components[f'm{i}_'])
        #
        #         max_y_plot = max_fitted_y
        #         if (max_fitted_y < max(y)):
        #             max_y_plot = max(y)
        #
        #         c = color_linear_gradient(start_hex="#FF0000", finish_hex="#0000ff", n=num_deconvoluted_peaks)['hex']
        #         ax3 = fig_deconvolution.add_subplot(2, 5, plot_counter)
        #         # Add rectangles
        #         for i, model in enumerate(spec['model']):
        #             ax3.plot(spec['x'], components[f'm{i}_'], color=c[i])
        #             rect = patches.Rectangle((rectangle_start_list[i], 0),
        #                                      width=rectangle_end_list[i] - rectangle_start_list[i],
        #                                      height=max_y_plot, facecolor=c[i], alpha=0.3)
        #             ax3.add_patch(rect)
        #         ax3.bar(spec['x'], spec['y'], width=1.0, color="black", edgecolor="black")
        #         ax3.set_xlabel('Relative Nucleotide Position')
        #         ax3.set_ylabel('Intensity')
        #         ax3.set_ylim([0, max_y_plot])
        #         ax3.axes.get_xaxis().set_ticks([])
        #
        #         plot_counter += 1
        # else:
        #     output_table_overview.write("{0}\t{1}\t{2}\n".format(id, "1", start))
        #     summit = real_coordinates_list[numpy.argmax(pre_y)]
        #     output_table_summits.write("{0}\t{1}\t{2}\t{3}\t{4}\t{5}\n".format(chr, summit, summit, id, score, strand))