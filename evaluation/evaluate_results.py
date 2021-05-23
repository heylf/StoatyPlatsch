import argparse
import os
import re

from cycler import cycler
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd


class EvaluationResult(object):
    """ Stores the result of the evaluation.

    Attributes
    ----------
    data : pandas.DataFrame
        Stores the result of the evaluation.
    data_re : pandas.DataFrame
        Stores the result of the evaluation with regular expression.
    contexts : dict
        Contains the context information for the motifs. The dictionary
        contains again dictionaries with keys 'all', 'motif', and
        'combined'. The dictionary under 'all' contains the context information
        for the motifs for all sequences. The dictionary under 'motif' contains
        the context information for the motifs only for the sequences that
        actually contain the considered motif. The dictionary under 'combined'
        contains the context information for all motifs defined with
        'motifs_combined_context' combined.
    contexts_re : dict
        Contains the context information for the motifs evaluated with regular
        expressions for all sequences. The dictionary contains again
        dictionaries with keys 'all' and 'motif'. The dictionary under 'all'
        contains the context information for the motifs for all sequences. The
        dictionary under 'motif' contains the context information for the
        motifs only for the sequences that actually contain the considered
        motif.
    """
    pass


class Evaluation(object):
    """ Stores and handles the evaluation of fasta files regarding motifs.

    Attributes
    ----------
    tag : int
        Unique tag for identifying the experiment.
    fasta_file_path : str
        The file path of the fasta file that should be used for the evaluation.
    default_motifs : list
        List of default motifs that is used for some evaluations.
    motifs : list
        List of motifs that should be evaluated.
    motifs_re : list
        List of motifs that should be evaluated using regular expressions.
    motifs_combined_context : list
        List of motifs that should be considered for the combined context
        analysis.
    fasta_data : dict
        Dictionary containing the data of the fasta file.
    result : EvaluationResult
        Stores the result of the evaluation.
    """

    def __init__(self, tag, fasta_file_path, default_motifs, motifs, motifs_re,
                 motifs_combined_context):
        """ Constructor

        Parameters
        ----------
        See attributes description of the class.
        """

        self.tag = tag
        self.fasta_file_path = fasta_file_path

        # Create lists with unique elements.
        self.default_motifs = []
        self.motifs = []
        self.motifs_re = []
        self.motifs_combined_context = []
        for i_self_motifs, i_motifs in [
                (self.default_motifs, default_motifs),
                (self.motifs, motifs),
                (self.motifs_re, motifs_re),
                (self.motifs_combined_context, motifs_combined_context)
                ]:
            for motif in i_motifs:
                if motif not in i_self_motifs:
                    i_self_motifs.append(motif)

        self.fasta_data = None
        self.result = None

    def load_fasta_data(self, verbose=False):
        """ Loads and processes the given fasta file.

        Parameters
        ----------
        verbose : bool (default: False)
            Print information to console when set to True.
        """
        if verbose:
            print("[NOTE] Read the fasta file from '{}'."
                  .format(self.fasta_file_path))

        with open(self.fasta_file_path, "r") as fasta_file:
            key = None
            data = ""
            self.fasta_data = {}
            for line in fasta_file:
                if line.startswith(">"):
                    if key is not None:
                        self.fasta_data[key] = data.upper().replace('T', 'U')
                    key = line[1:].strip()
                    data = ""
                else:
                    data += line.strip()
            if data:
                self.fasta_data[key] = data.upper().replace('T', 'U')

    def evaluate_fasta_data(self):
        """ Evaluates the fasta data. """

        self.result = EvaluationResult()
        er = self.result    # Used as abbreviation.

        counts = {'all': {}, 'once': {}}
        counts_re = {'all': {}, 'once': {}}
        er.contexts = {'all': {}, 'motif': {},
                       'combined': {'all': {}, 'motif': {}}
                       }
        er.contexts_re = {'all': {}, 'motif': {}}

        for i_motifs, i_counts, i_contexts in [
                (self.motifs, counts, er.contexts),
                (self.motifs_re, counts_re, er.contexts_re)
                ]:
            for motif in i_motifs:
                i_counts['all'][motif] = 0
                i_counts['once'][motif] = 0
                i_contexts['all'][motif] = {}
                i_contexts['motif'][motif] = {}

        for _key, data in self.fasta_data.items():

            for i_motifs, i_counts, i_contexts, regex in [
                    (self.motifs, counts, er.contexts, False),
                    (self.motifs_re, counts_re, er.contexts_re, True)
                    ]:
                for motif in i_motifs:
                    if not regex:
                        count = data.count(motif)
                    else:
                        count = len(re.findall(motif, data))
                    i_counts['all'][motif] += count
                    i_counts['once'][motif] += min(count, 1)
                    if not regex:
                        context_data = data.replace(motif, "")
                    else:
                        context_data = re.sub(motif, "", data)
                    context_count = len(context_data)
                    if not i_contexts['all'][motif].get(context_count):
                        i_contexts['all'][motif][context_count] = 1
                    else:
                        i_contexts['all'][motif][context_count] += 1
                    if count:
                        if not i_contexts['motif'][motif].get(context_count):
                            i_contexts['motif'][motif][context_count] = 1
                        else:
                            i_contexts['motif'][motif][context_count] += 1

            context_data, count = \
                re.subn('|'.join(self.motifs_combined_context), '', data)
            context_count = len(context_data)

            if not er.contexts['combined']['all'].get(context_count):
                er.contexts['combined']['all'][context_count] = 1
            else:
                er.contexts['combined']['all'][context_count] += 1
            if count:
                if not er.contexts['combined']['motif'].get(context_count):
                    er.contexts['combined']['motif'][context_count] = 1
                else:
                    er.contexts['combined']['motif'][context_count] += 1

        er.data = pd.DataFrame(index=self.motifs)
        er.data_re = pd.DataFrame(index=self.motifs_re)

        # Calculate the actual result data and store it as DataFrame. Could not
        # be directly stored as DataFrame above, as this was way too slow.
        for i_data, i_motifs, i_counts in [
                (er.data, self.motifs, counts),
                (er.data_re, self.motifs_re, counts_re)
                ]:
            i_data['count'] = 0
            i_data['count_once'] = 0
            i_data['fract'] = 0.0
            i_data['fract_once'] = 0.0

            for motif in i_motifs:
                i_data.loc[motif, 'count'] = i_counts['all'][motif]
                i_data.loc[motif, 'count_once'] = i_counts['once'][motif]

            n = len(self.fasta_data)
            i_data['fract'] = i_data['count'] / n
            i_data['fract_once'] = i_data['count_once'] / n


def init_evaluations(protein='RBFOX2'):
    """ Sets the information for reading the input data.

    Returns
    -------
    evaluations : dict
        A dictionary containing Evaluation objects with the information what
        data should be used. The key is the id of the evaluation object.

    References
    ----------
    .. [1] Begg, B. E., Jens, M., Wang, P. Y., and Burge, C. B. (2019).
       Secondary motifs enable concentration-dependent regulation by rbfox
       family proteins. bioRxiv, page 840272.
    .. [2] Uhl, M., Backofen, R., et al. (2020).
       Improving clip-seq data analysis by incorporating transcript
       information. BMC genomics, 21(1):1–8.
    """

    default_fasta_filename = 'Extract_Genomic_DNA.fasta'
    if protein == 'RBFOX2':
        protein_specific_dir = '01__HepG2_against_RBFOX2'
        folder_appendix = '__full_7477_peaks'
        default_motifs = \
            ['UGCAUG',
             # main motif in paper [1]:
             'GCAUG', 'GCACG',
             # secondary motifs in paper [1]:
             'GCUUG', 'GAAUG', 'GUUUG', 'GUAUG', 'GUGUG', 'GCCUG',
             ]
        motifs_re = []
    else:
        protein_specific_dir = '02__K562_against_PUM2'
        folder_appendix = ''

        # Motif is UGUANAUA, see [2]
        default_motifs = ['UGUAAAUA', 'UGUACAUA', 'UGUAGAUA', 'UGUAUAUA']
        motifs_re = ['UGUA[ACGU]AUA']
    data_root_dir = os.path.join('../../../Data/04_Postprocessing',
                                 protein_specific_dir, '01_GalaxyResults')

    evaluations = {}

    tag = '00__Raw'
    evaluations[tag] = Evaluation(
        tag=tag,
        fasta_file_path=os.path.join(data_root_dir,
                                     '00__Raw{}'.format(folder_appendix),
                                     default_fasta_filename),
        default_motifs=default_motifs,
        motifs=default_motifs,
        motifs_re=motifs_re,
        motifs_combined_context=default_motifs
        )

    tag = '01__FFT'
    evaluations[tag] = Evaluation(
        tag=tag,
        fasta_file_path=os.path.join(data_root_dir,
                                     '01__FFT{}'.format(folder_appendix),
                                     default_fasta_filename),
        default_motifs=default_motifs,
        motifs=default_motifs,
        motifs_re=motifs_re,
        motifs_combined_context=default_motifs
        )

    tag = '02__STFT'
    evaluations[tag] = Evaluation(
        tag=tag,
        fasta_file_path=os.path.join(data_root_dir,
                                     '02__STFT{}'.format(folder_appendix),
                                     default_fasta_filename),
        default_motifs=default_motifs,
        motifs=default_motifs,
        motifs_re=motifs_re,
        motifs_combined_context=default_motifs
        )

    tag = '03__STFT__with_transcript_annotations'
    evaluations[tag] = Evaluation(
        tag=tag,
        fasta_file_path=os.path.join(
            data_root_dir,
            '03__STFT{}__with_transcript_annotations'.format(folder_appendix),
            default_fasta_filename),
        default_motifs=default_motifs,
        motifs=default_motifs,
        motifs_re=motifs_re,
        motifs_combined_context=default_motifs
        )

    return evaluations


def create_motif_context_plot(evaluations, tags, motif, motif_no,
                              context_name, context_type,
                              output_path, output_format='svg'):
    """ Creates and saves the distribution plot for the given motif context.

    Parameters
    ----------
    evaluations : list
        List of the evaluation objects that should be used for plotting.
    tags : dict
        Dictionary with the tag that should be plotted as keys and further
        plotting information.
    motif : str
        The motif that should be plotted.
    motif_no : int
        The motiv number for file naming.
    context_name : str
        The name of the context that should be retrieved from the individual
        evaluation objects.
    context_type : str
        The context type that should be used for the evaluation.
    output_path : str
        The folder path where the plots should be saved. Non existing folders
        will be created.
    output_format : str (default: 'svg')
        The file format which should be used for saving the figures. See
        matplotlib documentation for supported values.
    """

    os.makedirs(output_path, exist_ok=True)

    fig, ax = plt.subplots()
    for tag, evaluation in evaluations.items():
        if tag not in tags:
            continue
        contexts = getattr(evaluation.result, context_name)
        values = contexts[context_type][motif]
        context_length = []
        context_value = []
        for length, value in sorted(values.items()):
            context_length.append(length)
            context_value.append(value)
        ax.plot(context_length, context_value, alpha=alpha, **tags[tag])

    ax.legend()
    ax.set_xlabel('Number of context base pairs')
    ax.set_ylabel('Occurrences')
    fig.tight_layout()

    file_path = os.path.join(
        output_path,
        'all_methods__{}_seq__{:02d}_{}.{}'.format(
            context_type, motif_no, motif, output_format)
        )
    fig.savefig(file_path, format=output_format)
    plt.close(fig)


def create_motif_context_plots(evaluations, output_path, output_format='svg',
                               protein='RBFOX2'):
    """ Creates and saves the distribution plots for the motif contexts.

    Parameters
    ----------
    evaluations : list
        List of the evaluation objects that should be used for plotting.
    output_path : str
        The folder path where the plots should be saved. Non existing folders
        will be created.
    output_format : str (default: 'svg')
        The file format which should be used for saving the figures. See
        matplotlib documentation for supported values.
    protein : str, (default: 'RBFOX2')
        Defines the target of the evaluation.
    """

    os.makedirs(output_path, exist_ok=True)

    plt.rcParams["figure.figsize"] = [12, 8]
    plt.rcParams.update({'font.size': 15})
    plt.rcParams['lines.marker'] = '.'
    plt.rcParams['lines.markersize'] = 4.0
    plt.rcParams['lines.linewidth'] = 1.0
    global alpha
    alpha = 0.8
    plt.rcParams['axes.prop_cycle'] = cycler(color='brcgmyk')

    for tag, evaluation in evaluations.items():
        fig, ax = plt.subplots()

        for context_information in [evaluation.result.contexts['all'],
                                    evaluation.result.contexts_re['all']]:
            for motif, values in context_information.items():
                context_length = []
                context_value = []
                for length, value in sorted(values.items()):
                    context_length.append(length)
                    context_value.append(value)

                ax.plot(context_length, context_value, label=motif,
                        alpha=alpha)

        ax.legend()
        ax.set_xlabel('Number of context base pairs')
        ax.set_ylabel('Occurrences')
        fig.tight_layout()

        file_path = os.path.join(
                output_path,
                '{}__all_seq__all_motifs.{}'
                .format(tag, output_format)
                )
        fig.savefig(file_path, format=output_format)
        plt.close(fig)

    motifs = evaluations['00__Raw'].motifs
    motifs_re = evaluations['00__Raw'].motifs_re
    tags = {'00__Raw': {'label': 'ENCFF871NYM' if protein == 'RBFOX2'
                                 else 'ENCFF880MWQ',
                        'linestyle': '-'},
            '01__FFT': {'label': 'FFT', 'linestyle': '--'},
            '02__STFT':
                {'label': 'STFT', 'linestyle': ':'},
            '03__STFT__with_transcript_annotations':
                {'label': 'STFT transcripts', 'linestyle': ':'}
            }

    for context_type in ['all', 'motif']:
        motif_no = 1
        for context_name, i_motifs in [('contexts', motifs),
                                       ('contexts_re', motifs_re)]:
            for motif in i_motifs:
                create_motif_context_plot(
                    evaluations=evaluations, tags=tags, motif=motif,
                    motif_no=motif_no, context_name=context_name,
                    context_type=context_type, output_path=output_path,
                    output_format=output_format)
                motif_no += 1

    for context_type in ['all', 'motif']:
        fig, ax = plt.subplots()
        for tag, evaluation in evaluations.items():
            if tag not in tags:
                continue
            values = evaluation.result.contexts['combined'][context_type]
            context_length = []
            context_value = []
            for length, value in sorted(values.items()):
                context_length.append(length)
                context_value.append(value)
            ax.plot(context_length, context_value, alpha=alpha, **tags[tag])
        ax.legend()
        ax.set_xlabel('Number of context base pairs')
        ax.set_ylabel('Occurrences')
        fig.tight_layout()
        file_path = os.path.join(
            output_path,
            'motifs_combined__{}_seq.{}'.format(context_type, output_format)
            )
        fig.savefig(file_path, format=output_format)
        plt.close(fig)


def save_results(evaluations, output_path, output_format='svg',
                 protein='RBFOX2'):
    """ Saves the evaluation results.

    Parameters
    ----------
    evaluations : list
        List of the evaluation objects that should be saved
    output_path : str
        The folder path where the results should be saved. Non existing folders
        will be created.
    output_format : str (default: 'svg')
        The file format which should be used for saving the figures. See
        matplotlib documentation for supported values.
    protein : str, (default: 'RBFOX2')
        Defines the target of the evaluation.
    """

    os.makedirs(output_path, exist_ok=True)

    # Save the init configurations.
    filepath = os.path.join(output_path, 'init_settings.txt')
    with open(filepath, 'w') as file:
        for tag, evaluation in evaluations.items():
            evaluation_info = \
                ("tag: {}\n"
                 "fasta_file_path: {}\n"
                 "number of peaks: {}\n"
                 "motifs: {}\n"
                 "motifs_re: {}\n"
                 "motifs_combined_context: {}\n\n"
                 ).format(tag, evaluation.fasta_file_path,
                          len(evaluation.fasta_data),
                          evaluation.motifs, evaluation.motifs_re,
                          evaluation.motifs_combined_context)
            file.write(evaluation_info)

    # Save the calculated results.
    filepath = os.path.join(output_path, 'results.txt')
    with open(filepath, 'w') as file:
        for tag, evaluation in evaluations.items():
            evaluation_info = \
                ("tag: {}\n"
                 "number of peaks: {}\n\n"
                 "data:\n{}\n\n"
                 "data_re:\n{}\n\n"
                 "-------------------------\n\n"
                 ).format(tag, len(evaluation.fasta_data),
                          evaluation.result.data, evaluation.result.data_re)
            file.write(evaluation_info)

    # Create distribution plots.
    create_motif_context_plots(
        evaluations,
        output_path=os.path.join(output_path, 'context_distribution_plots'),
        output_format=output_format, protein=protein
        )


def create_argument_parser():
    """ Creates the argument parser for parsing the program arguments.

    Returns
    -------
    parser : ArgumentParser
        The argument parser for parsing the program arguments.
    """
    parser = argparse.ArgumentParser(
        description=("Creates different evaluation plots."),
        usage="%(prog)s [-h] [options]"
        )
    parser.add_argument(
        "--protein",
        choices=["RBFOX2", "PUM2"],
        default="RBFOX2",
        help="Defines the target of the analysis.",
        type=str
        )
    parser.add_argument(
        "-o", "--output_folder",
        default=os.getcwd(),
        help=("Write results to this path."
              " (default: current working directory)"),
        metavar='path/to/output_folder'
        )

    # Define arguments for configuring the plotting behavior.
    parser.add_argument(
        "--plot_format",
        default="svg",
        help=("The file format which is used for saving the plots. See"
              " matplotlib documentation for supported values.")
        )

    return parser


if __name__ == '__main__':

    parser = create_argument_parser()
    args = parser.parse_args()

    evaluations = init_evaluations(protein=args.protein)

    for tag, evaluation in evaluations.items():
        evaluation.load_fasta_data(verbose=True)
        evaluation.evaluate_fasta_data()

    # Print results
    for tag, evaluation in evaluations.items():
        print("-------------------------")
        print("tag: {} ; number of peaks: {}"
              .format(tag, evaluation.tag, len(evaluation.fasta_data)))
        print(evaluation.result.data)
        print(evaluation.result.data_re)

    mpl.use('Agg')    # To speed up creating the plots.

    save_results(evaluations=evaluations, output_path=args.output_folder,
                 output_format=args.plot_format, protein=args.protein)
