# author: Justus Schock (justus.schock@rwth-aachen.de)

import yaml


class Config(object):
    """
    Implements parser for configuration files

    """

    def __init__(self, verbose=False):
        """

        Parameters
        ----------
        verbose : bool
            verbosity

        """
        self.verbose = verbose

    def __call__(self, config_file, config_group=None):
        """
        Actual parsing

        Parameters
        ----------
        config_file : string
            path to YAML file with configuration

        config_group : string or None
            group key to return
            if None: return dict of all keys
            if string: return only values of specified group

        Returns
        -------
        dict
            configuration dict
        """
        state_dict = {}

        # open config file
        with open(config_file, 'r') as file:
            docs = yaml.load_all(file)

            # iterate over document
            for doc in docs:

                # iterate over groups
                for group, group_dict in doc.items():
                    for key, vals in group_dict.items():

                        # set attributes with value 'None' to None
                        if vals == 'None':
                            group_dict[key] = None

                    state_dict[group] = group_dict

                    if self.verbose:
                        print("LOADED_CONFIG: \n%s\n%s:\n%s\n%s" % (
                            "=" * 20, str(group), "-" * 20, str(group_dict)))

        if config_group is not None:
            return state_dict[config_group]
        else:
            return state_dict
