import kaggle
import os
import zipfile
from shapedata.io import pts_exporter
import shutil
import pandas as pd
from multiprocessing import Pool
from functools import partial
from sklearn.model_selection import train_test_split
from shapedata import SingleShapeDataProcessing
import numpy as np


def _make_pca(data_dir, out_file, normalize_rot=False, rotation_idxs=()):
    """
    Creates a PCA from data in a given directory
    
    Parameters
    ----------
    data_dir : str
        directory containing the image and landmark files
    out_file : str
        file the pca will be saved to
    normalize_rot : bool, optional
        whether or not to normalize the data's rotation
    rotation_idxs : tuple, optional
        indices for rotation normalization, msut be specified if 
        ``normalize_rot=True``
    
    """

    data_dir = os.path.abspath(data_dir)
    out_file = os.path.abspath(out_file)

    data = SingleShapeDataProcessing.from_dir(data_dir)
    if normalize_rot:
        for idx in range(len(data)):
            data[idx] = data[idx].normalize_rotation(rotation_idxs[0],
                                                     rotation_idxs[1])

    pca = data.lmk_pca(True, True)

    if out_file.endswith(".npz"):
        np.savez(out_file, shapes=pca)
    elif out_file.endswith(".npy"):
        np.save(out_file, pca)
    elif out_file.endswith(".txt"):
        np.savetxt(out_file, pca)
    else:
        np.savez(out_file + ".npz", shapes=pca)


def _process_single_cat_file(file, target_dir):
    """
    Processes a single file of the cat dataset
    
    Parameters
    ----------
    file : str
        the file to process
    target_dir : str
        the target directory
    
    """

    file = os.path.abspath(file)
    target_dir = os.path.abspath(target_dir)

    pd_frame = pd.read_csv(str(file) + ".cat", sep=' ', header=None)
    landmarks = (pd_frame.as_matrix()[0][1:-1]).reshape((-1, 2))
    # switch xy
    landmarks[:, [0, 1]] = landmarks[:, [1, 0]]

    target_file = os.path.join(target_dir, os.path.split(
        os.path.split(file)[0])[-1] + "_" + os.path.split(file)[-1])

    # export landmarks
    pts_exporter(landmarks, str(target_file.rsplit(".", 1)[0]) + ".pts")

    # move image file
    shutil.move(file, target_file)
    os.remove(file + ".cat")


def _prepare_cats(out_dir, remove_zip=False, normalize_pca_rot=False,
                  **split_options):
    """
    Prepares the cat dataset (with multiprocessing)
    
    Parameters
    ----------
    out_dir : str
        the output directory
    remove_zip : bool, optional
        whether or not to remove the ZIP file after finishing the preparation
    normalize_pca_rot : bool, optional
        whether or not to normalize the data's rotation during PCA

    See Also
    --------
    `Cat Dataset <https://www.kaggle.com/crawford/cat-dataset>`_
    
    """


    out_dir = os.path.abspath(out_dir)

    data_path = os.path.join(out_dir, "Cats")
    os.makedirs(data_path, exist_ok=True)

    if not os.path.isfile(os.path.join(data_path, "cats.zip")):
        print("\tDownloading Data")
        kaggle.api.dataset_download_cli("crawford/cat-dataset",
                                        path=data_path, unzip=True)

    if not (os.path.isdir(os.path.join(data_path, "train")) and
            os.path.isdir(os.path.join(data_path, "test"))):

        if not os.path.isdir(os.path.join(data_path, "tmp_data")):
            print("\tExtracting Data")
            with zipfile.ZipFile(os.path.join(data_path, "cats.zip")) as zip_ref:
                zip_ref.extractall(os.path.join(data_path, "tmp_data"))

        # get all jpeg files
        sub_dirs = [os.path.join(data_path, "tmp_data", x)
                    for x in os.listdir(os.path.join(data_path, "tmp_data"))
                    if os.path.isdir(os.path.join(data_path, "tmp_data", x))]

        img_files = []
        for _dir in sub_dirs:
            img_files += [os.path.join(_dir, x) for x in os.listdir(_dir)
                          if x.endswith(".jpg")]

        train_files, test_files = train_test_split(img_files, **split_options)

        if not (os.path.isdir(os.path.join(data_path, "train")) and
                os.path.isdir(os.path.join(data_path, "test"))):

            print("Preprocessing Data")

            os.makedirs(os.path.join(data_path, "train"), exist_ok=True)
            with Pool() as p:
                p.map(partial(_process_single_cat_file,
                              target_dir=os.path.join(data_path, "train")),
                      train_files)

            os.makedirs(os.path.join(data_path, "test"), exist_ok=True)
            with Pool() as p:
                p.map(partial(_process_single_cat_file,
                              target_dir=os.path.join(data_path, "test")),
                      test_files)

        shutil.rmtree(os.path.join(data_path, "tmp_data"))

    print("Make PCA")
    _make_pca(os.path.join(data_path, "train"),
              os.path.join(data_path, "train_pca.npz"),
              normalize_rot=normalize_pca_rot, rotation_idxs=(0, 1))

    if remove_zip:
        os.remove(os.path.join(data_path, "cats.zip"))


def _prepare_ibug_dset(zip_file, dset_name, out_dir, remove_zip=False,
                       normalize_pca_rot=True):
    """
    Prepares an ibug dataset (from a given zipfile)
    
    Parameters
    ----------
    zip_file : str
        the zip archive containing the data
    dset_name : str
        the dataset's name
    out_dir : str
        the output directory
    remove_zip : bool, optional
        whether or not to remove the ZIP file after finishing the preparation
    normalize_pca_rot : bool, optional
        whether or not to normalize the data's rotation during PCA

    See Also
    --------
    `iBug Datasets <https://ibug.doc.ic.ac.uk/resources/facial-point-annotations/>`_
    
    """

    zip_file = os.path.abspath(zip_file)
    out_dir = os.path.abspath(out_dir)

    data_path = os.path.join(out_dir, dset_name)
    os.makedirs(data_path, exist_ok=True)

    print("\tExtracting Data")
    with zipfile.ZipFile(zip_file) as zip_ref:
        zip_ref.extractall(data_path)

    print("\tPreprocessing Data")
    _make_pca(os.path.join(data_path, "trainset"),
              os.path.join(data_path, "train_pca.npz"),
              normalize_rot=normalize_pca_rot, rotation_idxs=(37, 46))

    if remove_zip:
        os.remove(zip_file)


def prepare_lfpw_dset():
    """
    Prepares the LFPW Dataset from commandline arguments

    See Also
    --------
    :meth:`_prepare_ibug_dset`
    `iBug Datasets <https://ibug.doc.ic.ac.uk/resources/facial-point-annotations/>`_
    `LFPW Dataset <https://neerajkumar.org/databases/lfpw/>`_
    
    """

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--zip_file", type=str,
                        help="Zipfile containing the lfpw database")

    parser.add_argument("-d", "--ddir", type=str,
                        help="Target data directory")

    parser.add_argument("--normalize_pca_rot", action="store_true",
                        help="Whether or not to normalize the pca's rotation")

    parser.add_argument("--remove_zip", action="store_true",
                        help="Zipfiles will be removed after processing data",
                        default=False)

    args = parser.parse_args()

    _prepare_ibug_dset(args.zip_file, "lfpw", args.ddir, args.remove_zip,
                       args.normalize_pca_rot)


def prepare_helen_dset():
    """
    Prepares the HELEN Dataset from commandline arguments

    See Also
    --------
    :meth:`_prepare_ibug_dset`
    `iBug Datasets <https://ibug.doc.ic.ac.uk/resources/facial-point-annotations/>`_
    `HELEN Dataset <http://www.ifp.illinois.edu/~vuongle2/helen/>`_

    """
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--zip_file", type=str,
                        help="Zipfile containing the helen database")

    parser.add_argument("-d", "--ddir", type=str,
                        help="Target data directory")

    parser.add_argument("--normalize_pca_rot", action="store_true",
                        help="Whether or not to normalize the pca's rotation")

    parser.add_argument("--remove_zip", action="store_true",
                        help="Zipfiles will be removed after processing data",
                        default=False)

    args = parser.parse_args()

    _prepare_ibug_dset(args.zip_file, "helen", args.ddir, args.remove_zip,
                       args.normalize_pca_rot)


def prepare_cat_dset():
    """
    Prepares the Cat Dataset from commandline arguments
    
    See Also
    --------
    :meth:`_prepare_cats`
    `Cat Dataset <https://www.kaggle.com/crawford/cat-dataset>`_

    """
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--ddir", type=str,
                        help="Target data directory")
    parser.add_argument("--normalize_pca_rot", action="store_true",
                        help="Whether or not to normalize the pca's rotation")
    parser.add_argument("--test_size", type=float, default=0.25,
                        help="Testsize for \
                            sklearn.model_selection.train_test_split")
    parser.add_argument("--train_size", type=float, default=None,
                        help="Testsize for \
                            sklearn.model_selection.train_test_split")
    parser.add_argument("--no_shuffle", action="store_true",
                        help="If specified, data will not be shuffled during \
                            train_test_split ")
    parser.add_argument("--random_state", type=int, default=None,
                        help="random state for \
                            sklearn.model_selection.train_test_split ")
    parser.add_argument("--remove_zip", action="store_true",
                        help="Zipfiles will be removed after processing data",
                        default=False)
    args = parser.parse_args()

    split_options = {
        "test_size": args.test_size,
        "train_size": args.train_size,
        "shuffle": False if args.no_shuffle else True,
        "random_state": args.random_state
    }
    _prepare_cats(args.ddir, args.remove_zip, args.normalize_pca_rot,
                  **split_options)


def prepare_all_data():
    """
    Prepares all Datasets from commandline arguments

    See Also
    --------
    :meth:`_prepare_ibug_dset`
    :meth:`_prepare_cats`
    
    """
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--lfpw", action="store_true",
                        help="If Flag is set, the lfpw database will be \
                            preprocessed; Must specify '--lzip' argument ",
                        default=False)
    parser.add_argument("--helen", action="store_true",
                        help="If Flag is set, the helen database will be \
                            preprocessed; Must specify '--hzip' argument ",
                        default=False)
    parser.add_argument("--cats", action="store_true",
                        help="If Flag is set, the cat database will be \
                        downloaded and preprocessed ",
                        default=False)
    parser.add_argument("--lzip", type=str, default=None,
                        help="Zipfile containing the lfpw database")
    parser.add_argument("--hzip", type=str, default=None,
                        help="Zipfile containing the helen database")

    parser.add_argument("-d", "--ddir", type=str,
                        help="Target data directory")
    parser.add_argument("--test_size", type=float, default=0.25,
                        help="Testsize for \
                        sklearn.model_selection.train_test_split")
    parser.add_argument("--train_size", type=float, default=None,
                        help="Testsize for \
                        sklearn.model_selection.train_test_split")
    parser.add_argument("--no_shuffle", action="store_true",
                        help="If specified, data will not be shuffled during \
                        train_test_split ")
    parser.add_argument("--random_state", type=int, default=None,
                        help="random state for \
                        sklearn.model_selection.train_test_split ")
    parser.add_argument("--remove_zip", action="store_true",
                        help="Zipfiles will be removed after processing data",
                        default=False)

    parser.add_argument("--normalize_pca_rot", action="store_true",
                        help="Whether or not to normalize the pca's rotation")

    args = parser.parse_args()

    data_dir = args.ddir

    split_options = {
        "test_size": args.test_size,
        "train_size": args.train_size,
        "shuffle": False if args.no_shuffle else True,
        "random_state": args.random_state
    }

    if args.remove_zip:
        remove_zip = True
    else:
        args.remove_zip = False

    if args.cats:
        print("Prepare Cats Dataset")
        _prepare_cats(data_dir, remove_zip=remove_zip,
                      normalize_pca_rot=args.normalize_pca_rot,
                      **split_options)

    if args.lfpw and args.lzip is not None:
        print("Prepare LFPW Dataset")
        _prepare_ibug_dset(args.lzip, "lfpw", data_dir, remove_zip=remove_zip,
                           normalize_pca_rot=args.normalize_pca_rot,
                           )

    if args.helen and args.hzip is not None:
        print("Prepare HELEN Dataset")
        _prepare_ibug_dset(args.hzip, "helen", data_dir, remove_zip=remove_zip,
                           normalize_pca_rot=args.normalize_pca_rot,
                           )

    print("Preprocessed all dataset!")


if __name__ == '__main__':
    prepare_all_data()
