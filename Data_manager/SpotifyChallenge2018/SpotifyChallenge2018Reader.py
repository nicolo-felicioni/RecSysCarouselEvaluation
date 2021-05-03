#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 14/09/17

@author: Maurizio Ferrari Dacrema
"""


import numpy as np
import pandas as pd
import zipfile, os, shutil
from Data_manager.DatasetMapperManager import DatasetMapperManager
from Data_manager.DataReader import DataReader


class SpotifyChallenge2018Reader(DataReader):

    DATASET_URL = "https://www.aicrowd.com/challenges/spotify-million-playlist-dataset-challenge"
    DATASET_SUBFOLDER = "SpotifyChallenge2018/"

    DATASET_SPECIFIC_MAPPER = []
    AVAILABLE_ICM = []
    AVAILABLE_URM = ["URM_all", "URM_position"]


    def _get_dataset_name_root(self):
        return self.DATASET_SUBFOLDER



    def _load_from_original_file(self):
        # Load data from original

        self._print("Loading original data")

        compressed_file_folder = self.DATASET_OFFLINE_ROOT_FOLDER + self.DATASET_SUBFOLDER
        decompressed_file_folder = self.DATASET_SPLIT_ROOT_FOLDER + self.DATASET_SUBFOLDER


        try:

            dataFile = zipfile.ZipFile(compressed_file_folder + "dataset_challenge.zip")
            URM_path = dataFile.extract("interactions.csv", path=decompressed_file_folder + "decompressed/")

        except (FileNotFoundError, zipfile.BadZipFile):

            self._print("Unable to find data zip file.")
            self._print("Automatic download not available, please ensure the compressed data file is in folder {}.".format(compressed_file_folder))
            self._print("Data can be downloaded here: {}".format(self.DATASET_URL))

            # If directory does not exist, create
            if not os.path.exists(compressed_file_folder):
                os.makedirs(compressed_file_folder)

            raise FileNotFoundError("Automatic download not available.")


        self._print("Loading Interactions")
        URM_position_dataframe = pd.read_csv(filepath_or_buffer=URM_path, sep="\t", header=0,
                                          usecols=['pid','tid','pos'],
                                          dtype={'pid':str,'tid':str,'pos':int})

        # "Data" here represents the position in the playlist
        URM_position_dataframe.columns = ["UserID", "ItemID", "Data"]

        self._print("Removing duplicated interactions")
        URM_position_dataframe = URM_position_dataframe.groupby(['UserID', 'ItemID'], as_index=False )['Data'].max()

        URM_all_dataframe = URM_position_dataframe.copy()
        URM_all_dataframe["Data"] = 1

        # Start the information on the "Position" from 1 to avoid it gets accidentally removed in the sparse format
        URM_position_dataframe["Data"] += 1

        dataset_manager = DatasetMapperManager()
        dataset_manager.add_URM(URM_all_dataframe, "URM_all")
        dataset_manager.add_URM(URM_position_dataframe, "URM_position")

        loaded_dataset = dataset_manager.generate_Dataset(dataset_name=self._get_dataset_name(),
                                                          is_implicit=self.IS_IMPLICIT)


        self._print("Cleaning Temporary Files")

        shutil.rmtree(decompressed_file_folder + "decompressed", ignore_errors=True)

        self._print("Loading Complete")

        return loaded_dataset


