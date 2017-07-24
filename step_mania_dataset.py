from torch.utils.data import Dataset
import os
import wget
import re
import zipfile
import urllib
import random

from step_mania_chart import StepManiaChart

default_packs = {
    "Tsunamix III": "https://fra.xtil.net/simfiles/data/tsunamix/III/Tsunamix III [SM5].zip",
    "Fraxtil's Arrow Arrangements": "https://fra.xtil.net/simfiles/data/arrowarrangements/Fraxtil's Arrow Arrangements [SM5].zip",
    "Fraxtil's Beast Beats": "https://fra.xtil.net/simfiles/data/beastbeats/Fraxtil's Beast Beats [SM5].zip",
    "In The Groove": "http://stepmaniaonline.net/downloads/packs/In The Groove 1.zip",
    "In The Groove 2": "http://stepmaniaonline.net/downloads/packs/In The Groove 2.zip"
}

default_root_pack_dir = os.path.expanduser("~/.stepmania-5.0/Songs")


class StepManiaDataset(Dataset):
    """
    Dataset of stepmania charts and songs.

    Args:
        root_pack_dir : str ("~/.stepmania-5.0/Songs")
            Path to the root directory of stepmania packs. If doesn't
            exist will create it.

        download_default : bool (True)
            Whether or not to download the default dataset packs if
            they are not found in the root_pack_dir
            (Warning, large download)

        download_packs : dict<str, optional<str>> ({})
            Extra packs to index, and download if the url is provided.
            In the form of a dict mapping [pack (folder) name] => optional URL

            If no directory matching the given pack is found in the root_pack_dir,
            and a URL is provided it will attempt to download and extract the pack
            at that url to a directory renamed to the given pack (folder) name, within
            the provided root_pack_dir folder.
    """

    def __init__(self, root_pack_dir=default_root_pack_dir, download_default=True, download_packs={}):
        super(StepManiaDataset, self).__init__()
        if not os.path.isdir(root_pack_dir):
            os.mkdir(root_pack_dir)
        self.root_pack_dir = root_pack_dir
        self.included_packs = []

        if download_default:
            self.download_and_extract_packs(default_packs)
            self.included_packs += [k for k in default_packs]

        self.download_and_extract_packs(download_packs)
        self.included_packs += [k for k in download_packs]

        self.index_pack_dir()
        self.shuffle_index()

    def download_and_extract_packs(self, pack_urls):
        """
        Downloads the zip files located at the provided pack_urls if
        a pack of the same name has not already been extracted to the folder

        Args:
            pack_urls : dict<str, optional<str>>
                Extra packs to index, and download if the url is provided.
                In the form of a dict mapping [pack (folder) name] => optional URL

                If no directory matching the given pack is found in the root_pack_dir,
                and a URL is provided it will attempt to download and extract the pack
                at that url to a directory renamed to the given pack (folder) name, within
                the provided root_pack_dir folder.
        """
        for pack_name, pack_url in pack_urls.items():
            if os.path.isdir(os.path.join(self.root_pack_dir, pack_name)):
                print("pack_name ", pack_name, " from pack_url ",
                      pack_url, " already exists in pack_dir ",
                      self.root_pack_dir, ". Skipping download...")
            else:
                if pack_url is not None:
                    pack_url = urllib.parse.unquote(pack_url)
                    print("Downloading pack ", pack_name,
                          " from url ", pack_url)
                    filename = wget.download(pack_url)
                    with zipfile.ZipFile(filename, "r") as zip_ref:
                        zip_ref.extractall(self.root_pack_dir)
                    os.remove(filename)
                    print("Pack ", pack_name, "successfully downloaded!")
                else:
                    raise Exception("pack of name", pack_name, "was not found in root pack directory",
                                    self.root_pack_dir, "and no URL to download it was provided")

    def index_pack_dir(self):
        """
        Recursively finds and re-indexes all valid step-mania charts and songs
        (to this generator) in self.pack_dir
        """
        self.chart_list = []
        for included_pack in self.included_packs:
            pack_dir = os.path.join(self.root_pack_dir, included_pack)
            for root, dirnames, filenames in os.walk(pack_dir):
                for filename in filenames:
                    if filename.endswith('.sm'):
                        try:
                            StepManiaChart(root)
                            self.chart_list.append(root)
                        except ValueError as e:
                            print("Unable to parse stepmania file", root, "due to",
                                  str(e), "... Chart not included in dataset")
        self.index = [i for i, _ in enumerate(self.chart_list)]

    def __len__(self):
        return len(self.index)

    def __getitem__(self, i):
        return StepManiaChart(self.chart_list[self.index[i]]).to_ddc_io()

    def shuffle_index(self):
        random.shuffle(self.index)
