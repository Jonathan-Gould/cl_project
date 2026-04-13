from abc import ABC, abstractmethod
from contextlib import contextmanager

from dandi.dandiapi import DandiAPIClient
import fsspec
from fsspec.implementations.cached import CachingFileSystem
from pynwb import NWBHDF5IO
import h5py
import socket
import pathlib


match socket.gethostname().lower():
    case 'tycho':
        DATA_BASE_PATH = pathlib.Path("/mnt/data/")
    case 'CL1-2544-135':
        DATA_BASE_PATH = pathlib.Path("/data/tmp")
    case _:
        ValueError()


class DandiDataset(ABC):
    @property
    @abstractmethod
    def dandiset_id(self):
        pass

    @property
    @abstractmethod
    def version_id(self):
        pass

    @contextmanager
    def acquire(self, asset_path):
        # https://pynwb.readthedocs.io/en/latest/tutorials/advanced_io/streaming.html
        with DandiAPIClient() as client:
            asset = client.get_dandiset(self.dandiset_id, version_id=self.version_id).get_asset_by_path(asset_path)
            s3_url = asset.get_content_url(follow_redirects=1, strip_query=True)

        fs = fsspec.filesystem("http")
        fs = CachingFileSystem(
            fs=fs,
            cache_storage=[DATA_BASE_PATH / "nwb_cache"],
        )

        with fs.open(s3_url, "rb") as f:
            with h5py.File(f) as file:
                fhan = NWBHDF5IO(file=file, load_namespaces=True)
                yield fhan

class Atanas24Dataset(DandiDataset):
    dandiset_id = "000776"
    version_id = "0.241009.1509"
    dataset_base_path = DATA_BASE_PATH / 'atanas24'

    sub_datasets = [
        "sub-2022-06-14-01-SWF702/sub-2022-06-14-01-SWF702_ses-20220614_behavior+image+ophys.nwb"
    ]

    def __init__(self, sub_dataset_identifier=sub_datasets[0]):
        if isinstance(sub_dataset_identifier, int):
            sub_dataset_identifier = self.sub_datasets[sub_dataset_identifier]

        self.v, self.av = self.construct(sub_dataset_identifier)


    def construct(self, sub_dataset_identifier):
        with self.acquire(sub_dataset_identifier) as fhan:
            file = fhan.read()
            velocity = file.processing['Behavior']['velocity'].time_series['velocity'].data[:]
            angular_velocity = file.processing['Behavior']['angular_velocity'].time_series['angular_velocity'].data[:]

        return velocity, angular_velocity
