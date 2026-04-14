from abc import ABC, abstractmethod
from contextlib import contextmanager

from dandi.dandiapi import DandiAPIClient
import fsspec
from fsspec.implementations.cached import CachingFileSystem
from pynwb import NWBHDF5IO
import h5py
import socket
import pathlib
import numpy as np
import warnings

hostname = socket.gethostname()
if hostname == 'tycho':
    DATA_BASE_PATH = pathlib.Path("/mnt/data/")
elif 'CL1' in hostname:
    DATA_BASE_PATH = pathlib.Path("/home/labuser/storage/")
else:
    raise ValueError(f'Hostname {socket.gethostname()} not recognized.')



class ArrayWithTime(np.ndarray):
    "The idea is to subclass here, but it seems pretty involved."
    # https://numpy.org/doc/stable/user/basics.subclassing.html#slightly-more-realistic-example-attribute-added-to-existing-array
    # https://stackoverflow.com/a/51955094
    def __new__(cls, input_array, t):
        obj = np.asarray(input_array).view(cls)
        obj.t = t
        return obj

    def __array_finalize__(self, obj):
        if obj is None: return

        if hasattr(obj, 't'):
            self.t = obj.t

    def __reduce__(self):
        return self.__class__, (np.asarray(self), np.asarray(self.t))

    def slice(self, *args, all_axes=False):
        if not all_axes:
            return ArrayWithTime(self[*args], self.t[*args])
        elif all_axes:
            return ArrayWithTime(self[*args], self.t[args[0]])
        else:
            raise ValueError()

    def slice_by_time(self, *args, all_axes=False):
        def convert_from_time_to_indices(x):
            if isinstance(x, slice):
                assert x.step is None
                start, stop = x.start, x.stop
                if start is None:
                    start = self.t.min()
                if stop is None:
                    stop = self.t.max()
                if stop < start:
                    warnings.warn('stop greater than start; remember that time can be negative in slices')
                start = np.searchsorted(self.t, start, side='left')
                stop = np.searchsorted(self.t, stop, side='right')
                return slice(start, stop)
            elif x is ...:
                return x
            else:
                return self.time_to_sample(x)

        if len(args):
            if all_axes:
                args = (convert_from_time_to_indices(args[0]),) + args[1:]
            else:
                args = tuple(convert_from_time_to_indices(x) for x in args)

        return self.slice(*args, all_axes=all_axes)

    def as_array(self):
        return np.array(self)

    def time_to_sample(self, time):
        return np.searchsorted(self.t, time)

    @staticmethod
    def align_indices(a, b, complement=False):
        assert len(a.t) > 0 and len(b.t) > 0, 'neither of the arrays should be empty'
        # there's a faster way to do this with np.searchsorted
        a_t = np.array(a.t)
        b_t = np.array(b.t)
        a: ArrayWithTime
        assert (a_t[1:] - a_t[:-1] > 0).all()
        assert (b_t[1:] - b_t[:-1] > 0).all()
        idx_a = 0
        idx_b = 0
        a_indices = []
        b_indices = []

        while idx_a < len(a) and idx_b < len(b):
            d = a_t[idx_a] - b_t[idx_b]
            if np.isclose(0,d):
                a_indices.append(idx_a)
                b_indices.append(idx_b)
                idx_b += 1
                idx_a += 1
            elif d > 0:
                idx_b += 1
            else:
                idx_a += 1
        a_indices = np.array(a_indices)
        b_indices = np.array(b_indices)
        if complement:
            a_indices = np.setdiff1d(np.arange(len(a)), a_indices)
            b_indices = np.setdiff1d(np.arange(len(b)), b_indices)
        return ArrayWithTime(a[a_indices], a_t[a_indices]), ArrayWithTime(b[b_indices], b_t[b_indices])

    @staticmethod
    def subtract_aligned_indices(a, b):
        a, b = ArrayWithTime.align_indices(a, b)
        return ArrayWithTime(a - b, a.t)

    @property
    def dt(self):
        dts = np.diff(self.t)
        dt = np.median(dts)
        assert np.ptp(dts)/dt < 0.05
        return dt

    @staticmethod
    def from_list(input_list, squeeze_type='none', drop_early_nans=False, reshape_mid_nans=True):
        if len(input_list) and not hasattr(input_list[-1], 't'):
            warnings.warn("guessing t for input list")
            input_list = [ArrayWithTime(x, i) for i, x in enumerate(input_list)]

        if drop_early_nans:
            i = 0
            while i < len(input_list) and not np.isfinite(input_list[i]).all():
                i += 1
            input_list = input_list[i:]

        if reshape_mid_nans:
            for i in range(len(input_list)):
                hit = False
                if not np.isfinite(input_list[i]).any() and len(np.array(input_list[i]).shape) and np.array(input_list[i]).shape[-1] != np.array(input_list[0]).shape[-1]:
                    hit = True
                    input_list[i] = input_list[i][..., :np.shape(input_list[0])[-1]]
                    assert input_list[i].shape == np.array(input_list[0]).shape
                if hit:
                    warnings.warn('truncated an all-nan in the middle of a run')

        t = np.array([x.t for x in input_list])
        if squeeze_type == 'none' or squeeze_type is None:
            input_array = np.array(input_list)
        elif squeeze_type == 'to_2d':
            input_array = np.squeeze(input_list)
            if len(input_array.shape) == 1:
                input_array = input_array[:, None]
            elif len(input_array.shape) == 3:
                # warnings.warn("squeezing 3d array to 2d, this is unusual")
                input_array = input_array.reshape([-1, input_array.shape[-1]])
            assert len(input_array.shape) == 2
        elif squeeze_type == 'squeeze':
            input_array = np.squeeze(input_list)
        else:
            raise ValueError()

        return ArrayWithTime(input_array=input_array, t=t)

    @staticmethod
    def from_nwb_timeseries(timeseries):
        return ArrayWithTime(timeseries.data[:], timeseries.timestamps[:])

    @staticmethod
    def from_notime(a):
        return ArrayWithTime(a, np.arange(len(a)))


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
    doi = "https://doi.org/10.48324/dandi.000776/0.241009.1509"
    dandiset_id = "000776"
    version_id = "0.241009.1509"
    dataset_base_path = DATA_BASE_PATH / 'atanas24'

    sub_datasets = [
        "sub-2022-06-14-01-SWF702/sub-2022-06-14-01-SWF702_ses-20220614_behavior+image+ophys.nwb"
    ]

    def __init__(self, sub_dataset_identifier=sub_datasets[0]):
        if isinstance(sub_dataset_identifier, int):
            sub_dataset_identifier = self.sub_datasets[sub_dataset_identifier]

        self.velocity, self.angular_velocity, self.position, self.heading = self.construct(sub_dataset_identifier)


    def construct(self, sub_dataset_identifier):
        with self.acquire(sub_dataset_identifier) as fhan:
            file = fhan.read()
            velocity = ArrayWithTime.from_nwb_timeseries(file.processing['Behavior']['velocity'].time_series['velocity'])
            angular_velocity = ArrayWithTime.from_nwb_timeseries(file.processing['Behavior']['angular_velocity'].time_series['angular_velocity'])

        position, heading = self.position_from_velocity(velocity, angular_velocity)

        return velocity, angular_velocity, position, heading


    @staticmethod
    def position_from_velocity(velocity, angular_velocity):
        x = [np.array([0,0])]
        angle = [0]
        assert np.array_equal(velocity.t,  angular_velocity.t)
        for v, a, dt in zip(velocity, angular_velocity, np.diff(velocity.t)):
            dx = np.array([np.cos(angle[-1]), np.sin(angle[-1])]) * v * dt
            x.append(x[-1] + dx)
            angle.append(angle[-1] + a * dt)
        return np.array(x), np.array(angle)

