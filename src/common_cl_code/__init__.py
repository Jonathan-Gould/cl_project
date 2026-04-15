from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version(__package__)
except PackageNotFoundError:
    __version__ = "unknown"



import socket
import pathlib

hostname = socket.gethostname()
if hostname == 'tycho':
    datasets_base_path = pathlib.Path("/mnt/data/")
    recording_base_path = datasets_base_path / "cl-sdk_output"
elif 'CL1' in hostname:
    datasets_base_path = pathlib.Path("/home/labuser/storage/")
    recording_base_path = pathlib.Path("/home/labuser/recordings/")
else:
    raise ValueError(f'Hostname {socket.gethostname()} not recognized.')


baseline_fr_file = datasets_base_path / 'baseline_fr.npz'