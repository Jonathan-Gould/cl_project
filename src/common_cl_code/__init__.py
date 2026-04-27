from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version(__package__)
except PackageNotFoundError:
    __version__ = "unknown"



import socket
import pathlib
import cl

running_on_real_device = 'sdk' not in cl.get_system_attributes()['system_id']

hostname = socket.gethostname()
if hostname == 'tycho':
    assert not running_on_real_device
    datasets_base_path = pathlib.Path("/mnt/data/")
    recording_base_path = datasets_base_path / "cl-sdk_output"
elif 'CL1' in hostname:
    datasets_base_path = pathlib.Path("/home/labuser/storage/")
    recording_base_path = pathlib.Path("/home/labuser/recordings/")
    assert running_on_real_device
else:
    raise ValueError(f'Hostname {socket.gethostname()} not recognized.')


frames_per_second = 25_000

grid_size = (8,8)
no_stim_channels = {0, 56, 63, 7}
