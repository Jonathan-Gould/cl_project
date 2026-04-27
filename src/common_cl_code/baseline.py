from . import datasets_base_path, frames_per_second, recording_base_path
import cl
import tables
import numpy as np
from common_cl_code import running_on_real_device


baseline_fr_file = datasets_base_path / 'baseline_fr.npz'


def record_new_baseline(recording_length_minutes = 30):
    with cl.open() as neurons:
        recording = neurons.record(
            stop_after_seconds=60 * recording_length_minutes,
            include_spikes=True,
            include_stims=True,
            include_raw_samples=False,
            include_data_streams=False,
            file_location=str(recording_base_path.resolve())
        )

        if running_on_real_device:
            recording.wait_until_stopped()
        else:
            for tick in neurons.loop(ticks_per_second=10, stop_after_seconds=60 * recording_length_minutes + 1):
                pass
            recording.stop()

        attrs = recording.attributes
        recording_path = recording.file['path']

    print(f"Recorded {attrs['duration_frames']} frames ({attrs['duration_seconds']} seconds)")
    print(f"into file: {recording.file['path']}")

    del recording
    with tables.open_file(recording_path) as h5_file:
        counts, bins = np.histogram(h5_file.root.spikes.col('channel'), bins=np.arange(65))

    baseline_fr = counts / (attrs['duration_frames'] / frames_per_second)

    assert attrs['frames_per_second'] == frames_per_second

    np.savez(baseline_fr_file, baseline_fr=baseline_fr, recording_path=recording_path,
             duration_frames=attrs['duration_frames'], allow_pickle=False)


def load_baseline_data():
    return np.load(baseline_fr_file)
