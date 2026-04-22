from cl import ChannelSet, StimDesign, BurstDesign
import numpy as np
from common_cl_code.datasets import ArrayWithTime
from common_cl_code import grid_size, no_stim_channels


def make_spatial_footprint(sigma = 1., center=(0, 0)):
    """
    center between -1 and 1
    sigma beween 0 and 1
    """
    center = np.array(center) * 3.5
    sigma = sigma * 3.5

    rows, cols = np.meshgrid(
        np.arange(grid_size[0]) - (grid_size[0] - 1) / 2,
        np.arange(grid_size[1]) - (grid_size[1] - 1) / 2,
        indexing='ij'
    )

    spatial_footprint = np.exp(
        -(((rows - center[0]) ** 2 + (cols - center[1]) ** 2) / (2 * sigma ** 2))
    )
    spatial_footprint = spatial_footprint / spatial_footprint.sum()
    return spatial_footprint


def make_spatial_footprint_radial(sigma =.4, r=.7, theta=0, degrees=True):
    if degrees:
        theta = np.deg2rad(theta)
    return make_spatial_footprint(sigma=sigma, center=np.array([np.cos(theta), np.sin(theta)]) * r)


def make_sequence(seconds_per_rotation = 5, rotations=1., fps = 10, shuffle_axes = (), rng=None):
    seconds = seconds_per_rotation * rotations

    times = np.linspace(0, seconds, np.round(seconds*fps).astype(int)+1)[:-1]
    spatial_footprints = []

    for t in times:
        spatial_footprint = make_spatial_footprint_radial(r=.8, sigma=.3, theta=t/seconds_per_rotation * np.pi*2, degrees=False)
        spatial_footprints.append(spatial_footprint)

    spatial_footprints = np.array(spatial_footprints)


    if len(shuffle_axes):
        assert set(shuffle_axes).issubset({0,1}), "The only valid axes to shuffle are 0 (time) and 1 (space)"
        n_frames = spatial_footprints.shape[0]
        spatial_footprints = spatial_footprints.reshape((n_frames, 8*8))
        for axis in shuffle_axes:
            rng.shuffle(spatial_footprints, axis=axis)
        spatial_footprints = spatial_footprints.reshape((n_frames, 8,8))

    spatial_footprints = np.round(spatial_footprints, decimals=2)

    return ArrayWithTime(spatial_footprints, times)

def register_stim_plan(stim_plan, spatial_footprints, max_amplitude=2.5, phase_width_us=160, burst_hz=100, frame_time_s=0.04):
    unique_intensities = np.unique(spatial_footprints, return_inverse=True, sorted=True)[0]

    for spatial_footprint in spatial_footprints:
        for intensity in unique_intensities:
            channels = np.where(spatial_footprint.flatten() == intensity)[0]
            scaled_intensity = intensity/unique_intensities.max()
            if len(channels) and intensity != 0:
                stim_plan.stim(
                    ChannelSet([int(x) for x in channels if x not in no_stim_channels]),
                    StimDesign(phase_width_us, -max_amplitude * scaled_intensity, phase_width_us, max_amplitude * scaled_intensity),
                    BurstDesign(int(round(frame_time_s*burst_hz)), burst_hz)
                )
        stim_plan.sync(ChannelSet(list(set(range(64)) - no_stim_channels)))
