import pathlib
import datetime
from matplotlib.animation import FFMpegWriter, PillowWriter
import functools
import warnings
import numpy as np
import matplotlib.pyplot as plt

from .datasets import ArrayWithTime


class AnimationManager:
    """
    Examples
    --------
    >>> tmp_path = getfixture('tmp_path')  # this is mostly for the doctesting framework
    >>> with AnimationManager(outdir=tmp_path) as am:
    ...     for i in range(2):
    ...         for ax in am.axs.flatten():
    ...             ax.cla()
    ...         # animation things would go here
    ...         am.grab_frame()
    ...     fpath = am.outfile
    >>> assert fpath.is_file()
    """
    def __init__(self, outdir, filename_stem=None, n_rows=1, n_cols=1, fps=20, dpi=100, filetype="mp4", figsize=(10, 10), projection='rectilinear', make_axs=True, fig=None):
        outdir = pathlib.Path(outdir)


        if filename_stem is None:
            time_string = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
            filename_stem = f"movie_{time_string}-{str(hash(id(self)))[-3:]}"

        self.filetype = filetype
        self.outfile = pathlib.Path(outdir).resolve() / f"{filename_stem}.{filetype}"
        Writer = FFMpegWriter
        if filetype == 'gif':
            Writer = PillowWriter
        if filetype == 'webm':
            Writer = functools.partial(FFMpegWriter, codec='libvpx-vp9')

        self.movie_writer = Writer(fps=fps, bitrate=-1)
        if fig is None:
            if make_axs:
                self.fig, self.axs = plt.subplots(n_rows, n_cols, figsize=figsize, layout='constrained', squeeze=False, subplot_kw={'projection': projection})
            else:
                self.fig = plt.figure(figsize=figsize, layout='constrained')
        else:
            self.fig = fig
        self.movie_writer.setup(self.fig, self.outfile, dpi=dpi)
        self.seen_frames = 0
        self.finished = False

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.seen_frames:
            self.finish()
        else:
            warnings.warn('closed without any frame grabs')

    def finish(self):
        if not self.finished:
            self.movie_writer.finish()
            self.finished = True

    def grab_frame(self):
        self.movie_writer.grab_frame()
        self.seen_frames += 1

    def display_video(self, embed=False, width=None):
        from IPython import display
        if self.filetype == 'gif':
            import base64
            with open(self.outfile, 'rb') as f:
                data = base64.b64encode(f.read()).decode('ascii')
            display.display(display.HTML(f'<img width="{width}" src="data:image/gif;base64,{data}"/>'))
        else:
            display.display(display.Video(self.outfile, embed=embed, width=width))



def plot_history_with_tail(ax, data, current_t, tail_length=1, scatter_all=True, dim_1=0, dim_2=1, hist_bins=None, invisible=False, scatter_alpha=.1, scatter_s=5):
    """
    Examples
    --------
    >>> fig, ax = plt.subplots()
    >>> X = np.random.normal(size=(100,2))
    >>> X = ArrayWithTime.from_notime(X)
    >>> plot_history_with_tail(ax, data=X, current_t=75, tail_length=4, scatter_alpha=1)
    """
    ax.cla()

    s = np.ones_like(data.t).astype(bool)
    if scatter_all:
        s = data.t <= current_t
    if hist_bins is None:
        ax.scatter(data[s,dim_1], data[s,dim_2], s=scatter_s, c='gray', edgecolors='none', alpha= 0 if invisible else scatter_alpha)
        back_color = 'white'
        forward_color = 'C0'
    else:
        s = s & np.isfinite(data).all(axis=1)
        ax.hist2d(data[s,dim_1], data[s,dim_2], bins=hist_bins)
        back_color = 'black'
        forward_color = 'white'


    linewidth = 2
    size = 10
    s = (current_t - tail_length < data.t) & (data.t <= current_t)
    ax.plot(data[s, dim_1], data[s, dim_2], color=back_color, linewidth=linewidth * 1.5, alpha= 0 if invisible else 1)
    ax.scatter(data[s, dim_1][-1], data[s, dim_2][-1], s=size * 1.5, color=back_color, alpha= 0 if invisible else 1)
    ax.plot(data[s, dim_1], data[s, dim_2], color=forward_color, linewidth=linewidth, alpha= 0 if invisible else 1)
    ax.scatter(data[s,dim_1][-1], data[s,dim_2][-1], color=forward_color, s=size, zorder=3, alpha= 0 if invisible else 1)
    ax.axis('off')
