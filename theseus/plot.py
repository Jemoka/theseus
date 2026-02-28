import re
import wandb
from dataclasses import dataclass
from pathlib import Path
from queue import Queue, Empty
from threading import Thread
from typing import Any, Dict, List, Optional, Type

from theseus.config import field

BG = "#F5F4F1"
FG = "#1C1F24"
SPINE = "#444444"

TEAL = "#3A8FA3"
ORANGE = "#C46000"
GREEN = "#2A8A5E"
MAUVE = "#9E4A72"
VIOLET = "#6559A8"
RED = "#B83000"

PALETTE = [TEAL, ORANGE, GREEN, MAUVE, VIOLET, RED]
SERIF_STACK = ["Times New Roman", "Times", "DejaVu Serif", "serif"]


def apply_theme(
    context: str = "paper",
    font_scale: float = 1.35,
    palette: Optional[List[str]] = None,
    axes: Any = None,
) -> None:
    """
    Apply theme globally.

    Parameters
    ----------
    context    : seaborn context string
    font_scale : font scale multiplier
    palette    : list of hex colours; defaults to muted Jemoka palette
    axes       : Axes or array of Axes to label (a), (b), (c)...
    """

    import string
    import matplotlib as mpl
    import matplotlib.ticker as ticker
    import seaborn as sns
    import numpy as np

    if palette is None:
        palette = PALETTE

    sns.set_theme(
        context=context,
        style="ticks",
        palette=palette,
        font=SERIF_STACK[0],
        font_scale=font_scale,
        rc={
            "font.family": "serif",
            "font.serif": SERIF_STACK,
            "font.weight": "normal",
            "mathtext.fontset": "dejavuserif",
            "axes.titleweight": "normal",
            "axes.labelweight": "normal",
            "axes.titlesize": 10,
            "axes.labelsize": 9.5,
            "axes.titlecolor": FG,
            "axes.labelcolor": FG,
            "axes.titlepad": 9,
            "axes.labelpad": 5,
            "axes.linewidth": 0.9,
            "axes.edgecolor": SPINE,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.facecolor": BG,
            "axes.axisbelow": True,
            "axes.grid": True,
            "grid.color": "#D8D8D4",
            "grid.linewidth": 0.55,
            "grid.linestyle": "--",
            "grid.alpha": 0.65,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "xtick.top": False,
            "ytick.right": False,
            "xtick.major.width": 0.9,
            "ytick.major.width": 0.9,
            "xtick.major.size": 4.5,
            "ytick.major.size": 4.5,
            "xtick.major.pad": 5,
            "ytick.major.pad": 5,
            "xtick.color": SPINE,
            "ytick.color": SPINE,
            "xtick.labelcolor": FG,
            "ytick.labelcolor": FG,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            # Key setting: limit ticks to a sensible maximum
            "axes.formatter.use_mathtext": True,
            "axes.formatter.limits": (-4, 4),
            "lines.linewidth": 1.7,
            "lines.markersize": 5.5,
            "legend.frameon": True,
            "legend.framealpha": 0.95,
            "legend.edgecolor": "#CCCCCA",
            "legend.fancybox": False,
            "legend.labelcolor": FG,
            "legend.fontsize": 8.5,
            "legend.borderpad": 0.4,
            "legend.labelspacing": 0.35,
            "legend.handlelength": 1.4,
            "legend.handletextpad": 0.5,
            "figure.facecolor": BG,
            "savefig.facecolor": BG,
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.05,
            "patch.linewidth": 0.6,
        },
    )

    mpl.rcParams["axes.prop_cycle"] = mpl.cycler(color=palette)

    # seaborn's categorical palette resolver checks rcParams["axes.prop_cycle"]
    # but hue-keyed plots fall back to their own defaults. Setting both
    # "palette" keys ensures consistency.
    sns.set_palette(palette)

    # Patch Axes.__init__ so every new axes gets MaxNLocator with
    # nbins="auto" — matplotlib will compute how many ticks fit without
    # overlapping given the actual axis length in display units.
    _orig_init = mpl.axes.Axes.__init__

    def _auto_tick_init(self: Any, *args: Any, **kwargs: Any) -> None:
        _orig_init(self, *args, **kwargs)
        self.xaxis.set_major_locator(
            ticker.MaxNLocator(nbins="auto", steps=[1, 2, 2.5, 5, 10], prune="both")
        )
        self.yaxis.set_major_locator(
            ticker.MaxNLocator(nbins="auto", steps=[1, 2, 2.5, 5, 10], prune="both")
        )

    mpl.axes.Axes.__init__ = _auto_tick_init

    if axes is not None:
        if isinstance(axes, mpl.axes.Axes):
            axes = [axes]
        else:
            axes = np.asarray(axes).ravel().tolist()
        for ax, label in zip(axes, string.ascii_lowercase):
            ax.text(
                -0.12,
                1.02,
                f"({label})",
                transform=ax.transAxes,
                fontsize=11,
                fontweight="normal",
                fontfamily="serif",
                va="bottom",
                ha="right",
                color=FG,
            )


@dataclass
class PlotsConfig:
    save: bool = field("logging/plots/save", default=False)


class PlotsDispatcher:
    def __init__(
        self,
        model_cls: Type[Any],
        save: bool = False,
        save_dir: Optional[Path] = None,
    ) -> None:
        self.model_cls = model_cls
        self.save = save
        self.save_dir = save_dir
        self.queue: Queue[Any] = Queue(maxsize=8)
        self.stop_flag = False
        self.error: Optional[Exception] = None

        if self.save and self.save_dir:
            self.save_dir.mkdir(parents=True, exist_ok=True)

        self.thread = Thread(target=self._worker, daemon=True)
        self.thread.start()

    def submit(self, intermediates: Any, step: int) -> None:
        if self.error is not None:
            err = self.error
            self.error = None
            raise err
        self.queue.put((intermediates, step))

    def _worker(self) -> None:
        import matplotlib

        matplotlib.use("Agg")
        from matplotlib import pyplot as plt

        apply_theme()

        while True:
            try:
                item = self.queue.get(timeout=0.5)
            except Empty:
                if self.stop_flag:
                    break
                continue

            if self.stop_flag:
                break

            try:
                intermediates, step = item
                figures: Dict[str, Any] = self.model_cls.plot(intermediates)

                for name, fig in figures.items():
                    wandb.log({name: wandb.Image(fig)}, step=step)
                    if self.save and self.save_dir:
                        safe_name = re.sub(r"[^\w\-.]", "_", name)
                        fig.savefig(
                            self.save_dir / f"{safe_name}_step{step}.svg",
                            bbox_inches="tight",
                            pad_inches=0.3,
                        )
                    plt.close(fig)
            except Exception as e:
                self.error = e

    def close(self) -> None:
        self.stop_flag = True
        # drain queue
        while not self.queue.empty():
            try:
                self.queue.get_nowait()
            except Empty:
                break
        self.thread.join(timeout=1.0)

    def __del__(self) -> None:
        self.close()
