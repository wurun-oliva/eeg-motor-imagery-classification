"""Display a window prediction alongside a Mindlog mascot."""
from __future__ import annotations

from math import pi, sin
from pathlib import Path
from time import perf_counter

from matplotlib.image import imread
from matplotlib.transforms import Affine2D


ASSETS = Path(__file__).with_name("assets")
MASCOTS = ("Blueberry", "Tomato")
STATES = {
    "pending": ("Waiting", "#475569", "#E7EDF5"),
    "correct": ("✓ CORRECT", "#176344", "#DDF5E7"),
    "incorrect": ("✕ INCORRECT", "#A13722", "#FCE9E2"),
    "unscored": ("Cannot score", "#475569", "#E7EDF5"),
}


class PredictionFeedback:
    def __init__(self, figure):
        self.figure = figure
        self.mascot_index = 0
        self.state = "pending"
        self.motion_start = None
        self.images = {
            (mascot, pose): imread(ASSETS / f"{mascot.lower()}_{pose}.png")
            for mascot in MASCOTS
            for pose in ("idle", "correct", "incorrect")
        }

        panel = figure.add_axes([0.125, 0.127, 0.845, 0.198], facecolor="#F4F7FB")
        panel.set(xticks=[], yticks=[])
        for spine in panel.spines.values():
            spine.set_visible(False)
        for x, label in ((0.035, "PREDICTED"), (0.31, "TRUE CLASS"), (0.56, "RESULT")):
            panel.text(x, 0.77, label, fontsize=9, fontweight="bold", color="#526078",
                       va="center")
        self.prediction = panel.text(0.035, 0.50, "", fontsize=24, fontweight="bold",
                                     color="#14233B", va="center")
        self.truth = panel.text(0.31, 0.50, "", fontsize=18, color="#26364F", va="center")
        self.verdict = panel.text(0.56, 0.50, "", fontsize=13, fontweight="bold", va="center",
                                  bbox={"boxstyle": "round,pad=0.4", "edgecolor": "none"})
        self.details = panel.text(0.035, 0.16, "", fontsize=10, color="#475569", va="center")

        art = panel.inset_axes([0.77, 0.0, 0.21, 1.0])
        art.set_axis_off()
        self.motion = Affine2D()
        self.sprite = art.imshow(
            self.images[(self.mascot, "idle")], extent=(0.08, 0.92, 0.03, 0.87),
            transform=self.motion + art.transData,
        )
        art.set(xlim=(0, 1), ylim=(0, 1))

        self.timer = figure.canvas.new_timer(interval=30)
        self.timer.add_callback(self._advance)
        figure.canvas.mpl_connect("close_event", self._stop_motion)

    @property
    def mascot(self):
        return MASCOTS[self.mascot_index]

    def change_mascot(self):
        self.mascot_index = (self.mascot_index + 1) % len(MASCOTS)
        self._set_sprite()

    def clear(self, truth):
        self.truth.set_text(truth if truth is not None else "Unavailable")
        self.prediction.set_text("—")
        self.details.set_text("Click Predict to classify this window.")
        self._set_state("pending")

    def show_prediction(self, prediction, truth, probabilities):
        self.prediction.set_text(prediction)
        self.details.set_text(probabilities)
        if truth is None:
            state = "unscored"
        elif prediction == truth:
            state = "correct"
        else:
            state = "incorrect"
        self._set_state(state)
        if state in ("correct", "incorrect"):
            self.motion_start = perf_counter()
            self.timer.start()

    def _set_state(self, state):
        self._stop_motion()
        self.state = state
        caption, foreground, background = STATES[state]
        self.verdict.set_text(caption)
        self.verdict.set_color(foreground)
        self.verdict.get_bbox_patch().set_facecolor(background)
        self._set_sprite()

    def _set_sprite(self):
        pose = self.state if self.state in ("correct", "incorrect") else "idle"
        self.sprite.set_data(self.images[(self.mascot, pose)])

    def _stop_motion(self, _event=None):
        self.timer.stop()
        self.motion_start = None
        self.motion.clear()

    def _advance(self):
        if self.motion_start is None:
            return
        progress = (perf_counter() - self.motion_start) / 0.48
        if progress >= 1.0:
            self._stop_motion()
        else:
            lift = sin(pi * progress)
            self.motion.clear()
            if self.state == "correct":
                self.motion.translate(0, 0.10 * lift)
            else:
                self.motion.rotate_deg_around(0.5, 0.42, -13 * lift)
        self.figure.canvas.draw_idle()
