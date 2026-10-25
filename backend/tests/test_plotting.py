"""Charts never cross users (app/plotting.py, kernel.py, export.py, figures.py):
the app's renders stay out of pyplot's process-wide registry, and python_exec
collects and closes only the figures its own call made.

Run from backend/ with the repo's venv:  python -m unittest discover -s tests
"""

from __future__ import annotations

import sys
import tempfile
import threading
import unittest
import uuid
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib  # noqa: E402

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402

from app.api import export  # noqa: E402
from app.assistant import conversations, figures, tools  # noqa: E402

tools.load()

from app.assistant.tools import analysis  # noqa: E402

PNG = b"\x89PNG"
SUMMARY = {"affected_count": 3, "unaffected_count": 7, "total_features": 10}


class Isolation(unittest.TestCase):
    def setUp(self):
        self.conv = conversations.get_or_create(uuid.uuid4().hex)
        self.addCleanup(conversations.drop, self.conv.id)
        # A figure someone else is drawing with pyplot right now
        self.other = plt.figure()
        self.other.gca().plot([1, 2, 3])
        self.addCleanup(plt.close, self.other)

    def test_app_renders_stay_out_of_pyplot(self):
        before = plt.get_fignums()
        refuse = AssertionError("the app's renders must not create pyplot figures")
        with mock.patch.object(plt, "figure", side_effect=refuse), \
                mock.patch.object(plt, "close", side_effect=refuse):
            png = export.generate_barchart_png(SUMMARY, "Point", "Exposure")
            with tempfile.TemporaryDirectory() as tmp:
                figures.chart(Path(tmp) / "c.png", "bar", ["a", "b"],
                              [{"name": "s", "values": [1, 2]}])
        self.assertTrue(png.startswith(PNG))
        self.assertEqual(plt.get_fignums(), before)

    def test_python_exec_collects_only_its_own_figures(self):
        res = analysis.python_exec(self.conv, "plt.figure(); plt.plot([3, 1, 2]); None")
        self.assertTrue(res["ok"], res)
        self.assertEqual(len(res["figures"]), 1)
        self.assertIn(self.other.number, plt.get_fignums())  # not closed under its owner

    def test_renders_during_python_exec_do_not_land_in_its_files(self):
        code = (
            "import time\n"
            "fig = plt.figure(); plt.plot([1, 2])\n"
            "time.sleep(0.5)\n"
            "None"
        )
        pngs: list[bytes] = []

        def render_meanwhile():
            for _ in range(4):
                pngs.append(export.generate_barchart_png(SUMMARY, "Point", "Meanwhile"))

        worker = threading.Thread(target=render_meanwhile)
        worker.start()
        res = analysis.python_exec(self.conv, code)
        worker.join(30)
        self.assertTrue(res["ok"], res)
        self.assertEqual(len(res["figures"]), 1)
        self.assertEqual(len(pngs), 4)
        self.assertTrue(all(p.startswith(PNG) for p in pngs))


if __name__ == "__main__":
    unittest.main()
