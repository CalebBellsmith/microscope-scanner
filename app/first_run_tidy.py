"""Move hand-installed files out of the top folder and into app/.

run.bat calls this before launching.  The scanner folder is what the operator
sees, and it should hold run.bat and nothing else — but the ToupTek SDK, the
trained models and any loose helper scripts were installed by hand, are not in
git, and so a pull cannot move them.  This does.

It is deliberately dull: an explicit list of names, no globbing, no deleting.
Anything not on the list is left exactly where it is.  Running it twice is a
no-op, and if a file is already in place the copy in the root is left alone
rather than overwriting the good one.
"""
import os
import shutil
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)

# filename → subfolder of app/ it belongs in
FILING = {
    "toupcam.py":       "sdk",
    "toupcam.dll":      "sdk",
    "libtoupcam.dylib": "sdk",
    "model.onnx":       "models",
    "model.onnx.data":  "models",
    "model.pt":         "models",
    "re_export.py":     "tools",
}


def tidy(root: str = _ROOT, app: str = _HERE) -> list:
    """File away anything on the list that is loose in `root`."""
    moved = []
    for name, sub in FILING.items():
        src = os.path.join(root, name)
        if not os.path.isfile(src):
            continue
        dst_dir = os.path.join(app, sub)
        dst = os.path.join(dst_dir, name)
        if os.path.exists(dst):
            # Already filed.  Leave the stray copy rather than clobbering a
            # good file with one that might be older.
            continue
        try:
            os.makedirs(dst_dir, exist_ok=True)
            shutil.move(src, dst)
            moved.append(f"{name} -> app/{sub}/")
        except OSError as e:
            # A locked or read-only file must not stop the app from starting;
            # every lookup still falls back to the old location.
            print(f"  could not move {name}: {e}")
    return moved


if __name__ == "__main__":
    done = tidy()
    if done:
        print("Tidying the scanner folder:")
        for line in done:
            print("  " + line)
    sys.exit(0)          # never block the launch
