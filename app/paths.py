"""Where the hand-installed files live.

Three things the app needs are deliberately not in git — they are too big, or
platform-specific, or licensed separately:

    models/   model.onnx, model.onnx.data, model.pt   (trained locally)
    sdk/      toupcam.py, toupcam.dll                 (ToupTek, installed by hand)

Both now sit in subfolders of app/ so the top level of the scanner folder holds
nothing but run.bat.  A rig that was set up before that change still has them
loose in app/ or in the folder above, so every lookup here checks the tidy
location first and then falls back, which means an old install keeps working
even if first_run_tidy.py never gets to move anything.
"""
import os
import sys

APP  = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(APP)                      # the folder holding run.bat

MODEL_DIRS = [os.path.join(APP, "models"), APP, ROOT]
SDK_DIRS   = [os.path.join(APP, "sdk"),    APP, ROOT]


def find_model(name: str) -> str:
    """Absolute path to a model file, preferring app/models/.

    Returns the tidy path when the file is nowhere to be found, so error
    messages name the place the file is supposed to go.
    """
    for d in MODEL_DIRS:
        p = os.path.join(d, name)
        if os.path.exists(p):
            return p
    return os.path.join(MODEL_DIRS[0], name)


def prepare_sdk_import() -> str:
    """Make `import toupcam` work wherever the SDK was installed.

    Returns the folder the SDK was found in, or "" if it is not installed —
    which is normal on a Mac and simply means the camera falls back to OpenCV.

    Two separate things have to be arranged.  toupcam.py has to be importable,
    which is a sys.path question; and the toupcam.dll it binds to has to be
    findable, which since Python 3.8 is NOT a PATH question on Windows — the
    DLL search no longer looks there, so the directory has to be registered
    explicitly or the import fails with a bare "DLL load failed".
    """
    for d in SDK_DIRS:
        if not os.path.exists(os.path.join(d, "toupcam.py")):
            continue
        if d not in sys.path:
            sys.path.append(d)
        if hasattr(os, "add_dll_directory"):           # Windows, py3.8+
            try:
                os.add_dll_directory(d)
            except OSError:
                pass
        # Belt and braces for any loader that still consults PATH.
        os.environ["PATH"] = d + os.pathsep + os.environ.get("PATH", "")
        return d
    return ""
