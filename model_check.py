# which_model_used.py
import inspect, importlib, sys

# try the two possible model modules you may have
for mname in ("model", "model_EfficientNetB2"):
    try:
        mod = importlib.import_module(mname)
        print(f"Imported module: {mname} ->", inspect.getsourcefile(mod))
    except Exception as e:
        print(f"Could not import {mname}: {type(e).__name__}: {e}")

# Also show current working dir for clarity
import os
print("cwd:", os.path.abspath(os.getcwd()))
