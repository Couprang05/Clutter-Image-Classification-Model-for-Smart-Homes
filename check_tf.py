# check_tf.py
try:
    import tensorflow as tf
    print("tensorflow import: OK")
    print("tf.__version__ =", tf.__version__)
    try:
        from tensorflow.keras.applications import EfficientNetB2
        print("EfficientNetB2: import OK")
    except Exception as e:
        print("EfficientNetB2 import FAILED ->", type(e).__name__, str(e))
        try:
            from tensorflow.keras.applications import EfficientNetB0
            print("EfficientNetB0: import OK (fallback)")
        except Exception as e2:
            print("EfficientNetB0 import FAILED ->", type(e2).__name__, str(e2))
except Exception as e_main:
    print("tensorflow import FAILED ->", type(e_main).__name__, str(e_main))
