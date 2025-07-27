try:
    import flask
    import tensorflow as tf
    import pandas as pd
    import numpy as np
    import sklearn
    print("✅ All packages imported successfully!")
    print(f"Python version: {tf.__version__}")
    print(f"TensorFlow version: {tf.__version__}")
    print(f"Pandas version: {pd.__version__}")
except ImportError as e:
    print(f"❌ Import error: {e}")