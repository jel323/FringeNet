import os
import sys

"""
This file should be placed and ran in the folder that you want all of this project to reside in.
The directory management established in this file is assumed for the files in the project to run.
"""

path = sys.path[0]

os.makedirs(os.path.join(path, "Data"), exist_ok=True)
os.makedirs(os.path.join(path, "DirectoryFunctions"), exist_ok=True)
os.makedirs(os.path.join(path, "FringeFunctions"), exist_ok=True)
os.makedirs(os.path.join(path, "ImageFunctions"), exist_ok=True)
os.makedirs(os.path.join(path, "ApplyModel"), exist_ok=True)
os.makedirs(os.path.join(path, "Losses"), exist_ok=True)
os.makedirs(os.path.join(path, "Models"), exist_ok=True)
os.makedirs(os.path.join(path, "TrainingFunctions"), exist_ok=True)
os.makedirs(os.path.join(path, "util"), exist_ok=True)
