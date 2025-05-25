from roboflow import Roboflow

rf = Roboflow(api_key="Ffzx7jQXkLTgEc3bzzGy")
project = rf.workspace("dinesh-5zx7c").project("vehicle-detection-swjzd-x4cps")
version = project.version(1)
dataset = version.download("yolov8")