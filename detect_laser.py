from detect_laser_line import view_line
from pathlib import Path

laser_path = Path("/home/viva/Fishsense/fishsense-lite-python-pipeline/laser-calibration-output-4-12-bot.dat")
calibration_path = Path("/home/viva/Fishsense/fishsense-lite-python-pipeline/calibration-output.dat")
data_path = Path("/home/viva/Fishsense/fishsense-lite-python-pipeline/data/laser_mask_data")

view_line(laser_path, calibration_path, data_path)