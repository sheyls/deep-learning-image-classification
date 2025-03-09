import json
import numpy as np

CATEGORIES = {0: 'Cargo plane', 1: 'Helicopter', 2: 'Small car', 3: 'Bus', 4: 'Truck', 5: 'Motorboat', 6: 'Fishing vessel', 7: 'Dump truck', 8: 'Excavator', 9: 'Building', 10: 'Storage tank', 11: 'Shipping container'}

JSON_FILE = 'xview_recognition/xview_ann_train.json'
DATA_DIR = "/mnt/g/DataSets/Images"

with open(JSON_FILE) as ifs:
    JSON_DATA = json.load(ifs)
ifs.close()

COUNTS = dict.fromkeys(CATEGORIES.values(), 0)
ANNS = []


