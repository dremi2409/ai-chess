import json
import os

file_path = os.getcwd() + "\\PyXiangQi\\models\\Mapping.json"
with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)  # Parse JSON into Python object

print(data)