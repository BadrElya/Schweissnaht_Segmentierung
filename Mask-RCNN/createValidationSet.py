import os
import shutil
import random

# Pfad zum ursprünglichen Trainingsordner und zum neuen Validierungsordner
train_dir = 'dataset/train'
validation_dir = 'dataset/validation'

# Erstelle den Validierungsordner, falls er noch nicht existiert
os.makedirs(validation_dir, exist_ok=True)

# Liste aller Dateien im Trainingsordner
files = os.listdir(train_dir)

# Filtere nur Bild-Dateien und prüfe, ob eine entsprechende JSON-Datei existiert
paired_files = []
for file in files:
    if file.endswith(('.jpg', '.jpeg', '.png')):
        json_file = os.path.splitext(file)[0] + '.json'
        if json_file in files:
            paired_files.append((file, json_file))

# Wähle zufällig 30% dieser Paare aus
num_to_select = int(len(paired_files) * 0.3)
selected_pairs = random.sample(paired_files, num_to_select)

# Verschiebe die ausgewählten Paare in den Validierungsordner
for image_file, json_file in selected_pairs:
    shutil.move(os.path.join(train_dir, image_file), os.path.join(validation_dir, image_file))
    shutil.move(os.path.join(train_dir, json_file), os.path.join(validation_dir, json_file))

print(f"{len(selected_pairs)*2} Dateien wurden in den Ordner {validation_dir} verschoben.")
