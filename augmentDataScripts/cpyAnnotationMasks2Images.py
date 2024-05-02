import os
import shutil
import argparse

def copy_json_files(input_folder, output_folder):
    # Überprüfe, ob der Ausgabeordner vorhanden ist. Wenn nicht, erstelle ihn.
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Durchlaufe den Eingabeordner und seine Unterordner
    for root, dirs, files in os.walk(input_folder):
        # Erstelle den Zielordner, indem der Ausgabepfad an den Eingabepfad angehängt wird
        output_root = root.replace(input_folder, output_folder)

        # Erstelle den Zielordner, wenn er nicht vorhanden ist
        if not os.path.exists(output_root):
            os.makedirs(output_root)

        # Durchlaufe alle Dateien im aktuellen Ordner
        for file in files:
            # Überprüfe, ob die Datei eine JSON-Datei ist
            if file.endswith('.json'):
                # Konstruiere die Pfade für die Eingabe- und Ausgabedateien
                input_file_path = os.path.join(root, file)
                output_file_path = os.path.join(output_root, file)

                # Kopiere die JSON-Datei in den Zielordner
                shutil.copy(input_file_path, output_file_path)

    print("Kopieren der JSON-Dateien abgeschlossen.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Kopieren von JSON-Dateien aus einem Ordner in einen anderen mit Beibehaltung der Ordnerstruktur.')
    parser.add_argument('input_folder', help='Pfad zum Eingabeordner')
    parser.add_argument('output_folder', help='Pfad zum Ausgabeordner')

    args = parser.parse_args()

    copy_json_files(args.input_folder, args.output_folder)
