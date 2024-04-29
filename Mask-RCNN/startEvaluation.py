import subprocess

# Liste von Eingaben
inputs = [
    ("Testdaten_L", "Deck01", "train0"),
    ("Testdaten_L", "Deck01", "train1"),
    ("Testdaten_L", "Deck01", "train2"),

    ("Testdaten_L", "Deck02", "train0"),
    ("Testdaten_L", "Deck02", "train1"),
    ("Testdaten_L", "Deck02", "train2"),

    ("Testdaten_C", "Deck03", "train0"),
    ("Testdaten_C", "Deck03", "train1"),
    ("Testdaten_C", "Deck03", "train2"),
                
    ("Testdaten_C", "Deck04", "train0"),
    ("Testdaten_C", "Deck04", "train1"),
    ("Testdaten_C", "Deck04", "train2"),   

    ("Testdaten_C", "Deck05", "train0"),
    ("Testdaten_C", "Deck05", "train1"),
    ("Testdaten_C", "Deck05", "train2"),
                
    ("Testdaten_C", "Deck06", "train0"),
    ("Testdaten_C", "Deck06", "train1"),
    ("Testdaten_C", "Deck06", "train2"),     
    
    ("Testdaten_G", "Deck07", "train0"),
    ("Testdaten_G", "Deck07", "train1"),
    ("Testdaten_G", "Deck07", "train2"),
                
    ("Testdaten_G", "Deck08", "train0"),
    ("Testdaten_G", "Deck08", "train1"),
    ("Testdaten_G", "Deck08", "train2"),   
                
    ("Testdaten_G", "Deck09", "train0"),
    ("Testdaten_G", "Deck09", "train1"),
    ("Testdaten_G", "Deck09", "train2"),
                
    ("Testdaten_G", "Deck10", "train0"),
    ("Testdaten_G", "Deck10", "train1"),
    ("Testdaten_G", "Deck10", "train2"),  
    
]

# Iteriere über die Liste von Eingaben
for test_data_folder, deck, trainNumber in inputs:
    # Erstelle den Befehl, der das Skript mit den aktuellen Eingaben ausführt
    command = ['python', 'mrcnn_evaluation.py', test_data_folder, deck, trainNumber]
    
    # Führe den Befehl aus
    subprocess.run(command)
