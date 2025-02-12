"""
Dieses Modul enthält die Konfigurationseinstellungen für das Projekt, einschließlich 
Eingangs- und Ausgangsverzeichnissen sowie Schlüsselwörtern für die Kodierung.
"""

from pathlib import Path

class Config():
    """
    Eine Klasse zur Speicherung von Konfigurationseinstellungen für das Projekt.

    Attribute:
        INPUT_DIR (Path): Das Verzeichnis, in dem die Eingabedaten gespeichert sind.
        OUTPUT_DIR (Path): Das Verzeichnis, in dem die Ausgabedaten gespeichert werden.
        CODING_KEYWORDS (dict): Ein Dictionary mit Schlüsselwörtern für verschiedene Kategorien.
    """
    
    INPUT_DIR = Path("data/")
    OUTPUT_DIR = Path("reports/")
    CODING_KEYWORDS = {
        "PRIMARY_PROBLEM": [
            r"root cause",
            r"primary (failure|problem)",
            r"fundamental (issue|flaw)",
            r"main contributing factor"
        ],
        "MECHANICAL": [
            r"structural (failure|damage)",
            r"system malfunction",
            r"component (failure|wear)",
            r"mechanical fault"
        ],
        "HUMAN_FACTOR": [
            r"pilot error",
            r"crew resource management",
            r"human factors",
            r"misjudgment"
        ],
        "PROCEDURAL": [
            r"procedure violation",
            r"non-compliance",
            r"checklist (error|omission)",
            r"SOP deviation"
        ]
    }
