from PyPDF2 import PdfReader

class TextExtractor:
    """
    Eine Klasse zum Extrahieren von Text aus PDF-Dateien.
    """


    @staticmethod
    def extract_text_from_pdf(pdf_path):
        """
        Extrahiert Text aus einer PDF-Datei.

        Parameter:
            pdf_path (str): Der Pfad zur PDF-Datei.

        Rückgabe:
            str: Der extrahierte Text aus der PDF-Datei.
        """

        with open(pdf_path, "rb") as f:
            reader = PdfReader(f)
            return "\n".join([page.extract_text() for page in reader.pages])
