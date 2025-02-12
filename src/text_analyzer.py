import re
import os
import gc
import torch
from rich.console import Console

from src.config import Config

from transformers import pipeline
from huggingface_hub import login

# Hugging Face anbindung
HF_TOKEN = os.getenv("HUGGINGFACE_TOKEN")
login(token=HF_TOKEN)

console = Console()

class TextAnalyzer:
    """
    Eine Klasse zur Analyse von Texten mit einem vortrainierten Modell.

    Attribute:
        coding_keywords (dict): Ein Dictionary mit Schlüsselwörtern für die Kodierung.
        model: Das vortrainierte Modell für die Textanalyse.
        tokenizer: Der Tokenizer für die Textverarbeitung.
        categories (list): Eine Liste der Kategorien für die Klassifikation.
        device (torch.device): Das Gerät, auf dem das Modell trainiert wird (CPU oder GPU).
        _model_loaded (bool): Statusflag, ob das Modell geladen ist.
    """

    def __init__(self, coding_keywords):
        self.coding_keywords = coding_keywords
        self.model = None
        self.tokenizer = None
        self.categories = list(coding_keywords.keys())
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self._model_loaded = False # Neuer Status-Flag
    
    def is_ready(self):
        """
        Überprüft, ob das Modell und der Tokenizer bereit sind für die Analyse.

        Rückgabe:
            bool: True, wenn das Modell und der Tokenizer bereit sind, andernfalls False.
        """

        return self._model_loaded and self.model is not None and self.tokenizer is not None

    def __del__(self):
        """Cleanup resources on deletion"""
        self.cleanup()

    def cleanup(self):
        """
        Führt eine explizite Bereinigung der Ressourcen durch.
        """

        if self.model:
            self.model.cpu()
            del self.model
            self.model = None
        if hasattr(self, 'tokenizer'):
            del self.tokenizer
            self.tokenizer = None
        torch.cuda.empty_cache()
        gc.collect()

    def set_model_and_tokenizer(self, model, tokenizer):
        """
        Setzt das Modell und den Tokenizer für die Textanalyse.

        Parameter:
            model: Das vortrainierte Modell.
            tokenizer: Der Tokenizer für die Textverarbeitung.

        Rückgabe:
            None
        """

        try:
            # Validate inputs
            if model is None or tokenizer is None:
                raise ValueError("Model and tokenizer must be provided")

            # Cleanup previous resources
            self.cleanup()

            # Set new model and tokenizer
            self.model = model
            self.tokenizer = tokenizer

            # Validate model architecture
            required_attributes = [
                'category_classifier', 
                'problem_classifier'
            ]
            for attr in required_attributes:
                if not hasattr(self.model, attr):
                    raise ValueError(f"Invalid model: Missing {attr}")

            # Move to device and set evaluation mode
            self.model.to(self.device)
            self.model.eval()

            # Set loaded status
            self._model_loaded = True

        except Exception as e:
            console.print(f"[bold red]Model Setup Error: {e}[/bold red]")
            self._model_loaded = False
            raise

    def analyze_text(self, text):
        """
        Führt die Textanalyse durch und verwaltet den Speicher.

        Parameter:
            text (str): Der zu analysierende Text.

        Rückgabe:
            dict: Ein Dictionary mit den Analyseergebnissen, einschließlich Kodierungen, primärem Problem, Ursache und Fehlerkette.
        """

        if not self.is_ready():
            raise ValueError("Model and tokenizer must be set before analysis.")

        try:
            # Move model to device
            self.model.to(self.device)

            # Prepare inputs
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=512
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            # Get predictions
            with torch.no_grad():
                outputs = self.model(**inputs)

                # Kategorien-Wahrscheinlichkeiten (Multi-Label)
                category_probs = torch.sigmoid(outputs.category_logits).cpu()

                # Problem-Wahrscheinlichkeiten (Multi-Class)
                problem_probs = torch.softmax(outputs.problem_logits, dim=1).cpu()  # dim=1 für Batch × Klassen

            # Process results
            return self._process_predictions(text, category_probs, problem_probs)

        except Exception as e:
            raise RuntimeError(f"Analysefehler: {str(e)}")

        finally:
            torch.cuda.empty_cache()
    
    def determine_cause(self, primary_problem, coding_results):
        """
        Analysiert die Ursache des Problems mithilfe des Mistral-Modells.

        Parameter:
            primary_problem (dict): Das primäre Problem mit Kategorie und Beweisen.
            coding_results (dict): Die Ergebnisse der Kodierung.

        Rückgabe:
            str: Die generierte Analyse der Ursache.
        """

        try:
            # Initialisiere die Pipeline einmalig (nicht bei jedem Aufruf)
            if not hasattr(self, 'mistral_pipe'):
                self.mistral_pipe = pipeline(
                    "text-generation",
                    model="mistralai/Mistral-7B-Instruct-v0.3",
                    device=0 if torch.cuda.is_available() else -1,
                    torch_dtype=torch.float16,
                    device_map="auto"
                )

            # Erweitere den Prompt mit Kategoriendefinitionen
            categories_prompt = "\n".join(
                f"- {cat}: {', '.join(keys)}"
                for cat, keys in Config.CODING_KEYWORDS.items()
            )

            prompt = f"""
            Als Flugunfallexperte analysiere diesen Vorfall anhand folgender Kategorien:
            {categories_prompt}

            Aktueller Fall:
            1️⃣ Primärproblem: {primary_problem['category']}
            2️⃣ Beweise: {primary_problem['evidence'] or 'Keine direkten Beweise'}
            3️⃣ Zusatzbefunde: {[f"{k}: {v}" for k,v in coding_results.items() if k != primary_problem['category']]}

            Erstelle eine professionelle Analyse:
            - Ursachenkette gemäß SHELL-Modell
            - Kritische Schlüsselfaktoren
            - Empfohlene Präventionsmaßnahmen
            """

            # Generiere die Antwort
            response = self.mistral_pipe(
                prompt,
                max_new_tokens=300,
                do_sample=True,
                temperature=0.7,
                top_p=0.9
            )[0]['generated_text']

            return response.split("Empfohlene Präventionsmaßnahmen")[-1].strip()

        except Exception as e:
            console.print(f"[red]Analysefehler: {e}[/red]")

    def generate_failure_chain(self, primary_problem, coding_results):
        """
        Generiert eine visuell strukturierte Fehlerkette basierend auf dem primären Problem und den Kodierungsergebnissen.

        Parameter:
            primary_problem (dict): Das primäre Problem mit Kategorie und Beweisen.
            coding_results (dict): Die Ergebnisse der Kodierung.

        Rückgabe:
            str: Eine formatierte Fehlerkette als String.
        """

        chain = [
            f"🔥 [bold]Primäres Problem[/bold]: {primary_problem['category']}",
            f"   - Confidence: {primary_problem['confidence']*100:.1f}%",
            f"   - Key Evidence: {', '.join(primary_problem['evidence'][:3]) or 'N/A'}"
        ]

        # Füge hierarchische Unterkategorien hinzu
        for cat, evidence in coding_results.items():
            if cat != primary_problem['category'] and evidence:
                chain.append(f"├─➤ [yellow]{cat}[/yellow]")
                chain.append(f"│  └─ {' | '.join(evidence[:2])}")

        chain.append("└─ [italic]...weitere Faktoren[/italic]")
        return "\n".join(chain)

    def _process_predictions(self, text, category_probs, problem_probs):
        """
        Verarbeitet die Vorhersagen des Modells.

        Parameter:
            text (str): Der analysierte Text.
            category_probs (torch.Tensor): Die Wahrscheinlichkeiten für die Kategorien.
            problem_probs (torch.Tensor): Die Wahrscheinlichkeiten für das primäre Problem.

        Rückgabe:
            dict: Ein Dictionary mit den Kodierungsergebnissen und dem primären Problem.
        """

        coding_results = {}

        # Process categories
        for idx, category in enumerate(self.categories):
            if category_probs[0][idx] > 0.5:  # [Batch, Kategorien]
                matches = []
                for pattern in self.coding_keywords[category]:
                    matches.extend(re.findall(pattern, text, re.IGNORECASE))
                coding_results[category] = list(set(matches))
            else:
                coding_results[category] = []

        # Process primary problem
        # Korrektur 1: Richtige Indexierung für 2D-Tensor
        primary_idx = problem_probs.argmax(dim=1).item()  # dim=1 für Klassen-Dimension
        primary_category = self.categories[primary_idx]

        # Korrektur 2: Tensor-Indexierung anpassen
        primary_problem = {
            'category': primary_category,
            'evidence': coding_results.get(primary_category, [])[:3],
            'confidence': problem_probs[0, primary_idx].item()  # [Batch, Klasse]
        }

        return {
            "codings": coding_results,
            "primary_problem": primary_problem,
            "cause": self.determine_cause(primary_problem, coding_results),
            "failure_chain": self.generate_failure_chain(primary_problem, coding_results)
        }
