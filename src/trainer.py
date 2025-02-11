import re
import numpy as np
import gc
import os
from PyPDF2 import PdfReader

from pathlib import Path
import torch

from torch.utils.data import Dataset, DataLoader, random_split
from torch.optim import AdamW

from transformers import (
    BertTokenizer, 
    BertForSequenceClassification,
    get_linear_schedule_with_warmup
)

from src.config import Config

class AccidentReportDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        inputs = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        if len(self.texts) == 0 or len(self.labels) == 0:
            raise ValueError("Error: No texts or labels available to create dataset!")
        
        return {
            'input_ids': inputs['input_ids'].squeeze(),
            'attention_mask': inputs['attention_mask'].squeeze(),
            'labels': {
                'categories': torch.tensor(self.labels[idx]['categories'], dtype=torch.float),
                'primary_problem': torch.tensor(self.labels[idx]['primary_problem'], dtype=torch.long)
            }
        }

class ModelTrainer:
    def __init__(self, input_dir: Path):
        self.input_dir = input_dir
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.tokenizer = None
        self.categories = list(Config.CODING_KEYWORDS.keys())

    def __del__(self):
        """Cleanup resources on deletion"""
        self.cleanup()

    def cleanup(self):
        """Explicit cleanup of resources"""

        if hasattr(self, 'model') and self.model:
            self.model.cpu()
            del self.model
            self.model = None

        if hasattr(self, 'tokenizer'):
            del self.tokenizer
            self.tokenizer = None
        torch.cuda.empty_cache()
        gc.collect()
    
    def train_model(self, progress=None):
        """Train the BERT model with memory management"""
        try:
            # Initialize model and tokenizer
            self.cleanup()  # Ensure clean state
            self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
            self.model = BertForMultiTaskClassification.from_pretrained(
                'bert-base-uncased',
                num_categories=len(self.categories)
            ).to(self.device)

            # Prepare data
            train_loader, val_loader = self._prepare_data_loaders()

            if not train_loader:
                raise ValueError("No valid training data found!")

            # Training setup
            optimizer = AdamW(self.model.parameters(), lr=1e-5)
            scheduler = get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=0,
                num_training_steps=5 * len(train_loader)
            )

            best_val_loss = float('inf')
            best_model_state = None

            # Training loop
            for epoch in range(5):
                # Train
                self._train_epoch(train_loader, optimizer, scheduler, progress)

                # Validate
                val_loss = self._validate(val_loader)

                # Save best model
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_model_state = {
                        'state_dict': self.model.state_dict(),
                        'val_loss': val_loss
                    }

                # Clear memory after each epoch
                torch.cuda.empty_cache()
                gc.collect()

            # Define output directory
            output_dir = Path('models/accident-classifier')
            output_dir.mkdir(parents=True, exist_ok=True)

            # Save best model state and tokenizer
            if best_model_state:
                # Save model state
                torch.save(best_model_state, output_dir / 'pytorch_model.bin')

                # Save model config
                self.model.config.save_pretrained(output_dir)

                # Save tokenizer
                self.tokenizer.save_pretrained(output_dir)

            return self.model, self.tokenizer

        except Exception as e:
            self.cleanup()
            raise e

        finally:
            torch.cuda.empty_cache()
            gc.collect()
            
    def _train_epoch(self, train_loader, optimizer, scheduler, progress=None):
        """Train one epoch with memory management"""
        self.model.train()
            
        epoch_task_id = progress.add_task(
            description="[cyan] Training batches...",
            total=len(train_loader)
        )
        
        for batch_idx, batch in enumerate(train_loader, 1):
            # Move batch to GPU
            batch = {k: v.to(self.device) if torch.is_tensor(v) else v 
                    for k, v in batch.items()}
            
            # Update progress
            progress.update(epoch_task_id, advance=1)
            
            # Forward pass
            optimizer.zero_grad()
            outputs = self.model(**batch)
            loss = outputs.loss  # Access the loss
            
            # Backward pass
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            # Update Progress
            progress.update(epoch_task_id, advance=1)
            
            # Clear memory
            del outputs, loss
            batch = {k: v.cpu() if torch.is_tensor(v) else v 
                    for k, v in batch.items()}
        
        progress.remove_task(epoch_task_id)
        torch.cuda.empty_cache()


    def _validate(self, val_loader):
        """Validate model with memory management"""
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for batch in val_loader:
                # Move batch to GPU
                batch = {k: v.to(self.device) if torch.is_tensor(v) else v 
                        for k, v in batch.items()}
                
                outputs = self.model(**batch)
                total_loss += outputs.loss.item()  # Access the loss
                
                # Clear memory
                del outputs
                batch = {k: v.cpu() if torch.is_tensor(v) else v 
                        for k, v in batch.items()}
        
        torch.cuda.empty_cache()
        return total_loss / len(val_loader)

    def _prepare_data_loaders(self):
        """Prepare data loaders for training and validation"""
        texts = []
        labels = []
        categories_order = list(Config.CODING_KEYWORDS.keys())  # WICHTIG: Reihenfolge festlegen

        if not categories_order:
            raise ValueError("Config.CODING_KEYWORDS is empty or not properly defined")

        # Überprüfe Input-Verzeichnis
        if not self.input_dir.exists():
            raise FileNotFoundError(f"Input directory {self.input_dir} does not exist")

        pdf_files = list(self.input_dir.glob("*.pdf"))
        if not pdf_files:
            raise ValueError(f"No PDF files found in {self.input_dir}")

        for pdf_file in pdf_files:
            try:
                with open(pdf_file, 'rb') as f:
                    reader = PdfReader(f)

                    text = " ".join([page.extract_text() or "" for page in reader.pages]).strip()

                    if not text:
                        continue  # Überspringe leere Texte

                    # Label-Generierung
                    keyword_counts = []
                    for category in categories_order:
                        patterns = [re.compile(rf"\b{re.escape(kw)}\b", flags=re.IGNORECASE) 
                                   for kw in Config.CODING_KEYWORDS[category]]
                        keyword_counts.append(
                            sum(1 for pattern in patterns if pattern.search(text))
                        )
                    
                    if not keyword_counts:
                        print(f"Warning: No keywords found in {pdf_file}")
                        keyword_counts = [0] * len(categories_order) # Setze leere Kategorien auf Null
                    
                    label = {
                        'categories': [1 if count > 0 else 0 for count in keyword_counts],
                        'primary_problem': np.argmax(keyword_counts) if any(keyword_counts) else 0
                    }

                    texts.append(text)
                    labels.append(label)

            except Exception as e:
                print(f"Error processing {pdf_file}: {e}")
                continue

        # Überprüfe auf gültige Daten
        if len(texts) == 0:
            raise ValueError("No valid training data after processing PDFs")

        # Dataset-Split mit random_split
        full_dataset = AccidentReportDataset(texts, labels, self.tokenizer)
        train_size = int(0.8 * len(full_dataset))
        val_size = len(full_dataset) - train_size

        if len(full_dataset) < 2:
            raise ValueError("Dataset too small for splitting")

        # Sicherstellen, dass die Größen > 0
        if train_size == 0 or val_size == 0:
            raise ValueError(
                f"Dataset too small for split. Total samples: {len(full_dataset)}"
            )
        

        if len(full_dataset) == 0:
            raise ValueError("Dataset is empty")

        train_dataset, val_dataset = random_split(
            full_dataset, 
            [train_size, val_size],
            generator=torch.Generator().manual_seed(42)
        )

        return (
            DataLoader(train_dataset, batch_size=16, shuffle=True),
            DataLoader(val_dataset, batch_size=16, shuffle=False)
        )

class BertForMultiTaskClassification(BertForSequenceClassification):
    """Custom BERT model for multi-task classification"""
    def __init__(self, config, num_categories):
        super().__init__(config)
        self.num_categories = num_categories
        
        # Nur diese beiden Classifier behalten
        self.category_classifier = torch.nn.Linear(config.hidden_size, num_categories)

        self.problem_classifier = torch.nn.Linear(config.hidden_size, num_categories)

        # Original Classifier deaktivieren
        self.classifier = torch.nn.Identity()

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, 
                position_ids=None, head_mask=None, inputs_embeds=None, 
                labels=None, output_attentions=None, output_hidden_states=None,
                return_dict=None):
        
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )
        
        pooled_output = outputs[1]
        
        # Kategorien-Logits (Multi-Label)
        category_logits = self.category_classifier(pooled_output)  # Shape [Batch, num_categories]
        
        # Primäres Problem (Multi-Class)
        problem_logits = self.problem_classifier(pooled_output)  # Shape [Batch, num_categories]
        
        # Sicherstellen, dass die Ausgaben die richtige Form haben
        if problem_logits.dim() == 1:
            problem_logits = problem_logits.unsqueeze(0) # Form [1, num_categories]

        # Loss-Berechnung
        loss = None
        if labels is not None:

            # Multi-Label Loss für Kategorien
            category_loss_fct = torch.nn.BCEWithLogitsLoss()
            category_loss = category_loss_fct(
                category_logits, 
                labels['categories']
            )
            
            # Multi-Class Loss für primäres Problem
            problem_loss_fct = torch.nn.CrossEntropyLoss()
            problem_loss = problem_loss_fct(
                problem_logits,
                labels['primary_problem'] # Muss ein Klassenindex sein (LongTensor)
            )
            
            loss = category_loss + problem_loss

        return MultiTaskOutput(
            category_logits=category_logits,
            problem_logits=problem_logits,
            loss=loss
        )
        
class MultiTaskOutput:
    """Container for multi-task model outputs"""
    def __init__(self, category_logits, problem_logits, loss=None):
        self.category_logits = category_logits
        self.problem_logits = problem_logits
        self.loss = loss
