from pathlib import Path
from rich.console import Console
from rich.progress import Progress, TextColumn, BarColumn, TimeRemainingColumn, SpinnerColumn, TimeElapsedColumn
from rich.traceback import install
import torch
from transformers import BertTokenizer

from src.text_extractor import TextExtractor
from src.text_analyzer import TextAnalyzer
from src.report_generator import ReportGenerator
from src.trainer import ModelTrainer, BertForMultiTaskClassification
from src.config import Config

install(show_locals=True)

console = Console()
progress_columns = [
    TextColumn("[bold blue]{task.description}"), 
    BarColumn(),
    TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
    TimeRemainingColumn()
]

MODLE_DIR = "models/accident-classifier"

def load_trained_model(config_keywords):
    try:
        console.print("[bold yellow]Loading trained model...[/bold yellow]")
        
        # Initialize analyzer with coding keywords
        analyzer = TextAnalyzer(config_keywords)
        
        # Load model architecture
        model = BertForMultiTaskClassification.from_pretrained(
            MODLE_DIR, 
            num_categories=len(config_keywords)
        )
        
        # Load saved model state
        model_path = Path("models/accident-classifier/pytorch_model.bin")
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        state_dict = torch.load(model_path)
        model.load_state_dict(state_dict['state_dict'])
        
        # Load tokenizer
        tokenizer_path = Path("models/accident-classifier")
        if not tokenizer_path.exists():
            raise FileNotFoundError(f"Tokenizer directory not found: {tokenizer_path}")
        
        tokenizer = BertTokenizer.from_pretrained(str(tokenizer_path))
        
        # Set model and tokenizer in analyzer
        analyzer.set_model_and_tokenizer(model, tokenizer)
        
        console.print("[bold green]✓ Model loaded successfully[/bold green]")
        return analyzer
    
    except Exception as e:
        console.print(f"[bold red]Model Loading Error: {e}[/bold red]")
        console.print_exception(show_locals=True)
        return None

def main():
    # Create necessary directories
    Config.OUTPUT_DIR.mkdir(exist_ok=True)
    Path("models").mkdir(exist_ok=True)
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeRemainingColumn(),
        TimeElapsedColumn(),
        refresh_per_second=10,
        console=console
    ) as progress:

# -------------------------------------------------------------

        # Training phase
        console.print("\n[bold blue]Phase 1: Model Training[/bold blue]")
        trainer = ModelTrainer(Config.INPUT_DIR)

        try:
            model, tokenizer = trainer.train_model(progress)
            console.print("[bold green]✓ Model training completed successfully![/bold green]")

            # Load trained model

            
            if not analyzer or not analyzer.is_ready():
                console.print("[bold red]❌ Analyzer not initialized![/bold red]")
                return

        except Exception as e:
            console.print(f"[bold red]Training Error: {e}[/bold red]")
            console.print_exception(show_locals=True)
            return
        

# -------------------------------------------------------------
        
        # Analysis phase
        analyzer = load_trained_model(Config.CODING_KEYWORDS)

        console.print("\n[bold blue]Phase 2: Report Analysis[/bold blue]")
        pdf_files = list(Config.INPUT_DIR.glob("*.pdf"))
        
        # Analysis Task
        analysis_task = progress.add_task(
            "[bold]Analyzing reports[/bold]",
            total=len(pdf_files)
        )

        for pdf_path in pdf_files:
            try:
                # Extract text
                text = TextExtractor.extract_text_from_pdf(pdf_path)
                if not text.strip():
                    console.print(f"[yellow]⚠ Skipping empty PDF: {pdf_path.name}[/yellow]")
                    continue
                
                # Perform analysis
                analysis = analyzer.analyze_text(text)
                
                # Generate report
                report_content = ReportGenerator.generate_report(pdf_path.stem, analysis)
                
                # Save report
                output_path = Config.OUTPUT_DIR / f"{pdf_path.stem}_analysis.txt"
                ReportGenerator.save_report(output_path, report_content)
                progress.console.print(f"✅ [bold green]Report saved:[/bold green] {output_path}")
                
            except Exception as e:
                console.print(f"[bold red]❌ Analysis error for {pdf_path.name}: {e}[/bold red]")
                console.print_exception(show_locals=True)
            
            progress.update(analysis_task, advance=1)

# -------------------------------------------------------------

        # Final summary
        total_reports = len(pdf_files)
        console.print(f"\n[bold blue]Analysis Complete![/bold blue]")
        console.print(f"Processed {total_reports} reports")
        console.print(f"Results saved in: {Config.OUTPUT_DIR}")

# -------------------------------------------------------------

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        console.print("[bold yellow]⚠ Process cancelled by user![/bold yellow]")
    except Exception as e:
        console.print(f"[bold red]❌ Critical error in main(): {e}[/bold red]")