# Flight Accident Analysis System Documentation

## Introduction
The Flight Accident Analysis System is a machine learning-powered tool designed to automate the analysis of flight accident reports. Utilizing BERT multi-task classification, this system aims to enhance aviation safety by providing insights into accident factors and causes.

## Installation Guide
To install the Flight Accident Analysis System, follow these steps:

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/wac0ku/aviatoin_project.git
   cd aviation_project
   ```

2. **Set Up a Virtual Environment**:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows use `.venv\Scripts\activate`
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

### Troubleshooting
- If you encounter issues during installation, ensure that you have Python 3.8+ and all required packages installed.

## Usage Instructions
To run the analysis, execute the following command:
```bash
python main.py
```

### Example Input
Place your PDF reports in the `data/` directory. The system will process these files and generate analysis reports.

## Configuration Options
To configure the system, modify `src/config.py` as follows:
- **Input Directory**: Set the path for the directory containing your PDF reports.
- **Output Directory**: Set the path for the directory where analysis reports will be saved.
- **Coding Keywords**: Customize the keywords used for analysis.

## Model Training
To train the model:
1. Uncomment the training section in `main.py`.
2. Ensure your PDF reports are prepared in the `data/` directory.
3. Run the command: `python main.py`.

### Cause Analysis with MISTRAL
The system utilizes MISTRAL for cause analysis. The `determine_cause` method in the `TextAnalyzer` class constructs a prompt that includes categorized evidence and the primary problem. MISTRAL then generates a professional analysis based on this prompt, providing insights into the cause chain and recommended preventive measures.


## Key Components
- **text_extractor.py**: Handles PDF text extraction.
- **text_analyzer.py**: Performs multi-task text analysis.
- **report_generator.py**: Generates automated reports based on analysis.
- **trainer.py**: Manages the model training pipeline.

## Contributing Guidelines
We welcome contributions to the Flight Accident Analysis System! Please submit issues and pull requests to help improve the project. For more details, refer to the `CONTRIBUTING.md` file.

## FAQs
- **What is the purpose of the HUGGINGFACE_TOKEN?**
  The HUGGINGFACE_TOKEN is required for accessing Hugging Face models. You can obtain it by creating an account on the Hugging Face website.

- **How can I contribute to the project?**
  Please submit issues and pull requests on GitHub. Refer to the `CONTRIBUTING.md` file for more details.

## License
This project is licensed under the MIT License.
