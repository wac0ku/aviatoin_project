# Flight Accident Analysis System

![Build Status](https://img.shields.io/badge/build-passing-brightgreen)
![License](https://img.shields.io/badge/license-MIT-blue)
![Version](https://img.shields.io/badge/version-1.0.3-orange)

## Overview
The Flight Accident Analysis System is a machine learning-powered tool designed to automate the analysis of flight accident reports. Utilizing BERT multi-task classification, this system aims to enhance aviation safety by providing insights into accident factors and causes.

## Table of Contents
- [Features](#features)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [Model Training](#model-training)
- [Key Components](#key-components)
- [Environment Variables](#environment-variables)
- [Contributing](#contributing)
- [License](#license)

## Features
- PDF text extraction for efficient data processing
- Multi-label classification of accident factors to identify key issues
- Automated cause analysis to streamline investigation processes
- Detailed report generation for comprehensive documentation

## Prerequisites
- Python 3.8+
- PyTorch
- Transformers
- Rich
- PyPDF2

## Installation
To install the Flight Accident Analysis System, follow these steps:
```bash
git clone https://github.com/wac0ku/flight-accident-analysis.git
cd flight-accident-analysis
python -m venv .venv
source .venv/bin/activate  # On Windows use `.venv\Scripts\activate`
pip install -r requirements.txt
```

## Configuration
To configure the system, modify `src/config.py` as follows:
- **Input Directory**: Set the path for the directory containing your PDF reports.
- **Output Directory**: Set the path for the directory where analysis reports will be saved.
- **Coding Keywords**: Customize the keywords used for analysis. The default keywords are:
  ```python
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
  ```

## Usage
To run the analysis, execute the following command:
```bash
python main.py
```
### Example Input
Place your PDF reports in the `data/` directory. The system will process these files and generate analysis reports.

## Model Training
To train the model:
- Uncomment the training section in `main.py`
- Ensure your PDF reports are prepared in the `data/` directory
- Run the command: `python main.py`

## Key Components
- `text_extractor.py`: Handles PDF text extraction
- `text_analyzer.py`: Performs multi-task text analysis
- `report_generator.py`: Generates automated reports based on analysis
- `trainer.py`: Manages the model training pipeline

## Environment Variables
- `HUGGINGFACE_TOKEN`: This token is required for accessing Hugging Face models. You can obtain it by creating an account on the Hugging Face website.

## Contributing
We welcome contributions to the Flight Accident Analysis System! Please submit issues and pull requests to help improve the project.

## License
This project is licensed under the MIT License.

## Documentation
For more detailed information about the Flight Accident Analysis System, please refer to the [documentation.md](documentation.md) file.
