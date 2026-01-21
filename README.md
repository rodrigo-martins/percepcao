# Perception Analysis

A Python project for exploratory analysis of survey data, generating visualizations and statistical graphs from CSV data.

## 📋 Dependencies

- **pandas**: Data manipulation
- **matplotlib**: Basic visualizations
- **seaborn**: Statistical graphs
- **numpy**: Numerical operations
- **scipy**: Advanced statistical functions
- **scikit-learn**: Machine learning and analysis
- **geopandas**: Geographical analysis
- **geobr**: Brazilian geographical data
- **factor_analyzer**: Exploratory Factor Analysis
- **pingouin**: Statistical tests
- **python-dotenv**: Environment variables management

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- pip (Python package manager)

### 1. Clone the Repository

```bash
git clone <seu-repositorio>
cd percepcao
```

### 2. Create Virtual Environment

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

## 📊 About the Project

This project processes survey data and generates various types of graphs and analyses, including:

- **Demographic analyses**: age, gender, professional level, education
- **Geographical analyses**: map of respondents by state
- **Thematic analyses**: surveys by area of expertise, experience, professional level
- **Likert scales**: visualizations of scale responses
- **Exploratory Factor Analyses (EFA)**: studies of factor structure
- **Significance analyses**: statistical tests and correlations

## 🏃 Running the Analysis

Each analysis script generates specific outputs. Execute any script from the `questions/` or `analises/` directories to generate results:

```bash
python ./questions/correlacao.py          # Outputs: correlation analysis graphs
python ./questions/likert.py              # Outputs: Likert scale visualization (likert.csv, likert.jasp)
python ./questions/idade.py               # Outputs: age-related analysis graphs
python ./questions/genero.py              # Outputs: gender analysis graphs
python ./questions/mapa_estados.py        # Outputs: geographical map of respondents
python ./analises/nivel_vs_experiencia.py # Outputs: level vs experience comparison graphs
python ./analises/significancia.py        # Outputs: statistical significance test results
```

## 📁 Project Structure

```
percepcao/
├── requirements.txt        # Project dependencies
├── data/                   # Input data
│   ├── raw.csv            # Raw data (input)
│   ├── tratado.csv        # Processed data
│   └── ordered.csv        # Ordered data
├── output/                # Generated graphs and results
│   ├── likert.csv         # Likert response table
│   ├── likert.jasp        # File for JASP (statistics)
│   └── [other graphs].png
├── questions/             # Main analysis module
│   ├── likert.py          # Likert scale
│   ├── idade.py           # Analysis by age
│   ├── genero.py          # Analysis by gender
│   ├── mapa_estados.py    # Geographical map
│   └── [other analyzers]
├── analises/              # Specific thematic analyses
│   ├── nivel_vs_experiencia.py
│   ├── area/              # Analyses by area
│   ├── estado/            # Analyses by state
│   └── [other themes]
└── assets/                # Auxiliary resources
    └── abreviacoes.txt    # Abbreviations used
```
