# **Time-Series-Portfolio-Optimization**

## **Project Description**
A data science and machine learning solution for Guide Me in Finance (GMF) Investments. The project uses time series forecasting to predict asset price movements, which are then used to optimize an investment portfolio of high-growth, stable, and diversified assets. The final strategy is validated through backtesting.

## Table of Contents
- [Project Description](#project-description)
- [Business Understanding](#business-understanding)
- [Project Overview](#project-overview)
- [Key Features](#key-features)
- [Business Objectives](#business-objectives)
- [Project Structure](#project-structure)
- [Technologies Used](#technologies-used)
- [Setup and Installation](#setup-and-installation)
- [Usage](#usage)
- [Development and Evaluation](#development-and-evaluation)
- [Contributing](#contributing)
- [License](#license)

### **Business Understanding**

GMF Investments is a financial advisory firm that uses cutting-edge technology and data-driven insights for personalized portfolio management. The project's goal is to enhance portfolio performance by integrating advanced time series forecasting models to predict market trends, optimize asset allocation, and manage risks.

### **Project Overview**

The project follows a full data science workflow: data extraction, preprocessing, model development, portfolio optimization, and strategy backtesting. Historical financial data for Tesla (TSLA), the Vanguard Total Bond Market ETF (BND), and the S&P 500 ETF (SPY) are used. The project builds and compares forecasting models, including classical statistical models (ARIMA/SARIMA) and deep learning models (LSTM). The most effective model's forecasts are then used in a Modern Portfolio Theory (MPT) framework to construct an optimal portfolio, which is validated against a simple benchmark portfolio through backtesting.

### **Key Features**

- **Data Sourcing**: Automated fetching of historical financial data using the `yfinance` API.
- **Time Series Forecasting**: Implementation and comparison of classical statistical models (ARIMA/SARIMA) and deep learning models (LSTM) for price prediction.
- **Portfolio Optimization**: Application of Modern Portfolio Theory (MPT) to generate an Efficient Frontier and identify optimal portfolios.
- **Strategy Backtesting**: Simulation of the optimized portfolio's performance against a benchmark to validate the investment strategy.

### **Business Objectives**

The project addresses GMF Investments' need for data-driven insights to provide tailored investment strategies by:
- Predicting future market movements to inform portfolio adjustments.
- Optimizing asset allocation to enhance portfolio performance and manage risk.
- Using advanced technology to maintain a competitive edge.

## **Project Structure**
```
├── .vscode/                 # VSCode specific settings
├── .github/                 # GitHub specific configurations (e.g., Workflows)
│   └── workflows/
│       └── unittests.yml    # CI/CD workflow for tests and linting
├── .gitignore               # Specifies intentionally untracked files to ignore
├── requirements.txt         # Python dependencies
├── pyproject.toml           # Modern Python packaging configuration (PEP 517/621)
├── README.md                # Project overview, installation, usage
├── Makefile                 # Common development tasks (setup, test, lint, clean)
├── .env                     # Environment variables (e.g., API keys - kept out of Git)
├── src/                          # Core source code for the project
│   ├── data_ingestion.py         # Fetches and loads data from YFinance
|   ├── data_preprocessing.py     # Cleans and preprocesses financial data
│   ├── EDA/                      # Scripts for Exploratory Data Analysis (EDA)
│   │   ├── basic_data_inspection.py    # Performs initial checks on data types, size, and structure.
│   │   ├── missing_values_analysis.py  # Identifies and analyzes patterns of missing data.
│   │   ├── univariate_analysis.py      # Analyzes distributions of individual variables.
│   │   ├── bivariate_analysis.py       # Explores relationships between pairs of variables.
│   │   ├── multivariate_analysis.py    # Investigates relationships among multiple variables.
│   │   ├── temporal_analysis.py        # Analyzes time-based patterns and trends.
│   │   └── outlier_analysis.py         # Detects and investigates anomalous data points.
|   ├── feature_engineering.py    # Creates features like returns and volatility
│   ├── models/                   # Bayesian change point modeling
│   │   ├── arima_model.py        # ARIMA/SARIMA model training and forecasting
│   │   └── lstm_model.py         # LSTM deep learning forecasting
│   ├── portfolio_optimization.py # Implements MPT and Efficient Frontier generation
│   ├── backtesting.py            # Simulates and evaluates the portfolio strategy
│   └── main.py                   # Main script to run the entire workflow
│   └── utils/                    # Utility functions and helper classes
│       └── helpers.py            # General helper functions
├── tests/                   # Test suite (unit, integration)
│   ├── unit/                # Unit tests for individual components
│   └── integration/         # Integration tests for combined components
├── notebooks/               # Jupyter notebooks for experimentation, EDA, prototyping
    └── eda.ipynb            # Notebook for initial data exploration and visualizations
├── scripts/                 # Standalone utility scripts (e.g., data processing, deployment)
├── docs/                    # Project documentation (e.g., Sphinx docs)
├── data/                    # Data storage (raw, processed)
│   ├── raw/                 # Original, immutable raw data
│   └── processed/           # Transformed, cleaned, or feature-engineered data
├── config/                  # Configuration files
└── examples/                # Example usage of the project components
```


## **Technologies Used**

- **Python**: Core programming language.
- **Pandas, NumPy**: For data manipulation and numerical operations.
- **YFinance**: API for fetching financial data.
- **Statsmodels, pmdarima**: For ARIMA/SARIMA model implementation.
- **TensorFlow/Keras**: For LSTM deep learning models.
- **PyPortfolioOpt**: For Modern Portfolio Theory (MPT) implementation.
- **Matplotlib, Seaborn**: For data visualization.

## **Setup and Installation**

### **Prerequisites**

- Python 3.8+
- Git

### **Steps**

1. **Clone the repository:**
    
    ```
    git clone https://github.com/michaWorku/Time-Series-Portfolio-Optimization.git
    cd Time-Series-Portfolio-Optimization
    
    ```
    
    
2. **Create and activate a virtual environment:**
    
    ```
    python3 -m venv venv
    source venv/bin/activate  # On Windows: .\venv\Scripts\activate
    
    ```
    
3. **Install dependencies:**
    
    ```
    pip install -r requirements.txt
    
    ```
    

## **Usage**

- To run the full project workflow from data fetching to backtesting, execute the main script:


    ```
    python src/main.py
    
    ```
    
   This will:
      - Fetch historical data for TSLA, BND, and SPY.
      - Preprocess the data and perform EDA.
      - Train and evaluate time series forecasting models.
      - Forecast future TSLA prices.
      - Optimize the portfolio using MPT.
      - Backtest the strategy against a benchmark.
    

## **Development and Evaluation**

This section will detail the process of model selection, parameter tuning, and performance evaluation. It will include a discussion on the trade-offs between ARIMA and LSTM models and the justification for the final chosen portfolio, which may prioritize maximum risk-adjusted return or minimum volatility.
    - ARIMA/SARIMA – Parameter tuning via auto_arima.
    - LSTM – Architecture tuning (layers, neurons, epochs, batch size).
    - Evaluation Metrics – MAE, RMSE, MAPE for forecasts.
    - Portfolio Metrics – Expected return, volatility, Sharpe Ratio.
    - Backtesting – Cumulative returns comparison.

## **Contributing**

Guidelines for contributing to the project.

## **License**

This project is licensed under the [MIT License](https://www.google.com/search?q=LICENSE).
