# **Time-Series-Portfolio-Optimization**

A data science and machine learning solution for Guide Me in Finance (GMF) Investments. This project uses time series forecasting to predict asset price movements, which are then used to optimize an investment portfolio and validate the strategy through backtesting.

## **Table of Contents**

1. [Project Description](https://www.google.com/search?q=%23project-description)
2. [Business Understanding](https://www.google.com/search?q=%23business-understanding)
3. [Project Overview](https://www.google.com/search?q=%23project-overview)
4. [Key Features](https://www.google.com/search?q=%23key-features)
5. [Business Objectives](https://www.google.com/search?q=%23business-objectives)
6. [Project Structure](https://www.google.com/search?q=%23project-structure)
7. [Technologies Used](https://www.google.com/search?q=%23technologies-used)
8. [Setup and Installation](https://www.google.com/search?q=%23setup-and-installation)
9. [Usage](https://www.google.com/search?q=%23usage)
10. [Development and Evaluation](https://www.google.com/search?q=%23development-and-evaluation)
11. [Contributing](https://www.google.com/search?q=%23contributing)
12. [License](https://www.google.com/search?q=%23license)

### **Project Description**

This project provides a comprehensive data science and machine learning solution for Guide Me in Finance (GMF) Investments. The core of the solution lies in leveraging time series forecasting models to predict future asset price movements. These predictions are then used as inputs for a portfolio optimization strategy, which aims to maximize returns for a given level of risk. The entire strategy is rigorously validated through backtesting against historical data.

### **Business Understanding**

GMF Investments is a financial advisory firm that uses data-driven insights for personalized portfolio management. The goal of this project is to enhance portfolio performance by integrating advanced time series forecasting models to predict market trends, optimize asset allocation, and manage risks more effectively. This allows GMF to provide clients with a more sophisticated and data-backed investment strategy.

### **Project Overview**

The project follows a full data science workflow:

1. **Data Extraction & Preprocessing**: Sourcing historical financial data for key assets.
2. **Model Development**: Building and training both statistical (ARIMA) and deep learning (LSTM) forecasting models.
3. **Portfolio Optimization**: Using model forecasts to create an optimal asset allocation strategy.
4. **Strategy Backtesting**: Evaluating the performance of the optimized portfolio against a benchmark.

### **Key Features**

- **Multi-model Forecasting**: Utilizes both traditional ARIMA and advanced LSTM neural networks for robust price predictions.
- **Automated Pipeline**: A modular and scalable architecture allows for easy extension and automation of the entire workflow.
- **Risk-Adjusted Optimization**: Implements Modern Portfolio Theory (MPT) to find the most efficient portfolio based on predicted returns and volatility.
- **Performance Evaluation**: Comprehensive evaluation of models and backtested strategies using standard metrics like MAE, RMSE, Sharpe Ratio, and cumulative returns.

### **Business Objectives**

- Develop a predictive model to forecast future asset prices.
- Create an optimized asset allocation strategy that maximizes the Sharpe Ratio.
- Validate the strategy's performance against a standard market index.
- Provide a clear, reproducible pipeline for future model updates and analysis.

### **Project Structure**

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
│   ├── data_fetch.py         # Fetches and loads data from YFinance
|   ├── data_preprocessing.py     # Cleans and preprocesses financial data
│   ├── EDA/                      # Scripts for Exploratory Data Analysis (EDA)
│   │   └── eda.py                # Visualize prices, analyze volatility, detect outliers, check for stationarity and calculate key risk metrics   
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
│       ├── __init__.py
│       ├── run_backtest.py  # Backtesting script to validate the portfolio strategy
│       ├── run_forecast.py  # Forecasting script to generate predictions
│       └── run_portfolio_opt.py # Portfolio optimization script to generate optimal asset allocation
├── docs/                    # Project documentation (e.g., Sphinx docs)
├── data/                    # Data storage (raw, processed)
│   ├── raw/                 # Original, immutable raw data
│   └── processed/           # Transformed, cleaned, or feature-engineered data
├── config/                  # Configuration files
└── examples/                # Example usage of the project components
```

### **Technologies Used**

- **Python 3.8+**
- **Data Science Stack**: `Pandas`, `Numpy`, `Scikit-learn`
- **Time Series**: `pmdarima`
- **Deep Learning**: `TensorFlow`, `Keras`
- **Visualization**: `Matplotlib`
- **Portfolio Optimization**: `PyPortfolioOpt`

### **Setup and Installation**

1. **Clone the repository:**
    
    ```
    git clone https://github.com/michaWorku/Time-Series-Portfolio-Optimization.git
    cd Time-Series-Portfolio-Optimization
    
    ```
    
2. **Create a virtual environment:**
    
    ```
    python3 -m venv venv
    source venv/bin/activate  # On Windows: .\venv\Scripts\activate
    
    ```
    
3. **Install dependencies:**
    
    ```
    pip install -r requirements.txt
    
    ```
    

### **Usage**

The main scripts for running the pipelines are located in the `src/scripts/` directory.

- **To train both models and evaluate their performance:**
    
    ```
    python src/scripts/run_train.py
    
    ```
    
- **To run a backtest on the trained models and export results:**
    
    ```
    python src/scripts/run_backtest.py
    
    ```
    
- **To generate a new forecast for future periods:**
    
    ```
    python src/scripts/run_forecast.py
    
    ```
    
- **To run the portfolio optimization and backtesting:**
    
    ```
    python src/scripts/run_portfolio_opt.py
    
    ```
    

### **Development and Evaluation**

This section will detail the process of model selection, parameter tuning, and performance evaluation. It will include a discussion on the trade-offs between ARIMA and LSTM models and the justification for the final chosen portfolio, which may prioritize maximum risk-adjusted return or minimum volatility.

- **ARIMA/SARIMA**: Parameter tuning via `auto_arima`.
- **LSTM**: Architecture tuning (layers, neurons, epochs, batch size).
- **Evaluation Metrics**: MAE, RMSE, MAPE for forecasts.
- **Portfolio Metrics**: Expected return, volatility, Sharpe Ratio.
- **Backtesting**: Cumulative returns comparison.

### **Contributing**

Guidelines for contributing to the project.

### **License**

This project is licensed under the [MIT License](https://www.google.com/search?q=LICENSE).
