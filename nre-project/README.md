# NRE Project

## Overview
The Nifty Risk Evaluation (NRE) system is designed to analyze and execute trading strategies based on options and futures data for the Nifty index. The system processes historical data, calculates optimal strike prices, and simulates trades to evaluate performance.

## Project Structure
```
nre-project
├── src
│   ├── nre.py          # Main logic of the NRE system
│   └── __init__.py     # Marks the directory as a Python package
├── parameter.csv       # Contains parameters for calculations and trading logic
├── requirements.txt     # Lists dependencies required for the project
└── README.md           # Documentation for the project
```

## Installation
To set up the NRE project, follow these steps:

1. Clone the repository:
   ```
   git clone <repository-url>
   cd nre-project
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage
To use the NRE system, ensure that you have the necessary data files in the correct format. The `parameter.csv` file should contain the required parameters for the trading strategy.

Run the main script:
```
python src/nre.py
```

## Parameters
The `parameter.csv` file should include the following columns:
- `start_date`: The starting date for data processing.
- `end_date`: The ending date for data processing.
- `start_time`: The starting time for trading.
- `end_time`: The ending time for trading.
- `method`: The trading method to be used.
- `sl`: The stop-loss value.
- `om`: The option multiplier.

## Contributing
Contributions are welcome! Please submit a pull request or open an issue for any enhancements or bug fixes.

## License
This project is licensed under the MIT License. See the LICENSE file for more details.