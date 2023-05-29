# Consumption-Savings Model with Labor Supply

This project implements two versions of the Consumption-Savings model, one with exogenous labor supply and another with endogenous labor supply. These models are an integral part of economic research, providing detailed insights into household behavior.

The consumption-saving model used here is inspired by the works of Imai (2004) and Druedahl (2017), with a focus on labor supply rather than human capital accumulation. The models are solved using the Endogenous Grid Point Method (EGM), considering aspects such as income uncertainty and taxation.

## Model Description

The Consumption-Savings model evaluates economic decisions by households. This model introduces labor supply into the consumption-saving decision-making framework. 

## Files in the repository

1. **egm.py:** Contains the algorithms using the Endogenous Grid Method (EGM) for solving the model.

2. **utility.py:** Contains the utility functions used in the models.

3. **ConSavModel.py:** Contains the definition of the Consumption-Savings model, including its parameters, settings, and allocation.

4. **ConSavModel.ipynb:** This is the main notebook which constructs and solves the Consumption-Savings model. It uses the classes and functions defined in the other files.

## How to run

1. Ensure you have Python 3.6 or later installed.

2. Install necessary libraries by running `pip install -r NumPy pandas Numba quantecon consav` in your terminal.

3. Run the ConSavModel.ipynb notebook using Jupyter Notebook.

## Authors

Elena Cristina Cero, Nicolai Christian Leth Bernsen and Mathis Kronbo Santesson.

## Acknowledgements

This project is inspired by the works of Imai (2004) and Druedahl (2017). The code is based on the Advanced Macroeconomics course: Heterogeneous Agent Models. Github: https://github.com/NumEconCopenhagen/AdvMacroHet.

We are grateful for the substantial contributions made by these researchers to the field of economic modeling.

## Support
For any questions or concerns, please open an issue on the GitHub repository or contact the maintainers directly.

## Authors and Contribution
We welcome contributions from others! If you'd like to contribute to this project, please see our Contribution Guide.

Please note that this project is released with a Contributor Code of Conduct. By participating in this project, you agree to abide by its terms.

## Disclaimer
This model is intended for educational and research purposes only.

