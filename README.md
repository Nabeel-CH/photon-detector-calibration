# S1 Coursework

Repository for the S1 coursework. 
The solution for this project is in the Jupyter Notebook `notebooks/solution.ipynb`.

The assignment question is provided in `notebooks/instructions.ipynb`.

The majority of the code functionality is implemented in the `s1_sol` package:
- `s1_sol/s1funcs.py` contains functions for fitting, model definitions and utilities etc 
- `s1_sol/s1plots.py` contains functions for generating plots

The solution notebook produces:
- Figures saved in the `figs/`
- A `results.json` file containing parameter estimates from different methods

## Installation

1. Clone this GitLab repository to your local machine:
   ```bash
   git clone <repository-url>
   ```

3. Create and activate a virtual environment (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  
   ```

2. Install the package and its dependencies:
   ```bash
   pip install -e .
   ```
   
   This will install the `s1_sol` package.

4. If needed, create a Jupyter kernel for this environment:
   ```python
   python -m ipykernel install --user --name s1_coursework --display-name "S1 Coursework"
   ```

5. Open the solution notebook.

You should now be able to run the notebook. 

## Declaration of Use of Autogeneration Tools

Copilot was used in the following: 
 - Copilot auto suggestions were used for addng docstrings and comments to code as there was a lot of repeition 
 - My initial bootstrap loop was very slow, so I used Copilot’s suggestions to restructure it into a more efficient implementation
