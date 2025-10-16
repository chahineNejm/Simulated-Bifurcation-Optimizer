# Simulated biffucation

## Installation
Clone the repository:

`git clone https://gitlab-student.centralesupelec.fr/chahine.nejma/bifurcation-simule.git`

Install the required dependencies (preferably in a virtual environment):

`pip install -r requirements.txt`


## Usage
### Step 1: Reduce the instance of your problem to an Ising instance
Here there are two options:
- You already have those from a custom reduction of the discrete NP-hard problem of your choice
- You can generate those matrices using the scripts in `/problem_reductions`, witch provide reductions for:
    - The number partitioning problem
    - More to come...

### Step 2: Run the simulation and optain the approximated solution
To do this, you will need to provite your matrices, as well as a couple of other parameters to the `compute_single_instance` function inside of `computing.py`

Then, reading the states and energies arrays from the saved file of from the input of the function, you will use the `extract_full_solution` function from `result_annalysis.py` to read thr apprixmated spin configuration

### Step 3: Perform the reduction the other way around to retreive your apprximated solution
Using the same method you used to step 1, you can go from your spin configuration to the approximated solution of your specific NP-hard problem

## Support