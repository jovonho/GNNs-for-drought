# GNNs for Drought - 599 Project


## Setup (conda)
- Create the conda environment. This automatically downloads dependencies.
    ```
    conda env create -f environment.yml
    conda activate drought-gnn
    ```
    (Must update environment.yml manually)

- Run models 
    ```
    python ./run_gcn.py
    python ./benchmark_mlp.py
    ```

## Setup (venv)
- Create a virtual environment and activate it
    ```
    python -m venv .venv
    .venv/Scripts/activate
    ```
- Install requirements
    ```
    pip install -r requirements.txt
    ```
    (Run `pip freeze > requirements.txt` to update the requirements file)
- Run models 
    ```
    python ./run_gcn.py
    python ./benchmark_mlp.py
    ```


