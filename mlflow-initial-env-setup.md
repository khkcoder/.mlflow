### **MLFLOW PROJECT SETUP FROM SCRATCH**:

---


2. create your first venv environment 

    ```bash
    python3 -m venv myfirstproject
    ```
3. activate your first environment

    ```bash
    source myfirstproject/bin/activate
    ```
4. install mlflow library in environment

    ```bash
    pip install mlflow pandas numpy scikit-learn tensorflow optuna hyperopt requests
    ```

7. Starting the mlflow tracking server

    ```bash
    mlflow server --host 127.0.0.1 --port 5000
    ```
8. Deactivate virtual environment 

    ```bash
    deactivate
    ```
9. Open another terminal in same folder and run this command

    ```bash
    source myfirstproject/bin/activate
    ```
10. Run this command

    ```bash
    export MLFLOW_TRACKING_URI="http://127.0.0.1:5000"
    ```
