# gesture-recognition

### Environment
 - Python 3.7

### Virtual Environment
 - Create virtual environment
 ```sh
 $ virtualenv -p python venv
 ```
 - Activate virtual environment
 ```sh
 $ source ./venv/bin/activate
 ```
 - Install all dependencies
 ```sh
 $ pip install -r requirements.txt
 ```
 - Exit virtual environment
 ```sh
 $ deactivate
 ```

### Create Kernel for notebook.ipynb
```sh
$ pip install ipykernel
$ python -m ipykernel install --user --name=gesture
```

### Model Training
 - Open notbook.ipynb in with jupter notebook, and run all cells.  (Training data: *2019Proj2_train/* are required)
 ```sh
 $ jupyter notebook
 ```

### Prediction
 - Place test data to folder *Test_Set/* inside the project folder
 ```sh
 $ python main.py
 ```