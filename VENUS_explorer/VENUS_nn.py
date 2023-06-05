"""
A VENUS model based on a pre-trained neural network. 
    Date: May 2023
    Author: Wenhan Sun
"""

import pickle
import pandas as pd
import os
import torch

class VENUS_NN():
    PASSWORD = "VERY_NAIVE_VENUS"
    def __init__(self, folder: str):
        """
        Constructor
        @params folder (str): Folder name containing the model
        @error ValueError: The file given by FOLDER does not contain the model. 
        """
        password_fname = os.path.join(folder, "password.txt")
        indicator = False
        if os.path.exists(password_fname):
            password_file = open(password_fname, "r")
            password = password_file.read()
            indicator = password == self.PASSWORD
        if not indicator:
            raise ValueError(f"{folder} is not the correct file!")
        
        accessory_fname = os.path.join(folder, "accessory")
        model_fname = os.path.join(folder, "model")
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.model = torch.load(model_fname, map_location=self.device)
        self.model.eval()
        
        d = pickle.load(open(accessory_fname, "rb"))
        self.standardizer = d["standardizer"]
        self.cols_dict = d["cols"]
        
    
    def get_cols(self, verbose: int = 0) -> dict[str, list]:
        """
        Get the input and output columns.
        @params verbose (int, default=0): Print the columns neatly in stdout if >0
        @return (dict[str, list]): A dictionary of two keys
            RETURN["input"] (list): The input column names
            RETURN["output"] (list): The output column names
        """
        if verbose > 0:
            for k,v in self.cols_dict.items():
                print(k)
                [print(f"\t{vv}") for vv in v]
        return self.cols_dict

    
    def predict(self, input_dict) -> pd.DataFrame:
        """
        Simulate VENUS.
        @params input_dict: The input dictionary. It could be in several forms. 
            It could also contain multiple data points, and the simulator computes all. 
            (pandas.DataFrame): The column names should be a superset of (* self.get_cols()["input"] *). 
                Otherwise error will be raised. Indices are ignored.
            (list[dict[KEY, float]]): A list of dictionary. Each dictionary is one data point. 
                The keys of each dictionary should be a superset of (* self.get_cols()["input"] *). 
            (dict[KEY, list[float]]): A dictionary whose keys should be a superset of (* self.get_cols()["input"] *). 
                For each key, the corresponding values should be the values of all data points.
        @return (pandas.DataFrame): The output dataframe. The columns are (* self.get_cols()["output"] *). 
            The order of the data points is retained. 
        
        The following inputs are the same:
        1): DataFrame
                a   b
            0   1.1 1.2
            1   1.3 1.4
            2   1.5 1.6
        2):
        [
            {"a": 1.1, "b": 1.2},
            {"a": 1.3, "b": 1.4},
            {"a": 1.5, "b": 1.6}
        ]
        3):
        {
            "a": [1.1, 1.3, 1.5],
            "b": [1.2, 1.4, 1.6]
        }
        
        For a single data point:
        1): DataFrame
                a   b
            0   1.1 1.2
        2):
        [
            {"a": 1.1, "b": 1.2}
        ]
        3):
        {
            "a": [1.1],
            "b": [1.2]
        }
        !!!! This is not accepted!
        {"a": 1.1, "b": 1.2} 
        # If the input is a dict, the values must be a list of floats. 
        # If you do not want to modify the dict, wrap a list around. 
        """
        df = pd.DataFrame(input_dict).reset_index()
        x = torch.tensor(df[self.cols_dict["input"]].to_numpy(), dtype=torch.float32, device=self.device)
        x = self.standardizer.transform_input(x)
        x = self.model(x)
        x = self.standardizer.inv_transform_output(x)
        pred = pd.DataFrame(x.detach().cpu().numpy(), columns=self.cols_dict["output"])
        return pred
    


    