"""
Move data from local to server. 
    Date: May 2023
    Author: Wenhan Sun
"""

import os
import shutil

def move_py_files_to_ssh(dest_ssh: str, dest_folder: str) -> None:
    """
    Move all the python files from local to the server. Note that the folder
        `__temp_py_dir` will be created and then deleted in the current 
        working directory. 
    @params dest_ssh (str): The server account. 
    @params dest_folder (str): The folder which the files are moved to. 
    """
    files = [f for f in os.listdir() if f[-3:] == ".py"]
    temp_dir = "__temp_py_dir"
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
    os.makedirs(temp_dir)
    [shutil.copy(f, os.path.join(temp_dir, f)) for f in files]
    origin = os.path.join(temp_dir, "*")
    dest = f"{dest_ssh}:{dest_folder}"
    os.system(f"scp {origin} {dest}")
    shutil.rmtree(temp_dir)

if __name__ == "__main__":
    move_py_files_to_ssh("wenhan@foundationg.dhcp.lbl.gov", "VENUS_explorer")
