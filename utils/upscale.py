import os
import subprocess


def find_file(file_name, directory):
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.startswith(file_name):
                return os.path.splitext(file)[1]
    return None


def run_cmd(cmd: str):
    subprocess.Popen(cmd)