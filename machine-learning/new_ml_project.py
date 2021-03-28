import os
import shutil
import argparse

ML_PROJ_DIR = 'ml-projects'
TEMPLATE_DIR = 'template'
CONFIG_NAME = 'config.yaml'

description = "Generate folder structure for a new machine learning project with pytorch"
parser = argparse.ArgumentParser(description=description, formatter_class=argparse.RawTextHelpFormatter)


parser.add_argument('ml_project_name', help="Folder name of the machine learning generation folder")

args = parser.parse_args()
ml_project_name = args.ml_project_name

dataset_project_path = os.path.join(ML_PROJ_DIR, ml_project_name)

shutil.copytree(TEMPLATE_DIR, dataset_project_path)

#config_yaml_path = os.path.join(dataset_project_path, CONFIG_NAME)
print(f"Created dataset project {ml_project_name} in {ML_PROJ_DIR}")
