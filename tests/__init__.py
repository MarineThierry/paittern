from os.path import join
from dotenv import load_dotenv
import os
# ...
env_path = join(dirname(dirname(__file__)),'.env') # ../.env
load_dotenv(dotenv_path=env_path)
