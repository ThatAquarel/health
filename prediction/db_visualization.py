import torch
import pandas as pd

db_x_infer = torch.load("./prediction/db_x_infer.pt")

db_x_infer_cases = pd.read_csv("./prediction/db_x_infer_cases.csv")
db_x_infer_cases = db_x_infer_cases[["Country Name", "Year"]]
