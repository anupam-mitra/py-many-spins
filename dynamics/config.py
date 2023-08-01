import os

data_directory = os.path.join("..", "..", "pkl")
mps_directory = os.path.join(data_directory, "mps")
marginal_directory = os.path.join(data_directory, "marginals")
index_directory = os.path.join(data_directory, "index")

if not os.path.isdir(data_directory):
    os.makedirs(data_directory)

if not os.path.isdir(index_directory):
    os.makedirs(index_directory)

if not os.path.isdir(mps_directory):
    os.makedirs(mps_directory)

if not os.path.isdir(marginal_directory):
    os.makedirs(marginal_directory)
