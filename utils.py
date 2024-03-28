from smart_open import open
import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.legend_handler import HandlerPathCollection


# Load S3 bucket raw data in a google drive folder and generates a csv file.
# It also updates the the google drive folder and csv file with the most recently added data from s3 bucket.

def get_update_raw_data(boto_client,google_drive_folder,s3_bucket_name,s3_folder,csv_data_path, update = True):
  # Store new raw json data
  new_data = []

  if update:
  # Create the Google Drive folder if it doesn't exist
    if not os.path.exists(google_drive_folder):
        os.makedirs(google_drive_folder)

    # Function to recursively list all objects in a folder
    def list_objects(s3_client, bucket, prefix):
        paginator = s3_client.get_paginator('list_objects_v2')
        pages = paginator.paginate(Bucket=bucket, Prefix=prefix)
        for page in pages:
            for obj in page.get('Contents', []):
                yield obj['Key']

    # Recursively list objects in the specified folder
    objects = list_objects(boto_client, s3_bucket_name, s3_folder)

    for obj_key in objects:
        # Extract the file name from the object key
        file_name = os.path.basename(obj_key)

        root = obj_key.replace(file_name,'')
        # Specify the local file path on Google Drive

        drive_file_root = os.path.join(google_drive_folder, root)

        drive_file_path = os.path.join(google_drive_folder, obj_key)

        # Create the Google Drive folder if it doesn't exist
        if not os.path.exists(drive_file_root):
          os.makedirs(drive_file_root)

        if not os.path.exists(drive_file_path):
          # Download the file
          boto_client.download_file(s3_bucket_name, obj_key, drive_file_path)

          print(f"Downloaded: {obj_key} -> {drive_file_path}")

          # Check if the downloaded file is a JSON file
          if obj_key.endswith('.json'):
              # Read JSON data
              with open(drive_file_path, 'r') as json_file:
                  data = json.load(json_file)
                  new_data.append(data)

  
  # read data csv_file
  all_data = pd.read_csv(csv_data_path)

  if len(new_data)!=0:
    # Convert new_data in a DataFrame
    new_data = pd.DataFrame(new_data)

    all_data = pd.concat([all_data,new_data], axis = 0 , ignore_index=True)

    all_data.to_csv(csv_data_path, index = False)
  
  return all_data


def raw_data_to_csv(root_folder_path, csv_data_path):
  # Initialize an empty list to store data from JSON files
  data_list = []

  # Walk through the directory and read JSON files
  for folder_name, _, files in os.walk(root_folder_path):
      for file_name in files:
          # Check if the file is a JSON file
          if file_name.endswith(".json"):
              # Construct the full path to the JSON file
              json_file_path = os.path.join(folder_name, file_name)
              
              # Read the JSON file
              with open(json_file_path, 'r') as json_file:
                  # Assuming each JSON file contains a dictionary
                  patient_data = json.load(json_file)
                  
                  # Add additional information like patient identifier, etc.
                  # For example, you can add the patient identifier as a key-value pair
                  patient_data['patient_id'] = os.path.basename(folder_name)
                  
                  # Append the data to the list
                  data_list.append(patient_data)

  # Create a DataFrame from the list of dictionaries
  df = pd.DataFrame(data_list)

  df.to_csv(csv_data_path, index=False)

  # Print or manipulate the DataFrame as needed
  print(f"CSV file saved at: {csv_data_path}")

def plot_signal(signal, labels=None,figsize=(10,10), s=4):
    plt.figure(figsize=figsize)
    plt.plot(signal.index, signal, label='signal')
    if labels is not None:
        nonzero = signal.index[labels != 0]
        smin, smax = np.min(signal),  np.max(signal)
        lvl = smin - 0.05 * (smax-smin)
        plt.scatter(nonzero, np.ones(len(nonzero)) * lvl,
                s=s, color='tab:orange')
    plt.grid()
    plt.tight_layout()
    plt.show()

def plot_anomaly_scatter(X):

  def update_legend_marker_size(handle, orig):
    "Customize size of the legend marker"
    handle.update_from(orig)
    handle.set_sizes([20])

  X_scores = X['scores']
  
  plt.scatter(X.iloc[:, 0], X.iloc[:, 1], color="k", s=3.0, label="Data points")
  # plot circles with radius proportional to the outlier scores
  radius = (X_scores.max() - X_scores) / (X_scores.max() - X_scores.min())
  scatter = plt.scatter(
      X.iloc[:, 0],
      X.iloc[:, 1],
      s=1000 * radius,
      edgecolors="r",
      facecolors="none",
      label="Outlier scores",
  )
  plt.axis("tight")
  plt.xlim((-5, 5))
  plt.ylim((-5, 5))
  #plt.xlabel("prediction errors: %d" % (n_errors))
  plt.legend(
      handler_map={scatter: HandlerPathCollection(update_func=update_legend_marker_size)}
  )
  plt.title("Local Outlier Factor (LOF)")
  plt.show()

