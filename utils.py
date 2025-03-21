import numpy as np

# Load CSV file (adjust 'filename.csv' to your file's name/path)
# Assumes the CSV file has no header and columns are separated by commas.
data = np.loadtxt("news/train_click_log.csv", delimiter=",", dtype=int, skiprows=1)

# Sort the data:
# We use np.lexsort with a tuple (click_timestamp, user_id).
# np.lexsort sorts by the last key first, so this sorts by user_id and then by click_timestamp within each user_id.
sorted_indices = np.lexsort((data[:, 2], data[:, 0]))
sorted_data = data[sorted_indices]

# Extract only the first two columns: user_id and click_article_id
result = sorted_data[:, :2]

# Save the result as a .npy file
np.save("news/train_click_log.npy", result)

