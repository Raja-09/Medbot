import matplotlib.pyplot as plt
import numpy as np

time_to_respond = [41, 27, 30, 56, 27, 35, 41, 33,39]
length_of_query = [3, 18, 17, 42, 19, 24, 30, 40,15]

# Create scatter plot
plt.figure(figsize=(8, 6))
plt.scatter(length_of_query, time_to_respond, color="blue", alpha=0.5)

# Add labels and title
plt.title("Time to Respond vs. Length of User Query")
plt.xlabel("Length of User Query")
plt.ylabel("Time to Respond (seconds)")

# Display grid
plt.grid(True)

# Show plot
plt.show()
