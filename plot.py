import matplotlib.pyplot as plt
import numpy as np

time_to_respond = [09.239320,
 09.491535,
 12.463158,
 09.690260]
length_of_query = [24,
  45,
  45,
  45]

 

  


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
