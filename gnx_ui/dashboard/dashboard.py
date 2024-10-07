# Import the necessary libraries
import tkinter as tk

# Define the dashboard class
class Dashboard:
  def __init__(self):
    self.window = tk.Tk()
    self.window.title("GNX Dashboard")

    # Create network status label
    self.network_status_label = tk.Label(self.window, text="Network Status: Offline")
    self.network_status_label.pack()

    # Create node count label
    self.node_count_label = tk.Label(self.window, text="Node Count: 0")
    self.node_count_label.pack()

    # Create transaction count label
    self.transaction_count_label = tk.Label(self.window, text="Transaction Count: 0")
    self.transaction_count_label.pack()

  def update_network_status(self, status):
    self.network_status_label.config(text=f"Network Status: {status}")

  def update_node_count(self, count):
    self.node_count_label.config(text=f"Node Count: {count}")

  def update_transaction_count(self, count):
    self.transaction_count_label.config(text=f"Transaction Count: {count}")

  def run(self):
    self.window.mainloop()

# Export the dashboard class
def dashboard():
  return Dashboard()
