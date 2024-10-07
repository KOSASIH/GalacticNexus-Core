# Import the necessary libraries
import tkinter as tk

# Define the wallet interface class
class WalletInterface:
  def __init__(self):
    self.window = tk.Tk()
    self.window.title("GNX Wallet")

    # Create wallet balance label
    self.wallet_balance_label = tk.Label(self.window, text="Wallet Balance: 0")
    self.wallet_balance_label.pack()

    # Create transaction history listbox
    self.transaction_history_listbox = tk.Listbox(self.window)
    self.transaction_history_listbox.pack()

  def update_wallet_balance(self, balance):
    self.wallet_balance_label.config(text=f"Wallet Balance: {balance}")

  def update_transaction_history(self, transactions):
    self.transaction_history_listbox.delete(0, tk.END)
    for transaction in transactions:
      self.transaction_history_listbox.insert(tk.END, transaction)

  def run(self):
    self.window.mainloop()

# Export the wallet interface class
def wallet_interface():
  return WalletInterface()
