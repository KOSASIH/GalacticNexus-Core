import tkinter as tk

class UserInterface:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Galactic Nexus Core")
        self.label = tk.Label(self.root, text="Welcome to Galactic Nexus Core")
        self.label.pack()
        self.button = tk.Button(self.root, text="Send Data to Blockchain", command=self.send_data_to_blockchain)
        self.button.pack()

    def send_data_to_blockchain(self):
        # Get data from user input
        data = self.get_user_input()
        # Send data to blockchain
        galactic_nexus_core = GalacticNexusCore()
        galactic_nexus_core.send_data_to_blockchain(data, "ethereum")

    def get_user_input(self):
        # Get user input from GUI
        pass

    def run(self):
        self.root.mainloop()
