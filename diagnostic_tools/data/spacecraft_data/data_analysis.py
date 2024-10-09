import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class DataAnalysis:
    def __init__(self, data):
        self.data = data

    def plot_temperature(self):
        # Plot temperature data
        temperature_data = self.data["temperature"]
        plt.plot(temperature_data)
        plt.xlabel("Time")
        plt.ylabel("Temperature (C)")
        plt.title("Temperature Data")
        plt.show()

    def plot_humidity(self):
        # Plot humidity data
        humidity_data = self.data["humidity"]
        plt.plot(humidity_data)
        plt.xlabel("Time")
        plt.ylabel("Humidity (%)")
        plt.title("Humidity Data")
        plt.show()

    def plot_pressure(self):
        # Plot pressure data
        pressure_data = self.data["pressure"]
        plt.plot(pressure_data)
        plt.xlabel("Time")
        plt.ylabel("Pressure (Pa)")
        plt.title("Pressure Data")
        plt.show()

# Example usage:
data = pd.read_csv("telemetry_data.csv ")
data_analysis = DataAnalysis(data)
data_analysis.plot_temperature()
data_analysis.plot_humidity()
data_analysis.plot_pressure()
