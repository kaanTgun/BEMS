import numpy as np
import pandas
import random
from datetime import datetime
import matplotlib.pyplot as plt

# Environment Constrains
max_episode_length = 200

class Enve():
	def __init__(self, DataFile_path, max_charge, min_charge, rate , battery_cap):

		self.df = pandas.read_csv(DataFile_path)        # data is in terms of Cents/kWh
		self.columnCount = self.df['Date'].count()

		self.max_charge = max_charge
		self.min_charge = min_charge
		self.rate = rate               # percent rate of charge and discharge in every time step
		self.battery_cap = battery_cap # kWh

	def reset(self):
		""" Reset the environment variables before experiencing new state space

		Returns:
			tuple: observations
		"""
		self.soc        = np.random.choice([0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])     # Percent state of charge 
		self.startIndex = random.randint(0, self.columnCount - max_episode_length -1)
		self.endIndex   = self.startIndex + random.randint(100, max_episode_length)
		self.index      = self.startIndex
		self.done       = False
		return self.step(2)
	
	def test(self, startIndex, endIndex,soc ):
		""" With the given start-end index and SOC text the Policy in the environment  

		Args:
			startIndex (int): start index of the time_span
			endIndex (int): end index of the time_span
			soc (float): precent State of Charge

		Returns:
			function: function call of the currrent observations
		"""
		self.soc        = soc     # Percent state of charge 
		self.startIndex = startIndex
		self.endIndex   = endIndex
		self.index      = self.startIndex
		self.done       = False
		return self.step(2)

	def step(self, action):
		""" Responce of the environment when the agent takes an action

		Args:
			action (int): 0=buy, 1=sell, 2=hold

		Returns:
			State(hour, month, price, SOC), reward, done
		"""
		rowData = self.df.iloc[self.index]
		power_price = float(rowData['HOEP'])/1000
		date_time   = datetime.strptime(rowData['Date'], '%Y-%m-%d %H:%M:%S')

		self.index += 1
		
		if (action==0 and self.max_charge >= self.soc + self.rate ):
			# Buy power
			self.soc += self.rate
			reward = -power_price * self.rate * self.battery_cap
		elif (action==1 and self.min_charge <= self.soc - self.rate ):
			# Sell power
			self.soc -= self.rate
			reward = power_price * self.rate * self.battery_cap
		else:
			reward = -0.001
		
		if self.index >= self.endIndex: self.done=True

		return (date_time.hour, date_time.month+ date_time.day *0.1, self.soc, power_price), reward, self.done