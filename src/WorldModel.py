import numpy as np
import pandas
import random
from datetime import datetime
import math
import matplotlib.pyplot as plt


class Enve():
	def __init__(self, DataFile_path, max_charge, min_charge, rate , battery_cap, max_episode_len, min_episode_len=24, chrg_eff=1, dischrg_eff=1):

		self.df = pandas.read_csv(DataFile_path)        # data is in terms of Cents/kWh
		self.columnCount = self.df['Date'].count()

		self.max_charge = max_charge
		self.min_charge = min_charge
		self.rate = rate               # percent rate of charge and discharge in every time step
		self.battery_cap = battery_cap # kWh

		self.chrg_eff = chrg_eff
		self.dischrg_eff = dischrg_eff

		self.max_episode_len = max_episode_len
		self.min_episode_len = min_episode_len

	def reset(self):
		""" Reset the environment variables before experiencing new state space

		Returns:
			tuple: observations
		"""
		self.soc        = np.random.choice([0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])     # Percent state of charge 
		self.startIndex = random.randint(self.min_episode_len, self.columnCount - self.max_episode_len -1)
		self.episode_len = random.randint(self.min_episode_len, self.max_episode_len)
		self.endIndex   = self.startIndex + self.episode_len
		self.index      = self.startIndex
		self.done       = False
		self.chrg_ctr		= 0
		return self.step(2)

	def ema(self, n_interval):
		""" Exponential moving average calc.
		https://www.learndatasci.com/tutorials/python-finance-part-3-moving-average-trading-strategy/
		Args:
				n_interval (int): past N intervals
		"""
		data = self.df.iloc[self.index - n_interval : self.index]
		power_price = data['HOEP']/1000

		ema_short = power_price.ewm(span=n_interval, adjust=False).mean()
		return ema_short
	
	def cycle_decay(self):
		return math.exp(-0.03*int(self.chrg_ctr))

	def test(self, startIndex, endIndex, soc):
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

		reward = self.strategy_2(power_price, action)
				
		if self.index >= self.endIndex: self.done=True

		return (date_time.hour, date_time.month+date_time.day*0.1, self.soc, power_price), reward, self.done

	def strategy_1(self, power_price, action):
		if (action==0 and self.max_charge >= self.soc + self.rate ):
			# Buy power
			self.soc += self.rate
			coeff = -1
		elif (action==1 and self.min_charge <= self.soc - self.rate ):
			# Sell power
			self.soc -= self.rate
			coeff = 1
		else:
			coeff = -0.001
		
		reward = power_price * self.rate * self.battery_cap * coeff
		return reward

	def strategy_2(self, power_price, action):
		
		ema = self.ema(24).iloc[-1]
		if action==0 and self.max_charge >= self.soc + self.rate:
			# Buy power
			self.soc += self.rate
			coeff = -0.999 if (ema > power_price) else -0.5
			coeff /= self.chrg_eff

		elif action==1 and self.min_charge <= self.soc - self.rate:
			self.soc -= self.rate
			# Sell power
			coeff = 0.5 if ema > power_price else 0.999
			coeff *= self.dischrg_eff
		else:
			coeff = -0.001
		reward = power_price * self.rate * self.battery_cap * coeff
		return reward

	def strategy_3(self, power_price, action):
		ema = self.ema(24).iloc[-1]
		alpha = self.cycle_decay()

		if action==0 and self.max_charge >= self.soc + self.rate:
			# Buy power
			self.soc += self.rate
			self.chrg_ctr += self.rate

			coeff = -0.99 if ema > power_price else -0.5
			coeff /= self.chrg_eff

		elif action==1 and self.min_charge <= self.soc - self.rate:
			self.soc -= self.rate
			# Sell power
			coeff = 0.5 if ema > power_price else 0.99
			coeff *= self.chrg_eff
		else:
			coeff = -0.001
		reward = power_price * self.rate * self.battery_cap * coeff * alpha
		
		return reward

	def __repr__(self):
		return f'Environment: \nBattery Min SOC = {self.min_charge}\n \
														Battery Max SOC = {self.max_charge}\n \
														Battery Charge/Discaherge Rate = {self.rate}\n \
														Battery Capacity (kWh) = {self.battery_cap}\n \
														Battery Charge Efficiency = {self.chrg_eff}\n \
														Battery Disharge Efficiency = {self.dischrg_eff}\n\
														Episode Length = {self.min_episode_len} - {self.max_episode_len}\n'




