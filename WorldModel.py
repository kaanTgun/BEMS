import numpy as np
import pandas
import random
from datetime import datetime
from scipy.optimize import linprog
import matplotlib.pyplot as plt

# Environment Constrains
max_episode_length = 60

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
        self.endIndex   = self.startIndex + random.randint(10, max_episode_length)
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
        """ This is the responce of the environment when the agent takes an action

        Args:
            action (int): 0=buy, 1=sell, 2=hold

        Returns:
            State(hour, month, price, SOC), reward, done
        """
        rowData = self.df.iloc[self.index]
        power_price = float(rowData['HOEP'])/1000
        date_time   = datetime.strptime(rowData['Date'], '%Y-%m-%d %H:%M:%S')

        self.index += 1
        
        if (action==0 and max_charge >= self.soc + self.rate ):
            # Buy power
            self.soc += self.rate
            reward = -power_price * self.rate * self.battery_cap
        elif (action==1 and min_charge <= self.soc - self.rate ):
            # Sell power
            self.soc -= self.rate
            reward = power_price * self.rate * self.battery_cap
        else:
            reward = -0.001
        
        if self.index >= self.endIndex: self.done=True

        return (date_time.hour, date_time.month+ date_time.day *0.1, self.soc, power_price), reward, self.done

    def Linprog_True(self, soc, startIndex, endIndex):
        """Solve for the optimal strategy using linear programming. 
        Maximise for the profits, given the power price over a period of time.

        Args:
            soc (float): Percent State of Charge
            startIndex (int): Index of starting data row contains the date and price of power
            endIndex (int): Index of ending data row contains the date and price of power

        Returns:
            obj_func (float List): price of power over time
            actions (int List): chage/hold/discharge over time
            soc_time (int List): percent soc over time

        """
        self.startIndex = startIndex
        self.endIndex   = endIndex
        N = self.endIndex - self.startIndex

        A = np.append(np.tril(np.ones((N,N))), \
                     -np.tril(np.ones((N,N))), 0)
        b = np.append((self.max_charge-soc)*np.ones((N,1)), \
                      (soc-self.min_charge)*np.ones((N,1)))
        
        lb = -self.rate
        ub = self.rate

        rowData = self.df.iloc[self.startIndex:self.endIndex]
        power_price = rowData['HOEP']/1000

        obj_func = np.array(power_price)

        res = linprog(obj_func, A_ub=A, b_ub=b, bounds =[lb,ub])
        actions = np.asarray(res.x)


        soc_time = np.matmul(np.transpose(np.expand_dims(actions, axis=1)), np.triu(np.ones((N,N))))

        soc_time[0,:] += soc
        soc_time = np.sum(soc_time, axis=0)

        return obj_func, actions, soc_time
    
    def Linprog_predict_interval(self, soc, startIndex, endIndex, horizon=24, step=1, noise_magnitude=0.03):
        self.soc        = soc     # Percent state of charge 
        self.startIndex = startIndex
        self.endIndex   = endIndex
        N = self.endIndex - self.startIndex 

        A = np.append(np.tril(np.ones((horizon,horizon))), \
                     -np.tril(np.ones((horizon,horizon))), 0)
        b = np.append((self.max_charge-self.soc)*np.ones((horizon,1)), \
                      (self.soc-self.min_charge)*np.ones((horizon,1)))
        
        lb = -self.rate
        ub = self.rate

        rowData = self.df.iloc[self.startIndex:self.endIndex]
        power_price = rowData['HOEP']/1000
        power_price = np.array(power_price)
        
        soc_overtime = [self.soc]
        actions_taken = [0]
        price_overtime = [0]

        for i in range(0, N-horizon+1, step):
            noise = np.random.rand(horizon) * noise_magnitude            
            price_with_noise = noise + power_price[i : i+horizon]

            obj_func = np.array(price_with_noise)
            b = np.append((self.max_charge-self.soc)*np.ones((horizon,1)), \
                      (self.soc-self.min_charge)*np.ones((horizon,1)))
            

            res = linprog(obj_func, A_ub=A, b_ub=b, bounds=[lb,ub])
            action_t = round(100*res.x[0])/100    # Take the first action computed
            self.soc += action_t

            actions_taken.append(action_t)
            soc_overtime.append(float(self.soc))
            price_overtime.append(obj_func[0])

        # actions_taken.append(res.x[1:])         # Store predicted Actions
        # price_overtime.append(obj_func[1:])      # Store Predicted (Noisy) Price 

        return np.asarray(price_overtime), np.asarray(power_price), np.asarray(actions_taken), np.asarray(soc_overtime)

Enve = Enve(max_charge = 0.8,min_charge = 0.2,rate = 0.1, battery_cap = 1500)
power_price, actions_taken, soc_overtime = Enve.Linprog_True(0.6, 1000,1030)
t = [i for i in range(len(power_price))]

plt.plot(t, soc_overtime, 'b--')
plt.plot(t, power_price, 'r--')
plt.show

price_overtime, power_price, actions_taken, soc_overtime = Enve.Linprog_predict_interval(0.6,1000,1030+24)
t = [i for i in range(len(soc_overtime))]
plt.plot(t, soc_overtime, 'g--')
plt.plot(t, price_overtime, 'p--')
plt.show