# Battery Energy Management System 

The purpose of this project is to compare different BEMS scheduling systems in an unregulated energy market. 

## Background knowledge
Energy prices vary over time due to the change in demand, fuel price fluctuations or power plant availability. 
A battery energy management system can be used to buy power during the off-peak hours and sell power during peak hours, profiting off the difference while contributing to grid stability. This is also called the energy storage arbitrage problem in real-time markets. 

The price of power data is taken from an online database provided by Independent Electricity System Operator in Ontario Canada "https://www.ieso.ca/en/Power-Data/Data-Directory". 



## Models used for the BEMS strategy
-   📈  Linear programming
-   📊  Linear programming with added noise within an interval
-   🤖  Deep Q Network


### 📈  Linear programming
Linear programming is used to find the optimal strategy to make the highest profits possible. However, linear programming requires the knowledge of the price power in a given interval. 

### 📊 Linear programming with added noise within an interval
One way to utilise the Linear programming is by assuming we would be able to predict the price power data with a small amount of error for the next t intervals. This system runs a linear programming model for each time step and takes one single action. 

### 🤖  Deep Q Network
DQN is a type of reinforcement learning method combined with deep neural networks. An agent by only knowing its current state(state-of-charge, current-power-price, month, hour) in a given environment, is required to take an action(charge, hold, discharge) and with regards to its action, it will receive a reward and a new state. The agent tries to explore and exploit this given model-free environment and come up with the best strategy for this control problem. 
