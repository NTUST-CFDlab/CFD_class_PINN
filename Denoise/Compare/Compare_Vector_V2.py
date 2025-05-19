################################################
# This is a version 2 of the code, 
# it should work with the current version of the save file
#
# Note:	The N_Output and Domain size is hardcoded 
#       (only applies for this BFS problem)
################################################

import numpy as np
import matplotlib.pyplot as plt
from Lib_Compare_Vector import Ex_Points_Box_MD
import tensorflow as tf



# Parameters
Main_File    = "BFS16.txt"
Noisy_File   = "BFS16_N03.txt"
PINN_File    = "XXX.keras"		# You need to change this
NN_Layers    = 3			# You need to change this
NN_Neurons   = 16			# You need to change this
Focus_Domain = [[3., 12.], [0., 2.]]


# NN Structure
class PINN_NeuralNet(tf.keras.Model):
    def __init__(self, All_Neurons = [64, 64, 64, 64],
                 activation='tanh',
                 kernel_initializer='glorot_normal',
                 **kwargs):
        super().__init__(**kwargs)
        
        # Hard Coded
        N_Output = 3
        Domain   = [[3., 0.], [12., 2.]]	# x min, y min, x max, y max

        self.NN_Layers = len(All_Neurons)
        self.lb = tf.constant(Domain[0])
        self.ub = tf.constant(Domain[1])

        # Define NN architecture
        self.scale = tf.keras.layers.Lambda(
            lambda x: 2.0 * (x - self.lb) / (self.ub - self.lb) - 1.0)
        self.hidden = [tf.keras.layers.Dense(All_Neurons[_],
                                             activation=tf.keras.activations.get(activation),
                                             kernel_initializer=kernel_initializer)
                       for _ in range(self.NN_Layers)]
        self.out = tf.keras.layers.Dense(N_Output)

    def call(self, X):
        Z = self.scale(X)
        for i in range(self.NN_Layers):
            Z = self.hidden[i](Z)
        return self.out(Z)


# Read Data
All_Layes   = [NN_Neurons for i in range(NN_Layers)]
Main_Data   = np.loadtxt(Main_File , unpack = True, usecols = (0,1,2,3,4))
Noise_Data  = np.loadtxt(Noisy_File, unpack = True, usecols = (0,1,2,3,4))
PINN_Model  = tf.keras.models.load_model(PINN_File, custom_obejcts={'PINN_NeuralNet': PINN_NeuralNet(All_Layes)})

def Extract_Main_Vals(Main_Data, Focus_Domain):
    F1_Coor, F1_Data, _ = Ex_Points_Box_MD(Main_Data[0:2], Main_Data[2:], Focus_Domain, "Include")
    All_Data = np.concatenate((F1_Coor, F1_Data), axis=0)
    return All_Data
    
def Extract_NN_Vals(PINN_Model, Coor):
    x = tf.convert_to_tensor(Coor[0])
    y = tf.convert_to_tensor(Coor[1])

    PINN_Data_All = PINN_Model(tf.stack([x, y], axis=1))
    u = PINN_Data_All[:, 0] #* Dev[0][0] + Dev[0][1]
    v = PINN_Data_All[:, 1] #* Dev[1][0] + Dev[1][1]
    p = PINN_Data_All[:, 2] #* Dev[2][0] + Dev[2][1]
    All_Data = np.array([u, v, p])
    
    return All_Data

# Process
Main_Data   = Extract_Main_Vals(Main_Data, Focus_Domain)
PINN_CMain  = Extract_NN_Vals(PINN_Model, Main_Data[0:2])
PINN_CNoise = Extract_NN_Vals(PINN_Model, Noise_Data[0:2])


# Plot
fig = plt.figure(figsize=(6, 3))
plt.quiver(Noise_Data[0], Noise_Data[1], Noise_Data[2] , Noise_Data[3],  color = 'red', label = "Noisy Data")
plt.quiver(Noise_Data[0], Noise_Data[1], PINN_CNoise[0], PINN_CNoise[1], color = 'blue', label = "PINN")
plt.legend(loc = 'lower left')
plt.savefig("Noise_PINN.png", bbox_inches='tight', dpi=300)
plt.close()


fig = plt.figure(figsize=(6, 3))
plt.quiver(Main_Data[0], Main_Data[1], Main_Data[2] , Main_Data[3],  color = 'red', label = "Real")
plt.quiver(Main_Data[0], Main_Data[1], PINN_CMain[0], PINN_CMain[1], color = 'blue', label = "PINN")
plt.legend(loc = 'lower left')
plt.savefig("Real_PINN.png", bbox_inches='tight', dpi=300)
plt.close()
