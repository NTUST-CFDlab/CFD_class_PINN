
# Import
import tensorflow as tf
import numpy as np

def Get_Eq_Class(Equation_Set):
    if Equation_Set == "Burgers_1D":
        return Burgers_1D()

def Set_Eq_Constants(GE, Constant_Names, Constant_Values):
    for i in range(len(Constant_Values)):
        if Constant_Names[i] == "alpha":
            GE.alpha = Constant_Values[i]


class Burgers_1D():
    def Equation_Info(self):
        Input_Names  = ["t", "x"]
        Output_Names = ["u"]
        D1_Names     = ["u_t", "u_x"]
        Residual_Names = ["Residual"]
        Constant_Names = ["alpha"]

        return [Input_Names, Output_Names, D1_Names, Residual_Names, Constant_Names]

    def Set_Model(self, NN_Model):
        self.model = NN_Model

    def Call_Calc_Until(self, Filter):
        Calc_Until = 0                  # 0 = Coor, 1 = Main Var, 2 = D1, 3 = D2
        for i in range(len(Filter)):
            if (Filter[i] == "M"):
                if Calc_Until < 1:
                    Calc_Until = 1
            elif (Filter[i] == "D1"):
                if Calc_Until < 2:
                    Calc_Until = 2
            else:
                Calc_Until = 3
        return Calc_Until

    def Get_Sim_Param(self, X_Array, Filter):
        Calc_Until = self.Call_Calc_Until(Filter)
        if Filter == "A":
            Filter = ["C", "M", "D1", "D2", "R"]

        # Calc Gradient
        t = X_Array[0] 
        x = X_Array[1]
        if Calc_Until > 0:
            with tf.GradientTape(persistent=True) as tape:
                # Coor
                tape.watch(t)
                tape.watch(x)

                # Main Var
                All_Var = self.model(tf.stack([t[:, 0], x[:, 0]], axis=1))
                u = All_Var[:, 0:1] #* self.Out_Dev[0][0] + self.Out_Dev[0][1]

                # First Order
                if Calc_Until > 1:
                    u_t = tape.gradient(u, t)
                    u_x = tape.gradient(u, x)

                # Second Order
                if Calc_Until > 2:
                    u_xx = tape.gradient(u_x, x)
            del tape

        # Export Result
        Results = []
        for i in range(len(Filter)):
            if (Filter[i] == "R"):
                Res= u_t + 2.*u*u_x - self.alpha * u_xx
                Results.append([Res])
            elif (Filter[i] == "C"):
                Coor = [t, x]
                Results.append(Coor)
            elif (Filter[i] == "M"):
                Main_Var = [u]
                Results.append(Main_Var)
            elif (Filter[i] == "D1"):
                Derivative1 = [u_t, u_x]
                Results.append(Derivative1)
            elif (Filter[i] == "D2"):
                Derivative2 = [u_xx]
                Results.append(Derivative2)

        return Results




