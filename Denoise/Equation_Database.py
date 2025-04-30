
# Import
import tensorflow as tf
import numpy as np


def Get_Eq_Class(Equation_Set):
    if Equation_Set == "NS_2D_SS":
        return NS_2D_SS()

def Set_Eq_Constants(GE, Constant_Names, Constant_Values):
    for i in range(len(Constant_Values)):
        if Constant_Names[i] == "rho":
            GE.rho = Constant_Values[i]
        elif Constant_Names[i] == "mew":
            GE.mew = Constant_Values[i]


class NS_2D_SS():
    def Equation_Info(self):
        Input_Names = ["x", "y"]
        Output_Names = ["u", "v", "p"]
        D1_Names = ["u_x", "u_y", "v_x", "v_y", "p_x", "p_y"]
        Residual_Names = ["Mass_Imbalance", "Mom-X_Imbalance", "Mom-Y_Imbalance"]
        Constant_Names = ["rho", "mew"]

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
        x = X_Array[0] 
        y = X_Array[1]
        if Calc_Until > 0:
            with tf.GradientTape(persistent=True) as tape:
                # Coor
                tape.watch(x)
                tape.watch(y)

                # Main Var
                All_Var = self.model(tf.stack([x[:, 0], y[:, 0]], axis=1))
                u = All_Var[:, 0:1] #* self.Out_Dev[0][0] + self.Out_Dev[0][1]
                v = All_Var[:, 1:2] #* self.Out_Dev[1][0] + self.Out_Dev[1][1]
                p = All_Var[:, 2:3] #* self.Out_Dev[2][0] + self.Out_Dev[2][1]

                # First Order
                if Calc_Until > 1:
                    u_x = tape.gradient(u, x)
                    u_y = tape.gradient(u, y)
                    v_x = tape.gradient(v, x)
                    v_y = tape.gradient(v, y)
                    p_x = tape.gradient(p, x)
                    p_y = tape.gradient(p, y)

                # Second Order
                if Calc_Until > 2:
                    u_xx = tape.gradient(u_x, x)
                    u_yy = tape.gradient(u_y, y)
                    v_xx = tape.gradient(v_x, x)
                    v_yy = tape.gradient(v_y, y)
            del tape

        # Export Result
        Results = []
        for i in range(len(Filter)):
            if (Filter[i] == "R"):
                nu = self.mew / self.rho

                R1 = self.Mass_Eq(u_x, v_y)
                R2 = self.Mom_x_Eq(u, v, u_x, u_y, p_x, u_xx, u_yy, self.rho, nu)
                R3 = self.Mom_y_Eq(u, v, v_x, v_y, p_y, v_xx, v_yy, self.rho, nu)

                Results.append([R1, R2, R3])
            elif (Filter[i] == "C"):
                Coor = [x, y]
                Results.append(Coor)
            elif (Filter[i] == "M"):
                Main_Var = [u, v, p]
                Results.append(Main_Var)
            elif (Filter[i] == "D1"):
                Derivative1 = [u_x, u_y, v_x, v_y, p_x, p_y]
                Results.append(Derivative1)
            elif (Filter[i] == "D2"):
                Derivative2 = [u_xx, u_yy, v_xx, v_yy]
                Results.append(Derivative2)

        return Results

    def Mass_Eq(self, u_x, v_y):
        return u_x + v_y

    def Mom_x_Eq(self, u, v, u_x, u_y, p_x, u_xx, u_yy, rho, nu):
        Convection_Terms = (u * u_x) + (v * u_y)
        Pressure_Terms = -p_x / rho
        Dissipation_Terms = nu * (u_xx + u_yy)
        return Convection_Terms - Pressure_Terms - Dissipation_Terms

    def Mom_y_Eq(self, u, v, v_x, v_y, p_y, v_xx, v_yy, rho, nu):
        Convection_Terms = u * v_x + v * v_y
        Pressure_Terms = -p_y / rho
        Dissipation_Terms = nu * (v_xx + v_yy)
        return Convection_Terms - Pressure_Terms - Dissipation_Terms
        

