# --------------------------------------------------------------
# Case Info: Contains all of the information
# --------------------------------------------------------------
# import tensorflow as tf
import numpy as np


class Case_Info_Class:
    # ----------------------------------------------------------------
    # Basic
    # ----------------------------------------------------------------
    def __init__(self):
        # Load Class (Default)
        self.Class_RP = Report_Param_Class()
        self.BC_EQ = BC_Equation_Class()

    def Load_Equation_Info(self):
        Case_Name = "HELLO_PINN_WORLD" 	   # Any Name is okay
        Governing_Equation = "Burgers_1D"  # See Equation_Database.py to see the equation in more detail
        Equation_Constants = [0.01]  	   # Alpha

        return [Case_Name, Governing_Equation, Equation_Constants]

    def Load_Domain_Size(self):
        Total_Domain = [[0., 1.], [-1., 1.]]  # x_min, x_max, ymin, ymax
        LB = np.float32(np.array(Total_Domain)[:, 0])  # Lower Bound
        UB = np.float32(np.array(Total_Domain)[:, 1])  # Upper Bound
        return [Total_Domain, LB, UB]

    # ----------------------------------------------------------------
    # NN
    # ----------------------------------------------------------------
    # Default
    def Load_NN_Size(self):
        Layers  = 1
        Neurons = 2

        return [Neurons for x in range(Layers)]

    # ----------------------------------------------------------------
    # Training
    # ----------------------------------------------------------------
    # Default
    def Load_Backup_Info(self):
        # iter Based
        Costum_Backup = [1000, 5000, 10000, 12000, 15000, 20000, 50000, 100000]
        Fixed_Backup = 5000
        Report_Interval = 100
        return [Report_Interval, Costum_Backup, Fixed_Backup]

    # ----------------------------------------------------------------
    # Geo Points
    # ----------------------------------------------------------------
    def Load_Loss_Function(self):
        #Scr_LF   = ["Name", Weight, Set_of_Points, OutVar]
        Scr_LF    = [[] for i in range(2)]
        Scr_LF[0] = ["BC_D", 1., ["BC", 0, 1, 2], ["M", 0]]
        Scr_LF[1] = ["GE"  , 1., ["C", 0]       , ["R", 0]]
        
        return Scr_LF
        

    def Load_Point_Gen_Info(self):
        # Domain Params
        D_Size  = [  [0., 1.]  , [-1. ,  1. ]]
        L_Size  = [ [[ 0.,  0.], [-1. ,  1.]],
                    [[ 0.,  1.], [-1.,  -1.]],
                    [[ 0.,  1.], [ 1.,   1.]]
                  ]
        
        #-------------------------------------------------------
        Scr_BP = [[] for i in range(3)]	# Boundary
        Scr_DP = [[] for i in range(0)] # Data
        Scr_CP = [[] for i in range(1)] # Collocation Point
        #-------------------------------------------------------
        
        Scr_BP[0] = ["Gen", "Unif_Box", L_Size[0], [0, 400], "Burger_Init"]
        Scr_BP[1] = ["Gen", "Unif_Box", L_Size[1], [400, 0], "Wall"]
        Scr_BP[2] = ["Gen", "Unif_Box", L_Size[2], [3, 0], "Wall"]
        
        Scr_CP[0] = ["Gen", "Unif_Box", D_Size,  [1, 2]]
        
        return Scr_BP, Scr_DP, Scr_CP#, Scr_UP

	
class Report_Param_Class:
    def Load_Plot_Point_Info(self):
        Plot_Domain = [[[0., 4.], [-1.1, 1.1]]]
        Img_Size    = [[7, 3]]
        
        return True, [2], [[0, 1]], Plot_Domain, Img_Size

    def Load_Image_Setting(self, Output_Filter):  # Contour
        # What variable to plot 
        # See the ID Numbers in Equation_Database.py
        Main_Var_ID = [0]
        Res_ID      = [0]
        
        # Domain & Resolution
        Domain_Size   = [[0., 1.], [-1., 1.]]
        Sample_Points = [100,      250]
        
        # MinMax Values in the plot, you can also just set it to [] to let it set automatically
        MainVar_MinMax  = [[-1., 1.]]
        Residual_MinMax = [[-0.1, 0.1]]
        
        # Figure Size
        Img_Size = [2.7, 4]

        if Output_Filter == "Folder":
            return [1, []]
        elif Output_Filter == "Report":
            return [1, [Main_Var_ID, Res_ID], [MainVar_MinMax, Residual_MinMax], 
                    [[[0, 1]], [Domain_Size[0]], [Domain_Size[1]], [[]]],
                    [Sample_Points], [Img_Size]]


class BC_Equation_Class():
    def Calc_BC(self, X, Code):
        # X[0] = t
        # X[1] = x
        if (Code == "Wall"):
            T = 0. * X[1]
        elif (Code == "Burger_Init"):
            T = np.sin(2. * np.pi * X[1]) 
        return [T]

