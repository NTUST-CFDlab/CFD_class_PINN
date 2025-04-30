# --------------------------------------------------------------
# Case Info: Contains all of the information
# --------------------------------------------------------------
import tensorflow as tf
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
        Case_Name = "BFS" 
        Governing_Equation = "NS_2D_SS"  # 2 Dimensional, Navier-Stokes, steady state
        Equation_Constants = [176., 432.]  # Density, viscosity

        return [Case_Name, Governing_Equation, Equation_Constants]

    def Load_Domain_Size(self):
        Total_Domain = [[3., 3.], [0., 0.]]  # x_min, x_max, ymin, ymax
        LB = np.float32(np.array(Total_Domain)[:, 0])  # Lower Bound
        UB = np.float32(np.array(Total_Domain)[:, 1])  # Upper Bound
        return [Total_Domain, LB, UB]

    # ----------------------------------------------------------------
    # NN
    # ----------------------------------------------------------------
    # Default
    def Load_NN_Size(self):
        Layers = 3
        Neurons = 32

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
        
        Scr_LF    = [[] for i in range(3)]
        Scr_LF[0] = ["BC_D" , 1., ["BC", 0, 1, 2, 3], ["M", 0, 1]]
        Scr_LF[1] = ["NR_DA", 1., ["DA", 0]         , ["M", 0, 1]]
        Scr_LF[2] = ["NR_GE", 1., ["C",  0, 1]      , ["R", 0, 1, 2]]
        
        NR_B_List = np.flip(np.float32(np.array([0.05, 0.1, 0.2, 0.3, 0.5, 1., 3., 5.])))
        
        return Scr_LF, NR_B_List
        
        
    def Load_Point_Gen_Info(self):
        # Domain Paramss
        L_Size  = [ 
                    [[  5.,  5.], [ 0. ,  1. ]],	# BFS
                    [[  3., 12.], [ 2. ,  2. ]],	# Top Wall 1
                    [[  5., 12.], [ 0. ,  0. ]],	# Bot wall 3
                    [[  3., 5. ], [ 1. ,  1. ]],	# Bot wall 2
                  ]
        
        
        
        Scr_BP = [[] for i in range(4)]	# Boundary
        Scr_DP = [[] for i in range(1)] # Data
        Scr_CP = [[] for i in range(2)] # Collocation Point
        Scr_UP = [[] for i in range(0)] # Unsteady

        Scr_BP[0] = ["Gen", "Unif_Box", L_Size[0], [0, 100 ], "Wall"]
        Scr_BP[1] = ["Gen", "Unif_Box", L_Size[1], [1800, 0], "Wall"]
        Scr_BP[2] = ["Gen", "Unif_Box", L_Size[2], [1400, 0], "Wall"]
        Scr_BP[3] = ["Gen", "Unif_Box", L_Size[3], [400, 0], "Wall"]
        
        File_List = ["BFS16_N03.txt"]
        
        Scr_DP[0] = ["Gen_File", File_List[0], [0, 1], [2, 3]]

        Scr_CP[0] = ["Gen", "Unif_Box", [[3.,  5.], [1., 2.]], [40, 25]]
        Scr_CP[1] = ["Gen", "Unif_Box", [[5., 12.], [0., 2.]], [140, 50]]
        
        return Scr_BP, Scr_DP, Scr_CP

	
class Report_Param_Class:
    def Load_Plot_Point_Info(self):
        Plot_Status = True  	# if Dimension > 3, it has to use projection.
        Plot_Dim = [2]
        Plot_Var_ID = [[0, 1]]  # Follows the input order in the GE
        Plot_Domain = [[[3., 12.], [0., 2.]]]
        Fig_Size = [[7, 3]]
        return Plot_Status, Plot_Dim, Plot_Var_ID, Plot_Domain, Fig_Size

    def Load_Image_Setting(self, Output_Filter):  # Contour
        # See the ID Numbers in Equation_Database.py
        Main_Var_ID = [0, 1, 2]
        Res_ID      = [0, 1, 2]
        
        # Domain & Resolution
        Domain_Size   = [[3., 12.], [0., 2.]]
        Sample_Points = [400, 150]
        
        # MinMax Values in the plot, you can also just set it to [] to let it set automatically
        MainVar_MinMax  = [[-0.2, 1.5], [-0.15, 0.15], []]
        Residual_MinMax = [[-0.01, 0.01], [-0.01, 0.01], [-0.01, 0.01]]
        
        # Figure Size
        Img_Size = [7, 3]

        if Output_Filter == "Folder":
            return [1, []]
        elif Output_Filter == "Report":
            return [1, [Main_Var_ID, Res_ID], [MainVar_MinMax, Residual_MinMax], 
                    [[[0, 1]], [Domain_Size[0]], [Domain_Size[1]], [[]]],
                    [Sample_Points], [Img_Size]]


class BC_Equation_Class():
    def Calc_BC(self, X, Code):
        u, v, p = X[0] * 0., X[0] * 0., X[0] * 0. # Default values
        if (Code == "Wall"):
            u, v, w, p = 0. * X[1], 0. * X[1], 0. * X[1], 0. * X[1]
        return [u, v, p]

