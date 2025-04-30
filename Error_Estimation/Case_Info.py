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
        
        
    # Default
    def Load_Equation_Info(self):
        Case_Name          = "HP_ID8"
        Governing_Equation = "NS_2D_SS"  # 2 Dimensional, Navier-Stokes, steady state
        Equation_Constants = [123., 123.]  # Density, viscosity
        GE_Out_Dev         = [[1., 1.], [0.3, 0.], [1., 0.]]	# [Multiplier, Deviation], u = multi * NN_Out + dev

        return [Case_Name, Governing_Equation, Equation_Constants, GE_Out_Dev]

    def Load_Domain_Size(self):
        Total_Domain = [[-5., 5.], [-0.5, 0.5]]  # x_min, x_max, ymin, ymax
        LB = np.float32(np.array(Total_Domain)[:, 0])  # Lower Bound
        UB = np.float32(np.array(Total_Domain)[:, 1])  # Upper Bound
        return [Total_Domain, LB, UB]

    # ----------------------------------------------------------------
    # NN
    # ----------------------------------------------------------------
    # Default
    def Load_NN_Size(self):
        Layers = 2
        Neurons = 16

        return [Neurons for x in range(Layers)]

    # ----------------------------------------------------------------
    # Training
    # ----------------------------------------------------------------
    # Default
    def Load_Backup_Info(self):
        # iter Based
        Costum_Backup = [1000, 3000, 5000, 7000, 10000, 10100, 12000, 15000, 20000, 50000, 100000]
        Fixed_Backup  = 5000
        Report_Interval = 100
        return [Report_Interval, Costum_Backup, Fixed_Backup]

    # Only for TL
    def Load_Optimizer_Info(self):
        Solver_Order = ["ADAM", "BFGS"]  # ADAM or BFGS
        Solver_Iter  = [10000, 300000]
        Adam_LR      = tf.keras.optimizers.schedules.PiecewiseConstantDecay([1000, 3000], [1e-2, 1e-3, 1e-4])

        return [Solver_Order, Solver_Iter, Adam_LR]

    # ----------------------------------------------------------------
    # Geo Points
    # ----------------------------------------------------------------
    def Load_Loss_Function(self):
        Scr_LF    = [[] for i in range(4)]
        Scr_LF[0] = ["BC_D" , 1.,   ["BC", 0, 1, 2], ["M", 0, 1]]
        Scr_LF[1] = ["BC_N" , 100., ["BC", 3],       ["D1", 0, 2]]
        Scr_LF[2] = ["NR_DA", 1.,   ["DA", 0],       ["M", 0, 1, 2]]
        Scr_LF[3] = ["NR_GE", 0.1,  ["BC", 0],       ["R", 0, 1, 2]]
        
        NR_B_List = np.flip(np.float32(np.array([0.001, 0.01, 0.1, 1., 10., 100., 1000.])))
        
        return Scr_LF, NR_B_List
  
        
    def Load_Point_Gen_Info(self):
        # Domain Params
        D_Size  = [[-5., 5.], [-0.5, 0.5]]
        L_Size  = [ [[-5.,-5.], [-0.5, 0.5]],	# Left
                    [[-5., 5.], [ 0.5, 0.5]],	# Top
                    [[-5., 5.], [-0.5,-0.5]],	# Bot
                    [[ 5., 5.], [-0.5, 0.5]]	# Right
                  ]
        
        # Points
        Scr_BP = [[] for i in range(4)]	# Boundary
        Scr_DP = [[] for i in range(2)] # Data
        Scr_CP = [[] for i in range(1)] # Collocation Point

        Scr_BP[0] = ["Gen", "Unif_Box", L_Size[0], [0, 100], "Para_Inlet"]
        Scr_BP[1] = ["Gen", "Unif_Box", L_Size[1], [500, 0], "Wall"]
        Scr_BP[2] = ["Gen", "Unif_Box", L_Size[2], [500, 0], "Wall"]
        Scr_BP[3] = ["Gen", "Unif_Box", L_Size[3], [0, 100], "Outflow"]
        
        File_List = ["HP_ID8.txt"]
        P_Dev     = -1. * np.average(np.loadtxt(File_List[0], unpack = True, usecols = (4)))
        
        Scr_DP[0] = ["Gen_File", File_List[0], [0, 1], [2, 3, 4]]	# Read x,y & u,v,p
        Scr_DP[1] = ["SP", "Data_Deviation", 0, [0., 0., P_Dev]]	# Normalize pressure

        Scr_CP[0] = ["Gen", "Unif_Box", D_Size, [150, 40]]
        
        return Scr_BP, Scr_DP, Scr_CP

	
class Report_Param_Class:
    def Load_Plot_Point_Info(self):
        Plot_Status = True      # if Dimension > 3, it has to use projection.
        Plot_Dim    = [2]       # Just 2
        Plot_Var_ID = [[0, 1]]  # Follows the input order in the GE
        Plot_Domain = [[[-5., 5.], [-0.5, 0.5]]]	# Domain size
        Fig_Size    = [[7, 3]]	# Figure size
        return Plot_Status, Plot_Dim, Plot_Var_ID, Plot_Domain, Fig_Size

    def Load_Image_Setting(self, Output_Filter):  # Contour
        # See the ID Numbers in Equation_Database.py
        Main_Var_ID = [0, 1, 2]
        Res_ID      = [0, 1, 2]
        
        # Domain & Resolution
        Domain_Size   = [[-5. , 5.], [-0.5 , 0.5]]
        Sample_Points = [300, 80]
        
        # MinMax Values in the plot, you can also just set it to [] to let it set automatically
        MainVar_MinMax  = [[0., 1.5], [-0.01, 0.01], [-0.6, 0.6]]
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
    # BC NOT USED
    def Calc_BC(self, X, Code):
        u, v, p = 0. * X[1], 0. * X[1], 0. * X[1]
        if (Code == "Wall"):
            u, v, p = 0. * X[1], 0. * X[1], 0. * X[1]
        elif (Code == "Inlet"):
            u = 0. * X[1] + 1.
        elif (Code == "Outflow"): # u --> du/dx, v --> dv/dx
            u, v, p = 0. * X[1], 0. * X[1], 0. * X[1]
        elif (Code == "Para_Inlet"):
            ys = X[1] * X[1] * 4.
            u = 1.5 * (1. - ys)
        return [u, v, p]

