##########################################
# Version 1.1	Added the unifrom sampling function
##########################################

# Import
import numpy as np

def Ex_Points_Box_MD(Coor_List, Val_List, Domain_Params, Exclusion_Dir):
    Total_Dim = len(Domain_Params)
    #print(Domain_Params)
    #print(np.min(Coor_List, axis = 1))
    #print(np.max(Coor_List, axis = 1))

    Filtered_Coor = [[] for i in range(len(Coor_List))]
    Filtered_Data = [[] for i in range(len(Val_List))]
    for i in range(len(Coor_List[0])):
        In_Range = True
        for j in range(Total_Dim):
            if Coor_List[j][i] < Domain_Params[j][0]:
                In_Range = False
            elif Coor_List[j][i] > Domain_Params[j][1]:
                In_Range = False

        if not(In_Range) and Exclusion_Dir == "Exclude":
            for j in range(len(Coor_List)):
                Filtered_Coor[j].append(Coor_List[j][i])
            for j in range(len(Val_List)):
                Filtered_Data[j].append(Val_List[j][i])
        if In_Range and Exclusion_Dir == "Include":
            for j in range(len(Coor_List)):
                Filtered_Coor[j].append(Coor_List[j][i])
            for j in range(len(Val_List)):
                Filtered_Data[j].append(Val_List[j][i])

    return np.array(Filtered_Coor), np.array(Filtered_Data), len(Filtered_Coor[0])
    
