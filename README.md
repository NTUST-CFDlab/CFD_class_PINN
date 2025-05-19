# CFD_class_PINN

This is a PINN code to demonstrate PINN capabilities. There are 3 cases provided: Burgers equation, eliminate noise in BFS flow, estimating error of a Hagen-Poussile flow. Please see the .pdf file to learn how to setup the code and how to run it.
 


# FAQ - BURGERS
   
Q:	How can I extract the data so I can compare it to CFD?

A:	Short answer:	No need, as long it roughly looks the same (the trends and values), then it is okay.

A:	Technical answer: 
        You need to pass on the coordinates of the final time at the network. 
	This may require its own code to make the process easy.
	

--- 
Q:	How large should I set my network?

A:	Up to you, in general, larger network is more powerful but harder to train.
	Smaller network is easier and faster to train, but may have limited capabilities.
	
 *Just for a reference, a deep neural network typically has 3 layers.


--- 	
Q:	How many points should I use?

A:	Boundary points is usually not that significant (just dont make it extremely low).
	Collocation points are like mesh, so more is better, but longer to simulate.
 
*For me, I prefer to have my cases (any case) to have around 10k-20k points.
	  This way, it is not to heavy and can still be quite accurate.

--- 	
Q:	Should I change the weight?

A:	Not recommended unless you know what you are expecting.


# FAQ - Denoise
Q:	The default code give me loss = nan (not a number)

A:	Maybe you change something to make it physically doesnt make sense, or
	some part of the network doesnt make sense.
	
	Some examples of things that doesnt make sense:
	(Line 20) Equation_Constants = [-1., -2.]	(Density and viscosity, should not be negative)
	(Line 25) Total_Domain = [[0., -1.]]		(The second number must be higher)
	(Line 25) Total_Domain = [[0., 0.]]		(The second number must be higher. not equal)
	(Line 35) Layers = 0				(You cannot use 0)
	(Line 58) Scr_LF[0] = ["B" , -1., [], []]	(Weight cannot be negative)
	(Line 62) NR_B_List = [-1, 0, 1]		(Beta values cannot be negative)
	
---
Q:	How to plot the vector?

A:	(I tried to explain it as clear as possible, so its a bit long)

	1. Go to the "Compare" folder and open "Compare_Vector_V2.py"
	2. You need to change 'PINN_File = "XXX.keras"' and the network size (based on your trained network)
	3. You need to set this with the network you already trained 
	   (the file containing the weight and bias of the network)
	4. This file is located in the "NN" folder, if you already trained the network.
	5. So, after the training, inside the result folder (whatever you named it with),
	   you can see 3 folders: "Data", "NN", "Residual"
	6. Inside the "NN" folder you will see a lot of files with ".keras" extension.
	7. This is the network weight and bias files at different conditions.
	   So, the network at:
	   	iter = 10000 		is "Backup-iter-10000.keras"
	   	Beta = 0.1 		is "Backup-Beta-B0.1.keras"
	   	Total_Loss = 1e-3	is "Backup-LS-0.001.keras"
	8. Copy the condition/file (.keras) you want to plot to the "Compare" folder and rename
	   the 'XXX.keras' to your filename.
	9. Run the Compare_Vector.py:
		python Compare_Vector.py	(Own machine)
		python3 Compare_Vector.py	(Own machine)
		!python Compare_Vector.py	(Collab)
		!python3 Compare_Vector.py	(Collab)

---
Q:	There are issues on running the "Compare_Vector_V2.py"

A:	You need everything in the "Compare" folder. (The .txt files and the Lib_Compare_Vector.py)
	All of these files must be in the same folder.
	If your .keras file is in a different folder then you might want to set it as:
		PINN_File = "YOURFOLDERNAME/Backup-iter-10000.keras"
		
---		
Q:	What is the ideal value of beta?

A:	That depends on the quality of the data.

*The example picture use beta = 3.0
	
---
Q:	The loss curve doesnt make sense

A:	Maybe you set beta too low or too high


# FAQ - Error Estimation

Q:	The loss curve doesnt make sense

A:	Maybe you set beta too low or too high. Also, that is not the data loss, but the modified data loss. You need to convert it manually if you want to get the pure data loss.



---
*Will be updated if there are more questions
