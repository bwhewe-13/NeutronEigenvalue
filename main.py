import multigroup as mg
import analysis as an
import numpy as np
import matplotlib.pyplot as plt
import math
import time
%matplotlib inline

# 70 Group Problem - Outer
# Critical Radius: 5.9671989440917965
# Load the Data
groups = 70 #number of groups the energy spectrum divided into

chi_out = np.loadtxt('chi_70_Pu_20pct240.csv') #birth rate for each of the different groups
centers_out = np.loadtxt('group_centers_70_Pu_20pct240.csv') #energies for the center of each of the 70 groups
edges_out = np.loadtxt('group_edges_70_Pu_20pct240.csv') #energies at the edges of the 70 groups (g+1)
nu_Sig_out = np.loadtxt('nuSigf_70_Pu_20pct240.csv') #Combination of neutron rate and the fission macroscopic cross section 
absorption_out = np.loadtxt('Siga_70G_Pu_20pct240.csv') #absorption cross section for the groups

temp2_out = np.loadtxt('D_70G_Pu_20pct240.csv') #diffusion coefficient for the different groups
temp3_out = np.loadtxt('Scat_70_Pu_20pct240.csv',delimiter=',') #macroscopic scattering cross sections for the different groups

# Need to get removal cross-section
# Combine the scattering and absorption cross sections to create total, subtract scattering for removal
temp_out = []
for i in range(70):
    temp_out.append(absorption_out[i]+np.sum(temp3_out, axis=0)[i]-temp3_out[i,i])

np.fill_diagonal(temp3_out,0) #remove i to i scattering in removal variable
removal_out = lambda r: temp_out #removal cross section function
scattering_out = temp3_out #scattering cross section function
diffusion_out = lambda r: temp2_out #diffusion coefficient cross section function
birth_rate_out = lambda r: chi_out #birthrate function
nu_Sig_fission_out = lambda r: nu_Sig_out #neutrons per fission multiplied by fission cross section function

# 70 Group Problem - Carbon - Inner
# Critical Radius: 25.44088821411133
# Load the Data
chi_in = np.loadtxt('chi_70_Pu_20pct240C.csv') #birth rate for each of the different groups
centers_in = np.loadtxt('group_centers_70_Pu_20pct240C.csv') #energies for the center of each of the 70 groups
edges_in = np.loadtxt('group_edges_70_Pu_20pct240C.csv') #energies at the edges of the 70 groups (g+1)
nu_Sig_in = np.loadtxt('nuSigf_70_Pu_20pct240C.csv') #Combination of neutron rate and the fission macroscopic cross section 
absorption_in = np.loadtxt('Siga_70G_Pu_20pct240C.csv') #absorption cross section for the groups

temp2_in = np.loadtxt('D_70G_Pu_20pct240C.csv') #diffusion coefficient for the different groups
temp3_in = np.loadtxt('Scat_70_Pu_20pct240C.csv',delimiter=',') #macroscopic scattering cross sections for the different groups

# Need to get removal cross-section
# Combine the scattering and absorption cross sections to create total, subtract scattering for removal
temp_in = []
for i in range(70):
    temp_in.append(absorption_in[i]+np.sum(temp3_in, axis=0)[i]-temp3_in[i,i])

np.fill_diagonal(temp3_in,0) #remove i to i scattering in removal variable
removal_in = lambda r: temp_in #removal cross section function
scattering_in = temp3_in #scattering cross section function
diffusion_in = lambda r: temp2_in #diffusion coefficient cross section function
birth_rate_in = lambda r: chi_in #birthrate function
nu_Sig_fission_in = lambda r: nu_Sig_in #neutrons per fission multiplied by fission cross section function

# Combine the Two Materials
#if r is <= split, outer is zero, if r > 0, inner is zero
split = 10 #edge of the inner ball, R is the total radius
removal = lambda r: removal_in(r)*(r <= split)+ removal_out(r)*(r>split) 
scattering = lambda r: scattering_in*(r <= split)+ scattering_out*(r>split)
diffusion = lambda r: diffusion_in(r)*(r <= split)+ diffusion_out(r)*(r>split)
birth_rate = lambda r: birth_rate_in(r)*(r <= split)+ birth_rate_out(r)*(r>split)
nu_Sig_fission = lambda r: nu_Sig_fission_in(r)*(r <= split)+ nu_Sig_fission_out(r)*(r>split)


# Other Parameters
groups = 70 #number of groups
R = 15 #total radius (cm)
BC = np.zeros((groups,2)) + 0.25 #Set up matrix of boundary conditions
BC[:,1] = 0.5*diffusion(R)
I = 100 #number of cells 
geometry = 2 #one-dimensional sphere 

#Set up matrix equation and solve for phi
A,B = mg.group_matrix(R,I,groups,BC,diffusion,scattering,birth_rate,nu_Sig_fission,
                              removal,geometry,epsilon=1.0e-6, LOUD=-1)

# Ostrowski
start_1 = time.time()
phi_1,rho_1,trace_1 = an.ostrowski(I,A,B,groups,epsilon=1.0e-8,tracker=True)
end_1 = time.time()
print('Time Elapsed:',end_1-start_1)

# Rayleigh Quotient
start_2 = time.time()
phi_2,lam_2,trace_2 = an.rayleighQuotient(I,A,B,groups,epsilon=1.0e-8,tracker=True)
end_2 = time.time()
print('Time Elapsed:',end_2-start_2)

# Inverse
start_3 = time.time()
phi_3, l_new_3,trace_3 = an.inversePower(I,A,B,groups,epsilon=1.0e-8, LOUD=True, tracker=True)
end_3 = time.time()
print('Time Elapsed:',end_3-start_3)

# Power
start_4 = time.time()
phi_4, l_new_4,trace_4 = an.power(I,A,B,groups,epsilon=1.0e-8, tracker=True)
end_4 = time.time()
print('Time Elapsed:',end_4-start_4)

# QR 
start_5 = time.time()
sm,all_val = an.qr(I,A,B,groups,20,all_val=True)    
end_5 = time.time()
print('Time Elapsed:',end_5-start_5)

t = 0
plt.plot(trace_4[t:],label='Power')
plt.plot(trace_3[t:],label='Inverse')
plt.plot(1/trace_2[t:],label='RQ')
plt.plot(1/trace_1[t:],label='Ostrowski')
plt.legend(loc='best')
plt.xlabel('Iteration')
# plt.ylim(0.65,0.7)
plt.show()

d = {'Method': ['Power', 'Inverse','Rayleigh Quotient','Ostrowoski','QR'], 'Time Elapsed (s)': [end_4-start_4,end_3-start_3,end_2-start_2,end_1-start_1,end_5-start_5]}
df = (pd.DataFrame(d))
df.index = np.arange(1, len(df)+1)
print(df)
