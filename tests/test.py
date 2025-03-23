import dislocations as dl
import numpy as np 

mu = 3.3e10
nu = 0.25

#model = np.array([
#    440               ,3940             ,15,80,50,50, 45,    1,    1, 0, 
#	])

model = np.array([
    [440.58095043254673,3940.114839963042,15,80,50,50, 45, 0.01, 0.01, 0], 
	[440.58095043254673,3940.114839963042,15,80,50,50, 45, 0.01, 0.01, 0]])
# print(f"model: {model}")

obs = np.array([
    [454, 3943, 10],
    [454, 3943, 10],
    [454, 3943, 0],
    [454, 3943, 0],
    [454, 3943, 0],
    [454, 3943, 0],
    [454, 3943, 10],
    [454, 3943, 0],
    [454, 3943, 0]])
#obs = np.array([454, 3943, 0])
#print(f"obs: {obs}")

# length, width, depth, dip, strike, easting, northing, str-slip, dip-selip. opening


results = dl.rde(obs, model, mu, nu)
print(f"U = {results[0]}")
#print(f"D = {results[1]}")
print(f"S = {results[1]}")
print(f"E = {results[2]}")
print(f"flags = {results[3]}")
print(results[1].shape)

[U, S, E, flags] = dl.rde(obs, model, mu, nu)

print(U)

model_tde = np.array([ [-1, -1, -4, 1, -1, -3, -1, -1, -2, 1, -1, 2] ])
obs_tde = np.array([
    [-1/3, -1/3, 0],
    [-1/3, -1/3, -3],
    [-1/3, -1/3, -5],
    [-1/3, -1/3, -8],
    # [7,     -1,  -5],
    # [-7,   -1,   -5],
    # [-1,   -3,   -6,]
])
        # -1/3.0, -1/3.0, 0.0, -1/3.0, -1/3.0, -3.0, -1/3.0, -1/3.0, -5.0, -1/3.0, -1/3.0, -8.0


[Utede, Stede, Etede] = dl.tde(obs_tde, model_tde, mu, nu)
print("------------------- Result TDE mehdi ------------------------------")
print(f"Utede = {Utede}")
print(f"Stede = {Stede}")
print(f"Etede = {Etede}")

[Utede2, Stede2, Etede2] = dl.tde_meade(obs_tde, model_tde, mu, nu)
print("------------------- Result TDE meade ------------------------------")
print(f"Utede2 = {Utede2}")
print(f"Stede2 = {Stede2}")
print(f"Etede2 = {Etede2}")
