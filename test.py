import dislocation as dl 
import numpy as np 

mu = 3e10
nu = 0.25

model = np.array([
    [440.58095043254673,3940.114839963042,15,80,50,50, 45, 0.01, 0.01, 0], 
	[440.58095043254673,3940.114839963042,15,80,50,50, 45, 0.01, 0.01, 0]])

# print(f"model: {model}")

obs = np.array([454, 3943, 0]).reshape([-1, 3])
#obs = np.array([454, 3943, 0])
print(f"obs: {obs}")

# length, width, depth, dip, strike, easting, northing, str-slip, dip-selip. opening

# TODO: single stations (or single fault patch) still has problem to be passed to C API. Still need test!

dl.okada_rect(obs, model, mu, nu)



