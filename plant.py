#! /usr/bin/python

import ctypes as ct
import time
import sys
import os
import Motiftoolbox.Tools.tools as tl
import numpy as np
import scipy.optimize as opt
import scipy.interpolate as trp

lib = ct.cdll.LoadLibrary(os.path.dirname(__file__)+'/lib/_plant.so')


CUDA_ENABLED = False
try:
	lib_cuda = ct.cdll.LoadLibrary(os.path.dirname(__file__)+'/lib/_plant_cuda.so')
	CUDA_ENABLED = True
except:
	pass

#===

PI2 = 2.*np.pi

params = {}
params['A'] = 0.15	  # ??
params['B'] = -50.	  # ??
params['C_1'] = 127./105.	  # ??
params['C_2'] = 8265./105.	  # ??
params['K_c'] = 0.0085	  # mV^-1
params['V_Ca'] = 140.	  # mV
#params['<++>'] = <++>	  # <++>
#params['<++>'] = <++>	  # <++>
#params['<++>'] = <++>	  # <++>

N_EQ1 = 6
N_EQ2 = 2*N_EQ1
N_EQ4 = 4*N_EQ1


def params_one():
	return np.array([params['V_Ca']])


def params_three():
	return np.array([params['V_Ca']])


#===
#===

lib.derivs_one.argtypes = [ct.POINTER(ct.c_double), ct.POINTER(ct.c_double), ct.POINTER(ct.c_double)]
def derivs_one(state):
	state = np.array(state, dtype=float)
	derivative = np.zeros((3), float)
	parameters = params_one()
	lib.derivs_one(state.ctypes.data_as(ct.POINTER(ct.c_double)),
			derivative.ctypes.data_as(ct.POINTER(ct.c_double)),
			parameters.ctypes.data_as(ct.POINTER(ct.c_double)))

	return derivative



lib.integrate_one_rk4.argtypes = [ct.POINTER(ct.c_double), ct.c_double, ct.c_uint, ct.c_uint, ct.POINTER(ct.c_double), ct.POINTER(ct.c_double)]
def integrate_one_rk4(initial_state, dt, N_integrate, stride=1):
	initial_state = np.asarray(initial_state)
	assert initial_state.size == N_EQ1
	X_out = np.zeros((N_EQ1*N_integrate), float)
	parameters = params_one()
	lib.integrate_one_rk4(initial_state.ctypes.data_as(ct.POINTER(ct.c_double)),
		ct.c_double(dt), ct.c_uint(N_integrate), ct.c_uint(stride),
		parameters.ctypes.data_as(ct.POINTER(ct.c_double)),
		X_out.ctypes.data_as(ct.POINTER(ct.c_double)))

	return np.reshape(X_out, (N_EQ1, N_integrate), 'F')

#===

initial_state = [-42.13934349, 0.51379418, 0.10720682, 0.74274042, 0.77535553, 0.1]
dt = 1.0
N_integrate = 5*10**4
stride = 10
THRESHOLD = -30.6
IDX_THRESHOLD = 0

def single_orbit(dt=dt, N_integrate=N_integrate, stride=stride, V_threshold=THRESHOLD):
	X = integrate_one_rk4(initial_state, dt=dt/float(stride), N_integrate=N_integrate, stride=stride)
	V, h, n, x, Ca, S= X[0], X[1], X[2], X[3], X[4], X[5]
	V_model, h_model, n_model, x_model, Ca_model, S_model =  tl.splineLS1D(), tl.splineLS1D(), tl.splineLS1D(), tl.splineLS1D(), tl.splineLS1D(), tl.splineLS1D()

	try:
		ni = tl.crossings(V, V_threshold) # convert to millivolts
		V, h, n, x, Ca, S = V[ni[-2]:ni[-1]], h[ni[-2]:ni[-1]], n[ni[-2]:ni[-1]], x[ni[-2]:ni[-1]], Ca[ni[-2]:ni[-1]], S[ni[-2]:ni[-1]]
		t = PI2*np.arange(V.size)/float(V.size-1)
		V_model.makeModel(V, t); h_model.makeModel(h, t); n_model.makeModel(n, t); x_model.makeModel(x, t); Ca_model.makeModel(Ca, t); S_model.makeModel(S, t)

	except:
		print '# single_orbit:  No closed orbit found!'
		raise ValueError

	
	T = dt*V.size

	return V_model, h_model, n_model, x_model, Ca_model, S_model, T


#===

lib.integrate_two_rk4.argtypes = [ct.POINTER(ct.c_double),
					ct.POINTER(ct.c_double),
					ct.POINTER(ct.c_double),
					ct.POINTER(ct.c_double),
					ct.c_double, ct.c_uint, ct.c_uint]

def integrate_two_rk4(initial_state, coupling, dt, N_integrate, stride=1):
	coup = np.zeros((2), float)
	coup[:coupling.size] = coupling

	initial_state = np.asarray(initial_state)
	X_out = np.zeros((2*N_integrate), float)
	parameters = params_three()

	lib.integrate_two_rk4(initial_state.ctypes.data_as(ct.POINTER(ct.c_double)),
				parameters.ctypes.data_as(ct.POINTER(ct.c_double)),
				coupling.ctypes.data_as(ct.POINTER(ct.c_double)),
				X_out.ctypes.data_as(ct.POINTER(ct.c_double)),
				ct.c_double(dt), ct.c_uint(N_integrate), ct.c_uint(stride))

	return np.reshape(X_out, (N_integrate, 2), 'C')



def alpha_m(V): return 0.1*(50.-V)/(np.exp((50.-V)/10.)-1.)
def beta_m(V):  return 4.*np.exp((25.-V)/18.)
def alpha_h(V): return 0.07*np.exp((25.-V)/20.)
def beta_h(V):  return 1./(1.+np.exp((55.-V)/10.))
def alpha_n(V): return 0.01*(55.-V)/(np.exp((55.-V)/10.)-1.)
def beta_n(V):  return 0.125*np.exp((45.-V)/80.)

def m_inf(V_tilde):	return alpha_m(V_tilde)/(alpha_m(V_tilde)+beta_m(V_tilde)) 
def h_inf(V_tilde):	return alpha_h(V_tilde)/(alpha_h(V_tilde)+beta_h(V_tilde))
def tau_h(V_tilde):	return 1./(alpha_h(V_tilde)+beta_h(V_tilde))
def n_inf(V_tilde):	return alpha_n(V_tilde)/(alpha_n(V_tilde)+beta_n(V_tilde))
def tau_n(V_tilde):	return 1./(alpha_n(V_tilde)+beta_n(V_tilde))
def x_inf(V):		return 1./(1.+np.exp(params['A']*(params['B']-V)))
def Vx_inf(x):		return params['B']-np.log(1./x-1.)/params['A']

def nullcline_h(V): # nullcline_h
	return h_inf(params['C_1']*V+params['C_2'])


def nullcline_n(V): # nullcline_n
	return n_inf(params['C_1']*V+params['C_2'])


def nullcline_x(x): # nullcline_x

	def func(Ca, x):
		return x_inf(params['V_Ca']-Ca/params['K_c']/x)-x

	Ca = list(opt.fsolve(func, [0.17], args=(x[0],)))
	for i in xrange(1, x.size):
		Ca.append(opt.fsolve(func, [Ca[-1]], args=(x[i],))[0])

	return Ca


def nullcline_Ca(x): # intersection of nullcline x and nullcline Ca
	return params['K_c']*x*(params['V_Ca']-Vx_inf(x))



if __name__ == '__main__':

	from pylab import *
	import time

	dt = 1.0
	stride = 10
	N = 8*10**4
	t = dt*arange(N)
	

	X = integrate_one_rk4(initial_state, dt=dt/float(stride), N_integrate=N, stride=stride)
	
	ax = subplot(111)
	plot(t[::5], X[0][::5], 'k')
	ylabel('V')
	show()
	
	X = integrate_two_rk4(0.1*rand(2), coupling=zeros((2), float), dt=dt.float(stride), N_integrate=N, stride=stride)
	t = dt*arange(X.shape[0])
	plot(t, X[:, 0])
	plot(t, X[:, 1])
	show()
