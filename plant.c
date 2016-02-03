
/*
 * Three-cell Leech heart interneuron model is implimented 
 * as in the draft  "Bifurcations of bursting polyrhythms
 * in 3-cell central pattern generators" of Jeremy Wojcik,
 * Robert Clewley, and Andrey Shilnikov, prepared for
 * Journal of Neuroscience 2012.
 */


#include <math.h>
#include <stdio.h>


#define N_EQ1	6
#define N_EQ2	2*N_EQ1
#define N_EQ4	4*N_EQ1

#define C_m	1.	  // uF/cm^2
#define V_Na	30.	  // mV
//#define V_Ca	140.	  // mV
#define V_K	-75.	  // mV
#define V_L	-40.	  // mV
#define g_Na	4.	// mmho/cm^2
#define g_Ca	0.004	// mmho/cm^2
#define g_K	0.3	// mmho/cm^2
#define g_K_Ca	0.03	// mmho/cm^2
#define g_L	0.003	// mmho/cm^2
#define C_1	127./105.
#define C_2	8265./105.
#define lambda	1./12.5
#define tau_x	235.	// ms ??
#define A	0.15	// ??
#define Bt	0.02
#define th	20
#define B	-50.	// ??
#define rho	0.0003	// ms^-1
#define K_c	0.0085	// mV^-1
#define THRESHOLD_SLOPE	1.	// mV^-1
#define THRESHOLD	-30.  // mV
#define E_syn	-62.5  // mV
//#define <++>	<++>	// <++>
//#define <++>	<++>	// <++>



double alpha_m(const double V)		{ return 0.1*(50.-V)/(exp((50.-V)/10.)-1.); } 
double beta_m(const double V)		{ return 4.*exp((25.-V)/18.); } 
double alpha_h(const double V)		{ return 0.07*exp((25.-V)/20.); } 
double beta_h(const double V)		{ return 1./(1.+exp((55.-V)/10.)); } 
double alpha_n(const double V)		{ return 0.01*(55.-V)/(exp((55.-V)/10.)-1.); } 
double beta_n(const double V)		{ return 0.125*exp((45.-V)/80.); }

double m_inf(const double V_tilde)	{ return alpha_m(V_tilde)/(alpha_m(V_tilde)+beta_m(V_tilde)); } 
double h_inf(const double V_tilde)	{ return alpha_h(V_tilde)/(alpha_h(V_tilde)+beta_h(V_tilde)); }
double tau_h(const double V_tilde)	{ return 1./(alpha_h(V_tilde)+beta_h(V_tilde)); }
double n_inf(const double V_tilde)	{ return alpha_n(V_tilde)/(alpha_n(V_tilde)+beta_n(V_tilde)); }
double tau_n(const double V_tilde)	{ return 1./(alpha_n(V_tilde)+beta_n(V_tilde)); }
double x_inf(const double V)		{ return 1./(1.+exp(A*(B-V))); }
double boltzmann(const double x, const double x_0, const double k)	{return 1./(1.+exp(-k*(x-x_0)));}

void derivs_one(const double* y, double* dydt, const double* p)
{
	double V=y[0], h=y[1], n=y[2], x=y[3], Ca=y[4], S=y[5];
	double V_tilde = C_1*V+C_2, V_Ca=p[0];

	dydt[0] = -g_Na*powf(m_inf(V_tilde), 3)*h*(V-V_Na) - g_Ca*x*(V-V_Ca) - (g_K*powf(n, 4)+g_K_Ca*Ca/(0.5+Ca))*(V-V_K) - g_L*(V-V_L);	// dV/dt
        dydt[1] = lambda*(h_inf(V_tilde)-h)/tau_h(V_tilde);	// dh/dt
        dydt[2] = lambda*(n_inf(V_tilde)-n)/tau_n(V_tilde);	// dn/dt
        dydt[3] = (x_inf(V)-x)/tau_x;				// dx/dt
        dydt[4] = rho * (K_c * x * (V_Ca - V) - Ca);		// d[Ca2+]/dt
	dydt[5] = A*(1.-S)*boltzmann(V, th, 100.)-Bt*S;         // dS/dt
};



void integrate_one_rk4(double* y, const double dt, const unsigned N, const unsigned stride, const double* P, double* output)
{
	unsigned i, j, k;
	double dt2, dt6;
	double y1[N_EQ1], y2[N_EQ1], k1[N_EQ1], k2[N_EQ1], k3[N_EQ1], k4[N_EQ1];
	dt2 = dt/2.; dt6 = dt/6.;

	for(j=0; j<N_EQ1; j++)
		output[j] = y[j];



	for(i=1; i<N; i++)
	{
		for(j=0; j<stride; j++)
		{
			derivs_one(y, k1, P);
			for(k=0; k<N_EQ1; k++)
			       y1[k] = y[k]+k1[k]*dt2; 			

			derivs_one(y1, k2, P);
			for(k=0; k<N_EQ1; k++)
				y2[k] = y[k]+k2[k]*dt2; 			

			derivs_one(y2, k3, P);
			for(k=0; k<N_EQ1; k++)
				y2[k] = y[k]+k3[k]*dt; 			

			derivs_one(y2, k4, P);
			for(k=0; k<N_EQ1; k++)
				y[k] += dt6*(k1[k]+2.*(k2[k]+k3[k])+k4[k]);
		}
		//printf("%i %lf\n", i, y[0]);
		for(j=0; j<N_EQ1; j++) output[N_EQ1*i+j] = y[j];
	}
};

void derivs_two(const double* y, double* dxdt, const double* p, const double* kij) {

	unsigned i, j;
	double bm_factor[2], bmf_sum;
	for(i=0; i<2; i++){
		bm_factor[i] = boltzmann(y[i*N_EQ1], COUPLING_THRESHOLD, THRESHOLD_SLOPE);
		derivs_one(y+i*N_EQ1, dxdt+i*N_EQ1, p);
	}
	for(i=0; i<2; i++) {
		bmf_sum = 0.;
		for(j+0; j<i; j++){
			bmf_sum += bm_factor[j];
		}
		for(j=i+1; j<2; j++){
			bmf_sum += bmfactor[j];
		}
		dxdt[i*N_EQ1] += g_inh*(p[4]-y[i*N_EQ1])*bmf_sum;
	}

};

void integrate_two_rk4(double* y, const double* params, const double* coupling, double* output, const double dt, const unsigned N, const unsigned stride)
{
	unsigned i, j, k;
	double dt2, dt6;
	double y1[N_EQ2], y2[N_EQ2], k1[N_EQ2], k2[N_EQ2], k3[N_EQ2], k4[N_EQ2];
	dt2 = dt/2.; dt6 = dt/6.;

	for(j=0; j<2; j++)
		output[j] = y[N_EQ1*j]; 	

	for(i=1; i<N; i++)
	{
		for(j=0; j<stride; j++)
		{
			derivs_two(y, k1, params, coupling);
			for(k=0; k<N_EQ2; k++)
				y1[k] = y[k]+k1[k]*dt2; 			

			derivs_two(y1, k2, params, coupling);

			for(k=0; k<N_EQ2; k++)
				y2[k] = y[k]+k2[k]*dt2; 			
			derivs_two(y2, k3, params, coupling);

			for(k=0; k<N_EQ2; k++)
				y2[k] = y[k]+k3[k]*dt; 			

			derivs_two(y2, k4, params, coupling);
			for(k=0; k<N_EQ2; k++)
				y[k] += dt6*(k1[k]+2.*(k2[k]+k3[k])+k4[k]);
		}
		for(j=0; j<2; j++)
			output[2*i+j] = y[N_EQ1*j]; 					
	}
};





