
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
#define N_EQ2   2*N_EQ1
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
#define tau_x	9400.	// ms ??
#define A	0.3	// ??
#define B	-40.	// ??
#define rho	0.00015	// ms^-1
#define K_c	0.00425	// mV^-1
#define THRESHOLD_SLOPE	1.	// mV^-1
#define THRESHOLD	-30.  // mV
#define E_syn	-80.  // mV
#define shift   -2.
//#define shift2   250.
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

double boltzmann(const double x, const double x_0, const double k)	
{
	return 1./(1.+exp(-k*(x-x_0)));
}

void derivs_one(const double* y, double* dydt, const double* p)
{
	double V=y[0], h=y[1], n=y[2], x=y[3], Ca=y[4], S=y[5];
	double V_tilde = C_1*V+C_2, V_Ca=p[0];

	dydt[0] = -g_Na*powf(m_inf(V_tilde), 3)*h*(V-V_Na) - g_Ca*x*(V-V_Ca) - (g_K*powf(n, 4)+g_K_Ca*Ca/(0.5+Ca))*(V-V_K) - g_L*(V-V_L);	// dV/dt
        dydt[1] = lambda*(h_inf(V_tilde)-h)/tau_h(V_tilde);	// dh/dt
        dydt[2] = lambda*(n_inf(V_tilde)-n)/tau_n(V_tilde);	// dn/dt
        dydt[3] = (x_inf(V)-x)/tau_x;				// dx/dt
        dydt[4] = rho * (K_c * x * (V_Ca - V - shift) - Ca);		// d[Ca2+]/dt
	dydt[5] = A*(1.-S)*boltzmann(V, 20., 100.)-0.02*S;  // dS/dt
}



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

void derivs_two(const double* y, double* dydt, const double* p, const double* kij)
{

	double V=y[0], h=y[1], n=y[2], x=y[3], Ca=y[4], S=y[5];
        double V2=y[6], h2=y[7], n2=y[8],x2=y[9], Ca2=y[10], S2=y[11];
	double V_tilde = C_1*V+C_2, V_Ca=p[0];
	double V_tilde2 = C_1*V2+C_2;
	
	//Cell One
	dydt[0] = -g_Na*powf(m_inf(V_tilde), 3)*h*(V-V_Na) - g_Ca*x*(V-V_Ca) - (g_K*powf(n, 4)+g_K_Ca*Ca/(0.5+Ca))*(V-V_K) - g_L*(V-V_L);	// dV/dt
        dydt[1] = lambda*(h_inf(V_tilde)-h)/tau_h(V_tilde);	// dh/dt
        dydt[2] = lambda*(n_inf(V_tilde)-n)/tau_n(V_tilde);	// dn/dt
        dydt[3] = (x_inf(V)-x)/tau_x;				// dx/dt
        dydt[4] = rho * (K_c * x * (V_Ca - V - shift) - Ca);		// d[Ca2+]/dt
	dydt[5] = 0.15*(1.-S)*boltzmann(V, 20., 100.)-0.02*S;  // dS/dt
	//Cell Two	
	dydt[6]  = -g_Na*powf(m_inf(V_tilde2), 3)*h2*(V2-V_Na) - g_Ca*x*(V2-V_Ca) - (g_K*powf(n2, 4)+g_K_Ca*Ca2/(0.5+Ca2))*(V2-V_K) - g_L*(V2-V_L);	// dV/dt
        dydt[7]  = lambda*(h_inf(V_tilde2)-h2)/tau_h(V_tilde2);	// dh/dt
        dydt[8]  = lambda*(n_inf(V_tilde2)-n2)/tau_n(V_tilde2);	// dn/dt
        dydt[9]  = (x_inf(V2)-x2)/tau_x;				// dx/dt
        dydt[10] = rho * (K_c * x2 * (V_Ca - V2 - shift) - Ca2);		// d[Ca2+]/dt
	dydt[11] = 0.15*(1.-S)*boltzmann(V, 20., 100.)-0.02*S;  // dS/dt
	
	//Coupling With Alpha Synapse	
	dydt[0] +=       (E_syn-V)*(kij[0]*(boltzmann(dydt[1], THRESHOLD, THRESHOLD_SLOPE)/C_m)*dydt[11]);
	dydt[N_EQ1] +=   (E_syn-V2)*(kij[1]*(boltzmann(dydt[0], THRESHOLD, THRESHOLD_SLOPE)/C_m)*dydt[5]);

	dydt[0]       	  += (kij[12]*(V2-V))/C_m;
	dydt[N_EQ1]       += (kij[12]*(V-V2))/C_m;
};

void integrate_two_rk4(double* y, const double* params, const double* coupling, double* output, const double dt, const unsigned N, const unsigned stride)
{
	unsigned i, j, k;
	double dt2, dt6;
	double y1[N_EQ4], y2[N_EQ4], k1[N_EQ4], k2[N_EQ4], k3[N_EQ4], k4[N_EQ4];
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

void derivs_four(const double* y, double* dydt, const double* p, const double* kij)
{
	int i;
	double bm_factor[4], V[4];

	for(i=0; i<4; i++)
	{
		V[i] = y[i*N_EQ1];
		derivs_one(y+i*N_EQ1, dydt+i*N_EQ1, p);
		bm_factor[i] = boltzmann(V[i], THRESHOLD, THRESHOLD_SLOPE)/C_m;
	}

	dydt[0] +=       (E_syn-V[0])*(kij[0]*bm_factor[1] + kij[1]* bm_factor[2] + kij[2]* bm_factor[3]);
	dydt[N_EQ1] +=   (E_syn-V[1])*(kij[3]*bm_factor[0] + kij[4]* bm_factor[2] + kij[5]* bm_factor[3]);
	dydt[2*N_EQ1] += (E_syn-V[2])*(kij[6]*bm_factor[0] + kij[7]* bm_factor[1] + kij[8]* bm_factor[3]);
	dydt[3*N_EQ1] += (E_syn-V[3])*(kij[9]*bm_factor[0] + kij[10]*bm_factor[1] + kij[11]*bm_factor[2]);

	dydt[0]       += (kij[12]*(V[1]-V[0]) + kij[13]*(V[2]-V[0]) + kij[14]*(V[3]-V[0]))/C_m;
	dydt[N_EQ1]   += (kij[12]*(V[0]-V[1]) + kij[15]*(V[2]-V[1]) + kij[16]*(V[3]-V[1]))/C_m;
	dydt[2*N_EQ1] += (kij[13]*(V[0]-V[2]) + kij[15]*(V[1]-V[2]) + kij[17]*(V[3]-V[2]))/C_m;
	dydt[3*N_EQ1] += (kij[14]*(V[0]-V[3]) + kij[16]*(V[1]-V[3]) + kij[17]*(V[2]-V[3]))/C_m;
};

void integrate_four_rk4(double* y, const double* params, const double* coupling, double* output, const double dt, const unsigned N, const unsigned stride)
{
	unsigned i, j, k;
	double dt2, dt6;
	double y1[N_EQ4], y2[N_EQ4], k1[N_EQ4], k2[N_EQ4], k3[N_EQ4], k4[N_EQ4];
	dt2 = dt/2.; dt6 = dt/6.;

	for(j=0; j<4; j++)
		output[j] = y[N_EQ1*j];

	for(i=1; i<N; i++)
	{
		for(j=0; j<stride; j++)
		{
			derivs_four(y, k1, params, coupling);
			for(k=0; k<N_EQ4; k++)
				y1[k] = y[k]+k1[k]*dt2; 			

			derivs_four(y1, k2, params, coupling);

			for(k=0; k<N_EQ4; k++)
				y2[k] = y[k]+k2[k]*dt2; 			
			derivs_four(y2, k3, params, coupling);

			for(k=0; k<N_EQ4; k++)
				y2[k] = y[k]+k3[k]*dt; 			

			derivs_four(y2, k4, params, coupling);
			for(k=0; k<N_EQ4; k++)
				y[k] += dt6*(k1[k]+2.*(k2[k]+k3[k])+k4[k]);
		}
		for(j=0; j<4; j++)
			output[4*i+j] = y[N_EQ1*j]; 					
	}
};





