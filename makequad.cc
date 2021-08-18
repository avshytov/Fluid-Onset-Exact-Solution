#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <Python.h>
#include <math.h>
#include <stdio.h>
#include <malloc.h>
#include <stdlib.h>
#include <numpy/numpyconfig.h>
#include <numpy/arrayobject.h>
//#include <npy_math.h>
#include <ndarrayobject.h>
//#include "pmrand.h"
#include <complex>

static PyObject *error_obj;

typedef std::complex<double> cdouble;
//typedef double  cdouble;

typedef cdouble (*weight_func)(double q1, double q2, cdouble z, void *param);
typedef struct {
  weight_func integrate_const;
  weight_func integrate_linear;
  weight_func integrate_quadratic;
} quad_weights;




static
void get_poly_weights(double qref, double q1, double q2, double q3,
		      double M[3][3]) {
  double x1 = q1 - qref;
  double x2 = q2 - qref;
  double x3 = q3 - qref;
  double W;
  M[0][0] = x2 * x3 * (x3 - x2);
  M[1][0] = (x2 - x3) * (x2 + x3);
  M[2][0] =  x3 - x2;
  M[0][1] =  x1 * x3 * (x1 - x3);
  M[1][1] = (x3 - x1) * (x3 + x1);
  M[2][1] = x1 - x3;
  M[0][2] = x1 * x2 * (x2 - x1);
  M[1][2] = (x1 - x2) * (x1 + x2);
  M[2][2] = x2 - x1;
  W = (x3 - x2) * (x3 - x1) * (x2 - x1);
  for (int i = 0; i < 3; i ++)
    for (int k = 0; k < 3; k ++)
      M[i][k] /= W;
  return;
}

static cdouble complexI(0, 1);


static
void get_quad_weights(const double M[3][3],
		      double qa, double qb, cdouble z,
		      cdouble *wt, quad_weights *wts, void *param)
{
  cdouble I[3];
  //double M[3][3];
  //find_Ia (qa, qb, x, &I[0]);
  //find_Ib (qa, qb, x, &I[1]);
  //find_Ic (qa, qb, x, &I[2]);
  I[0] = wts->integrate_const (qa, qb, z, param);
  I[1] = wts->integrate_linear(qa, qb, z, param);
  I[2] = wts->integrate_quadratic(qa, qb, z, param);
  //if (isnan(I[0].real()) || (isnan(I[1].real())) || (isnan(I[2].real()))) {
  //    fprintf (stderr, "nan: qa = %g qb = %g x = %g ", qa, qb, x);
  //}
  for (int i = 0; i < 3; i ++) {
    wt[i] = 0;
    for (int j = 0; j < 3; j ++) {
      wt[i] += M[j][i] * I[j];
    }
  }
}

static
void make_poly_quad(PyArrayObject *Qre,
		    PyArrayObject *Qim,
			cdouble *z, double *q, int Nx, int Nq,
			quad_weights *wts, void *param) {
  //cdouble *F = NULL;
  cdouble wt[3];
  double  M[3][3] = {{0, 0, 0}, {0, 0, 0}, {0, 0, 0}};
  //F = PyMem_New(cdouble, Nx * Nq);
  //if (F == NULL) {
  //  fprintf (stderr, "Cannot allocate F");
  //  return NULL;
  //}
  for(int j = 0; j < Nq; j++) {
    for (int i = 0; i < Nx; i++) {
      //F[i * Nq + j] = 0.0;
      *(double *)PyArray_GETPTR2(Qre, i, j) = 0.0;
      if (Qim) *(double *)PyArray_GETPTR2(Qim, i, j) = 0.0;
    }
  }
  //return F;
  for(int j = 1; j < Nq - 1; j++) {
    double q1 = q[j - 1];
    double q2 = q[j];
    double q3 = q[j + 1];
    get_poly_weights(q1, q1, q2, q3, M);
    for (int i = 0; i < Nx; i++) {
      get_quad_weights(M, q1, q3, z[i], wt, wts, param);
      for (int k = 0; k < 3; k ++) {
	//F[i * Nq + k + j - 1] += 0.5 * wt[k];
        *(double *)PyArray_GETPTR2(Qre, i, j + k - 1) += 0.5 * wt[k].real();
        if (Qim)
          *(double *)PyArray_GETPTR2(Qim, i, j + k - 1) += 0.5 * wt[k].imag();
      }
    }
  }
  //return F;
  double q1 = q[0];
  double q2 = q[1];
  double q3 = q[2];
  get_poly_weights(q1, q1, q2, q3, M);
  for (int i = 0; i < Nx; i++) {
      get_quad_weights(M, q1, q2, z[i], wt, wts, param);
      for (int k = 0; k < 3; k ++) {
	//F[i * Nq + k] += 0.5 * wt[k];
        *(double *)PyArray_GETPTR2(Qre, i, k) += 0.5 * wt[k].real();
        if (Qim)
         *(double *)PyArray_GETPTR2(Qim, i, k) += 0.5 * wt[k].imag();
      }
  }
  q1 = q[Nq - 3];
  q2 = q[Nq - 2];
  q3 = q[Nq - 1];
  get_poly_weights(q2, q1, q2, q3, M);
  for (int i = 0; i < Nx; i++) {
      get_quad_weights(M, q2, q3, z[i], wt, wts, param);
      for (int k = 0; k < 3; k ++) {
	//F[i * Nq + Nq - 3 + k] += 0.5 * wt[k];
        *(double *)PyArray_GETPTR2(Qre, i, k + Nq - 3) += 0.5 * wt[k].real();
	      if (Qim)
	   *(double *)PyArray_GETPTR2(Qim, i, k + Nq - 3) += 0.5 * wt[k].imag();
      }
  }
  //return F;
}


cdouble fourier_const(double qa, double qb, cdouble z, void *dummy) {
  double x = z.real();
  cdouble ex =  exp(complexI * qa * x);
  double dqx = (qb - qa) * x;
  cdouble edqx = exp(complexI * dqx);
  if (fabs(dqx) > 1e-8) {
      return  ex * (edqx - 1.0) / complexI / x;
  } else {
      return ex * (qb - qa);
  }
}

cdouble fourier_linear(double qa, double qb, cdouble z, void *dummy) {
  double x = z.real();
  cdouble ex =  exp(complexI * qa * x);
  double dqx = (qb - qa) * x;
  cdouble eqdx = exp(complexI * dqx);
  if (fabs(dqx) > 1e-4) {
      cdouble zx = 1.0 - eqdx * (1.0 - complexI * dqx);
      return  - ex * zx / x / x;
  } else {
      return ex * (qb - qa) * (qb - qa)/ 2.0;
  }
}

cdouble fourier_quadratic(double qa, double qb, cdouble z, void *dummy) {
  double x = z.real();
  cdouble ex =  exp(complexI * qa * x);
  double dqx = (qb - qa) * x;
  cdouble idqx = complexI * dqx;
  cdouble eqdx = exp(idqx);
  return 0.0;
  if (fabs(dqx) > 1e-3) {
      cdouble zx = -1.0 + 1.0 * eqdx * ((1.0 - idqx)  + idqx * idqx / 2.0);
      return   2.0 * ex * zx / x / x / x * complexI;
  } else {
      return ex * (qb - qa) * (qb - qa) * (qb - qa)/ 3.0;
  }
}



static quad_weights fourier_weights = {
  fourier_const,
  fourier_linear,
  fourier_quadratic
};

bool is_between(double qa, double qb, double qi) {
  if (qb < qa) {
    double tmp = qb;
    qb = qa;
    qa = tmp;
    fprintf (stderr, "!!! exchange %g and %g\n", qb, qa);
  }
  double h = fabs(qa - qb);
  if (qi <= qa - 1e-6*h) return false;
  if (qi >= qb + 1e-6*h) return false;
  return true;
}

double hilbert_log(double qa, double qb, double qi) {
      double m = (qa + qb) / 2.0;
      double d = m - qi;
      double dq = qb - qa;
      double xi = 0.5 * dq / d;
      if (fabs(xi) > 1e-3) {
         return log(fabs((1.0 + xi)/(1.0 - xi)));
      } else {
	 double xi2 = xi * xi;
	 double xi4 = xi2 * xi2;
	 return 2.0 * xi * (1.0 + 1.0/3.0 * xi2 + 1.0/5.0 * xi4 );
      }
}

cdouble hilbert_const(double qa, double qb, cdouble z, void *dummy) {
  double qi = z.real();
  //return 0.0;
  //double h = fabs(qb - qa);
  //if ((fabs(qa - qi) < 5*h) || (fabs(qb - qi) < 5*h)) {
  if (is_between(qa, qb, qi)) {
      return 0.0;
  } else {
      return hilbert_log(qa, qb, qi);
  }
}

cdouble hilbert_linear(double qa, double qb, cdouble z, void *dummy) {
  double qi = z.real();
  //return 0.0;
  //double h = fabs(qb - qa);
  //if ((fabs(qa - qi) < 0.5 * h)) {
  if (is_between(qa, qb, qi)) {
      return qb - qa;
  } else {
      double dq = qb - qa;
      double d  = qa - qi;
      double xi = dq / d;
      if (fabs(xi) > 1e-4) {
         return qb - qa + (qi - qa) * hilbert_log(qa, qb, qi);
      } else {
	return d * xi  * xi * (0.5 - xi * (1.0/3.0 - xi * (0.25 - 0.2 * xi)));
      }
  }
}

cdouble hilbert_quadratic(double qa, double qb, cdouble z, void *dummy) {
  double qi = z.real();
  //return 0.0;
  //double h = fabs(qb - qa);
  //return 0.0;
  if (is_between(qa, qb, qi)) {
    //return qb - qa;
    return (qb - qa) * (-0.5 * qb - qi + 1.5 * qa);
  } else {
    double dq = qb - qa;
    double m = (qa + qb)/2.0;
    double d = m - qi;
    double xi = dq / d;
    if (fabs(xi) > 1e-3) {
       //return qb - qa + qi * log(fabs((qb - qi)/(qa - qi)));
      double ret =  dq * d ;
      //(0.5 * qb - 1.5 * qa - 1.0 * qi) * (qb - qa) / 2.0 ;
      ret += 2.0 * (qi - qa) * dq;
      ret += (qi - qa) * (qi - qa) * hilbert_log(qa, qb, qi);
      return ret;
    } else {
      double eta = dq / (qa - qi);
      return (qa - qi) * (qa - qi) * eta * eta * eta * (1.0/3.0 - eta * (0.25 - 0.2 * eta));
    }
  }
}

void hilbert_inf_corr(PyArrayObject *arr_H, double *q, int Nq) {
  double q_last = q[Nq - 1], q_first = q[0];
  double h_last = q[Nq - 1] - q[Nq - 2];
  double h_first = q[1] - q[0];
  //return;
  for (int i = 0; i < Nq; i ++) {
      double qi = q[i];
      if (fabs(qi - q_last) < fabs(h_last)) {
        //H[i * Nq + Nq - 1] += 0.0;
	*(double *)PyArray_GETPTR2(arr_H, i, Nq - 1) += 0;
      } else {
	 double wt = 1.0;
	 if (fabs(qi) > 1e-6 * fabs(q_last))
	   wt = q_last / qi * log(fabs(q_last/(q_last - qi)));
	 //H[i * Nq + Nq - 1] += wt;
	   *(double *)PyArray_GETPTR2(arr_H, i, Nq - 1) += wt;
      }

      if ((fabs(qi - q_first) < fabs(h_first))) {
	//H[i * Nq + 0] += 0.0;
	*(double *)PyArray_GETPTR2(arr_H, i, 0) += 0.0;
      } else {
	 double wt = 1.0;
	 if (fabs(qi) > 1e-6 * fabs(q_first))
	   wt = q_first / qi * log(fabs(q_first/(q_first - qi)));
	 //H[i * Nq + 0] -= wt;
	 *(double *)PyArray_GETPTR2(arr_H, i, 0) -= wt;
      }
  }
}

void hilbert_inf_corr_2(PyArrayObject *arr_H, double *q, int Nq) {
  double q_last = q[Nq - 1], q_first = q[0];
  double h_last = q[Nq - 1] - q[Nq - 2];
  double h_first = q[1] - q[0];
  //return;
  for (int i = 0; i < Nq; i ++) {
      double z1 = q_last/q[Nq - 1], z2 = q_last / q[Nq - 21],
	z3 = q_last / q[Nq - 41];
      double M[3][3], I[3] = {0.0, 0.0, 0.0};
      get_poly_weights(0.0, z1, z2, z3, M);       
      double qi = q[i];
      if (fabs(qi - q_last) < fabs(h_last)) {
        //H[i * Nq + Nq - 1] += 0.0;
	I[1] = 0.0;
	I[2] = 0.0; 
	//*(double *)PyArray_GETPTR2(arr_H, i, Nq - 1) += 0;
      } else {
	//double wt = 1.0;
	 double  ln = log(fabs(q_last/(q_last - qi)));
	 if (fabs(qi) > 1e-6 * fabs(q_last)) {
  	   I[1] = q_last / qi * ln;
	   I[2] = 0 * q_last / qi * (-1.0 + q_last / qi * ln); 
	  //H[i * Nq + Nq - 1] += wt;
	  //  *(double *)PyArray_GETPTR2(arr_H, i, Nq - 1) += wt;
	 }
      }
      for (int k = 0; k < 3; k ++) {
	for (int l = 0; l < 3; l++) {
          *(double *)PyArray_GETPTR2(arr_H, i, Nq - 1 - 20*k) += M[l][k] * I[l]; 
	}
      }
      //  for (int i = 0; i < 3; i ++) {
      //wt[i] = 0;
      ///for (int j = 0; j < 3; j ++) {
      //wt[i] += M[j][i] * I[j];
      //}
      //}

      z1 = q_first / q[0], z2 = q_first / q[20], z3 = q_first / q[40];
      get_poly_weights(0.0, z1, z2, z3, M);       
      if ((fabs(qi - q_first) < fabs(h_first))) {
	//H[i * Nq + 0] += 0.0;
	//*(double *)PyArray_GETPTR2(arr_H, i, 0) += 0.0;
      } else {
	 //double wt = 1.0;
	 double  ln = log(fabs(q_first/(q_first - qi)));
	 if (fabs(qi) > 1e-6 * fabs(q_first)) {
	  I[1] =  q_first / qi * ln;
  	  I[2] =  - 0 * q_first / qi * (1.0 - q_first / qi * ln);
      }
	  //double wt = 1.0;
	 //if (fabs(qi) > 1e-6 * fabs(q_first))
	   //wt = q_first / qi * log(fabs(q_first/(q_first - qi)));
	 //H[i * Nq + 0] -= wt;
	 //*(double *)PyArray_GETPTR2(arr_H, i, 0) -= wt;
      }
      for (int k = 0; k < 3; k ++) {
	for (int l = 0; l < 3; l++) {
          *(double *)PyArray_GETPTR2(arr_H, i, 0 + k*20) += M[l][k] * I[l]; 
	}
      }
  }
}

double hilbert_log_z(double xa, double xb, double y) {
  return 0.5 * log((xb * xb + y * y)/( xa * xa + y * y));
}

double hilbert_atan_z(double xa, double xb, double y) {
  return atan(xb/y) - atan(xa/y);
}

cdouble hilbert_z_const(double qa, double qb, cdouble z, void *dummy) {
  double x = z.real();
  double y = z.imag();
  //fprintf (stderr, "h-z-const, z = %g + i * %g, q = %g %g\n", x, y, qa, qb);
  cdouble ret = 0.0;
  double xa = qa - x;
  double xb = qb - x;
  ret += hilbert_log_z(xa, xb, y);
  ret +=  complexI * hilbert_atan_z(xa, xb, y);
  //fprintf (stderr, "ret= %g %g\n", ret.real(), ret.imag()) ;
  return ret;
}

cdouble hilbert_z_linear(double qa, double qb, cdouble z, void *dummy) {
  //return 0.0; 
  double x = z.real();
  double y = z.imag();
  cdouble ret = 0.0;
  double xa = qa - x;
  double xb = qb - x;
  cdouble lnz = hilbert_log_z(xa, xb, y)
              + complexI * hilbert_atan_z (xa, xb, y) ;
  cdouble za = complexI * y - xa;
  //ret  += xb - xa -  (y + complexI * xa) * hilbert_atan_z(xa, xb, y);
  //ret += (complexI * y  - xa) * hilbert_log_z(xa, xb, y);
  ret = (xb - xa) + za * lnz;
  return ret;
}

cdouble hilbert_z_quadratic(double qa, double qb, cdouble z, void *dummy) {
  //return 0.0; 
  double x = z.real();
  double y = z.imag();
  cdouble ret = 0.0;
  double xa = qa - x;
  double xb = qb - x;
  ret += (0.5 * (xb + xa) + complexI * y - 2 * xa) *  (xb - xa);
  cdouble lnz = hilbert_log_z(xa, xb, y)
              + complexI * hilbert_atan_z (xa, xb, y) ;
  cdouble za = complexI * y - xa;
  ret += za * za * lnz;
  //return 0.0;
  return ret;
}


void hilbert_inf_corr_z(PyArrayObject *arr_H_re,
                        PyArrayObject *arr_H_im,
                        double *q, cdouble *z, int Nq, int Nz) {
  double q_last = q[Nq - 1], q_first = q[0];
  double h_last = q[Nq - 1] - q[Nq - 2];
  double h_first = q[1] - q[0];
  //return;
  for (int i = 0; i < Nz; i ++) {
      double x_last = q_last - z[i].real();
      double x_first = q_first - z[i].real();
      double y = z[i].imag();
      cdouble wt = 0.0;
      cdouble lnx = 0.5 * log((x_last * x_last  + y * y)/ x_last / x_last) ;
      cdouble atanx = atan(y/x_last);
      if (fabs(y) > 1e-6) {
          wt = 1 * q_last * complexI / y * (lnx - complexI * atanx);
      } else {
   	  wt = 1 * q_last  / x_last; 
      }
      *(double *)PyArray_GETPTR2(arr_H_re, i, Nq - 1) += wt.real();
      *(double *)PyArray_GETPTR2(arr_H_im, i, Nq - 1) += wt.imag();
      //fprintf (stderr, "corr %g %g\n", wt.real(), wt.imag());

      lnx = 0.5 * log((x_first * x_first  + y * y)/ x_first / x_first) ;
      atanx = atan(y/x_first);
      if (fabs(y) > 1e-6) {
           wt = - (q_first) * complexI / y * (lnx - complexI * atanx);
      } else {
   	   wt = - (q_first) / x_first;  
      }
      *(double *)PyArray_GETPTR2(arr_H_re, i, 0) += wt.real();
      *(double *)PyArray_GETPTR2(arr_H_im, i, 0) += wt.imag();

  }
}



static quad_weights hilbert_weights = {
  hilbert_const,
  hilbert_linear,
  hilbert_quadratic
};

static quad_weights hilbert_z_weights = {
  hilbert_z_const,
  hilbert_z_linear,
  hilbert_z_quadratic
};



static
PyObject* meth_make_fourier_quad(PyObject *self, PyObject *args) {
  PyObject *ret_dict = NULL;
  PyObject *arg_q = NULL, *arg_x = NULL;
  PyArrayObject *arr_Fre = NULL, *arr_Fim = NULL,
                *arr_q = NULL, *arr_x = NULL;
  double  *q = NULL;
  cdouble *x = NULL;
  int ret;
  int Nq, Nx;
  npy_intp arr_dims[2] = {1, 1};
  double C = 1.0/2.0/M_PI;

  //fprintf (stderr, "enter the method\n");
  if (!(ret = PyArg_ParseTuple(args, "OO", &arg_q, &arg_x))) {
    fprintf (stderr, "Failure reading args");
    PyErr_SetString(error_obj, "Invalid arguments");
    return NULL;
  }
  arr_q = (PyArrayObject*)PyArray_FromAny(arg_q,
					  PyArray_DescrFromType(NPY_DOUBLE),
			                  0, 0, 0, 0);
  arr_x = (PyArrayObject*)PyArray_FromAny(arg_x,
					  PyArray_DescrFromType(NPY_DOUBLE),
			                  0, 0, 0, 0);
  //fprintf (stderr, "got arrays\n");
  Nq = PyArray_DIM(arr_q, 0);
  Nx = PyArray_DIM(arr_x, 0);
  q = PyMem_New(double, Nq);
  x = PyMem_New(cdouble, Nx);
  //fprintf (stderr, "made buffers Nq = %d Nx = %d\n", Nq, Nx);
  if ((q == NULL) || (x == NULL)) {
    goto fail;
  }
  //fprintf (stderr, "copy arg in\n");
  for (int j = 0; j < Nq; j++)
    q[j] = *(double *)PyArray_GETPTR1(arr_q, j);
  for (int i = 0; i < Nx; i++)
    x[i] = *(double *)PyArray_GETPTR1(arr_x, i);
  //fprintf (stderr, "decref\n");
  //fprintf (stderr, "do my thing\n");
  //fprintf (stderr, "make output array\n");
  arr_dims[0] = Nx;
  arr_dims[1] = Nq;
  arr_Fre = (PyArrayObject*)PyArray_SimpleNew(2, arr_dims, NPY_FLOAT64);
  arr_Fim = (PyArrayObject*)PyArray_SimpleNew(2, arr_dims, NPY_FLOAT64);
  make_poly_quad(arr_Fre, arr_Fim, x, q, Nx, Nq, &fourier_weights, NULL);
  for (int i = 0; i < Nx; i++) {
    for (int j = 0; j < Nq; j++) {
      *(double *)PyArray_GETPTR2(arr_Fre, i, j) *= C;
      *(double *)PyArray_GETPTR2(arr_Fim, i, j) *= C;
    }
  }
  //fprintf (stderr, "data copied out\n");
  ret_dict = PyDict_New();
  PyDict_SetItemString(ret_dict, "Fre", (PyObject *)arr_Fre);
  PyDict_SetItemString(ret_dict, "Fim", (PyObject *)arr_Fim);
  PyDict_SetItemString(ret_dict, "q",   (PyObject *)arr_q);
  PyDict_SetItemString(ret_dict, "x",   (PyObject *)arr_x);
  {
    PyObject *key, *value;
    Py_ssize_t pos = 0;
    while (PyDict_Next(ret_dict, &pos, &key, &value)) {
      Py_DECREF(value);
    }
  }
  //Py_DECREF(arr_x);
  //Py_DECREF(arr_q);
  //fprintf (stderr, "output dict made\n");
fail:
  //fprintf (stderr, "free q = %p\n", q);
  if (q) { PyMem_Free(q); q = NULL; }
  //fprintf (stderr, "free x = %p\n", x);
  if (x) { PyMem_Free(x); x = NULL; }
  //fprintf (stderr, "free F = %p\n", F);
  //if (F) { PyMem_Free(F); F = NULL; }
  //fprintf (stderr, "return\n");
  return ret_dict;
}

static
PyObject* meth_make_hilbert_quad(PyObject *self, PyObject *args) {
  PyObject *ret_dict = NULL;
  PyObject *arg_q = NULL;
  PyArrayObject *arr_H = NULL,
                *arr_q = NULL;
  double  *q = NULL;
  cdouble *q_z = NULL;
  int ret;
  int Nq;
  npy_intp arr_dims[2] = {1, 1};
  double C = 1.0 / M_PI;

  //fprintf (stderr, "enter the method\n");
  if (!(ret = PyArg_ParseTuple(args, "O", &arg_q))) {
    fprintf (stderr, "Failure reading args");
    PyErr_SetString(error_obj, "Invalid arguments");
    return NULL;
  }
  arr_q = (PyArrayObject*)PyArray_FromAny(arg_q,
					  PyArray_DescrFromType(NPY_DOUBLE),
			                  0, 0, 0, 0);
  //fprintf (stderr, "got arrays\n");
  Nq = PyArray_DIM(arr_q, 0);
  q = PyMem_New(double, Nq);
  q_z = PyMem_New(cdouble, Nq);
  //fprintf (stderr, "made buffers Nq = %d\n", Nq);
  if ((q == NULL) || (q_z == NULL)) {
    goto fail;
  }
  //fprintf (stderr, "copy arg in\n");
  for (int j = 0; j < Nq; j++) {
    q[j]   = *(double *)PyArray_GETPTR1(arr_q, j);
    q_z[j] = *(double *)PyArray_GETPTR1(arr_q, j);
  }
  //fprintf (stderr, "decref\n");
  //fprintf (stderr, "do my thing\n");
  arr_dims[0] = Nq;
  arr_dims[1] = Nq;
  //fprintf (stderr, "make output array\n");
  arr_H = (PyArrayObject*)PyArray_SimpleNew(2, arr_dims, NPY_FLOAT64);
  //fprintf (stderr, "do run");
  make_poly_quad(arr_H, NULL, q_z, q, Nq, Nq, &hilbert_weights, NULL);
  hilbert_inf_corr(arr_H, q, Nq);
  for (int i = 0; i < Nq; i ++) {
     double H_sum = 0.0;
     for (int j = 0.0; j < Nq; j++) {
       H_sum += *(double *)PyArray_GETPTR2(arr_H, i, j);   //[i * Nq + j];
     }
     *(double *)PyArray_GETPTR2(arr_H, i, i) -= H_sum;
     for (int j = 0; j < Nq; j++) {
       *(double *)PyArray_GETPTR2(arr_H, i, j) *= C;
     }
     //H[i * Nq + i] -= H_sum;
  }
  //
  // Symmetrisation
  //
  for (int i = 0; i < Nq/2; i++) {
    int i1 = Nq - 1 - i;
    for (int j = 0; j < Nq/2; j++) {
       int j1 = Nq - 1 - j;
       double *h1 = (double *)PyArray_GETPTR2(arr_H, i,  j);
       double *h2 = (double *)PyArray_GETPTR2(arr_H, i,  j1);
       double *h3 = (double *)PyArray_GETPTR2(arr_H, i1, j);
       double *h4 = (double *)PyArray_GETPTR2(arr_H, i1, j1);
       //double d_even = 1.0/16.0 * (*h1 + *h2 + *h3 + *h4);
       //double d_odd  = 1.0/16.0 * (*h1 - *h2 - *h3 + *h4);
       double d14 = 0.5 * (*h1 + *h4);
       double d23 = 0.5 * (*h2 + *h3);
       *h1 -= d14;
       *h2 -= d23;
       *h3 -= d23;
       *h4 -= d14;
    }
  }
  //for (int i = 0; i < Nq; i++) {
  //  for (int j = 0; j < Nq; j++) {
  //    *(double *)PyArray_GETPTR2(arr_H, i, j) = H[i * Nq + j].real() * C;
  //  }
  //}
  //fprintf (stderr, "data copied out\n");
  ret_dict = PyDict_New();
  PyDict_SetItemString(ret_dict, "H", (PyObject *)arr_H);
  PyDict_SetItemString(ret_dict, "q", (PyObject *)arr_q);
  {
    PyObject *key, *value;
    Py_ssize_t pos = 0;
    while (PyDict_Next(ret_dict, &pos, &key, &value)) {
      Py_DECREF(value);
    }
  }
  //Py_DECREF(arr_q);
  //fprintf (stderr, "output dict made\n");
fail:
  //fprintf (stderr, "free q = %p\n", q);
  if (q) { PyMem_Free(q); q = NULL; }
  // fprintf (stderr, "free H = %p\n", H);
  //if (H) { PyMem_Free(H); H = NULL; }
  //fprintf (stderr, "return\n");
  return ret_dict;
}

static
PyObject* meth_make_hilbert_quad_z(PyObject *self, PyObject *args) {
  PyObject *ret_dict = NULL;
  PyObject *arg_q = NULL, *arg_x= NULL, *arg_y = NULL;
  PyArrayObject *arr_H_re = NULL, *arr_H_im = NULL,
    *arr_x = NULL, *arr_y = NULL, *arr_q = NULL;
  double  *q = NULL;
  cdouble *z = NULL;
  int ret;
  int Nq, Nz, Ny;
  npy_intp arr_dims[2] = {1, 1};
  cdouble C = - 0.5 / M_PI / complexI;

  //fprintf (stderr, "enter the method\n");
  if (!(ret = PyArg_ParseTuple(args, "OOO", &arg_q, &arg_x, &arg_y))) {
    fprintf (stderr, "Failure reading args");
    PyErr_SetString(error_obj, "Invalid arguments");
    return NULL;
  }
  arr_q = (PyArrayObject*)PyArray_FromAny(arg_q,
					  PyArray_DescrFromType(NPY_DOUBLE),
			                  0, 0, 0, 0);
  arr_x = (PyArrayObject*)PyArray_FromAny(arg_x,
					  PyArray_DescrFromType(NPY_DOUBLE),
			                  0, 0, 0, 0);
  arr_y = (PyArrayObject*)PyArray_FromAny(arg_y,
					  PyArray_DescrFromType(NPY_DOUBLE),
			                  0, 0, 0, 0);
  //fprintf (stderr, "got arrays\n");
  Nq = PyArray_DIM(arr_q, 0);
  Nz = PyArray_DIM(arr_x, 0);
  Ny = PyArray_DIM(arr_y, 0);
  //fprintf (stderr, "arr dim %d %d %d\n", Nq, Nz, Ny); 
  if (Nz != Ny) {
    goto fail;
  }
  //fprintf (stderr, "alloc arrays\n");  
  q = PyMem_New(double, Nq);
  z = PyMem_New(cdouble, Nz);
  //fprintf (stderr, "made buffers Nq = %d\n", Nq);
  if ((q == NULL) || (z == NULL)) {
    goto fail;
  }
  //fprintf (stderr, "copy arg in\n");
  for (int j = 0; j < Nq; j++)
    q[j] = *(double *)PyArray_GETPTR1(arr_q, j);
  for (int i = 0; i < Nz; i++)
    z[i] = cdouble(*(double *)PyArray_GETPTR1(arr_x, i),
		   *(double *)PyArray_GETPTR1(arr_y, i));
  //fprintf (stderr, "decref\n");
  //fprintf (stderr, "do my thing\n");
  arr_dims[0] = Nz;
  arr_dims[1] = Nq;
  //fprintf (stderr, "make output array\n");
  arr_H_re = (PyArrayObject*)PyArray_SimpleNew(2, arr_dims, NPY_FLOAT64);
  arr_H_im = (PyArrayObject*)PyArray_SimpleNew(2, arr_dims, NPY_FLOAT64);
  //fprintf (stderr, "do run\n");
  make_poly_quad(arr_H_re, arr_H_im, z, q, Nz, Nq, &hilbert_z_weights, NULL);
  hilbert_inf_corr_z(arr_H_re, arr_H_im, q, z, Nq, Nz);
  //hilbert_inf_corr_z(arr_H_re, arr_H_im, q, z, Nq, Nz);
  for (int i = 0; i < Nz; i ++) {
  //   double H_sum = 0.0;
  //   for (int j = 0.0; j < Nq; j++) {
  //     H_sum += *(double *)PyArray_GETPTR2(arr_H, i, j);   //[i * Nq + j];
  //   }
  //   *(double *)PyArray_GETPTR2(arr_H, i, i) -= H_sum;
     for (int j = 0; j < Nq; j++) {
       cdouble oldval =  *(double *)PyArray_GETPTR2(arr_H_re, i, j)
                    + *(double *)PyArray_GETPTR2(arr_H_im, i, j) * complexI;
       cdouble newval = C * oldval;
       *(double *)PyArray_GETPTR2(arr_H_re, i, j) = newval.real();
       *(double *)PyArray_GETPTR2(arr_H_im, i, j) = newval.imag();
     }
  //   //H[i * Nq + i] -= H_sum;
  }
  //
  // Symmetrisation
  //
  //for (int i = 0; i < Nq/2; i++) {
  //  int i1 = Nq - 1 - i;
  //  for (int j = 0; j < Nq/2; j++) {
  //     int j1 = Nq - 1 - j;
  //     double *h1 = (double *)PyArray_GETPTR2(arr_H, i,  j);
  //     double *h2 = (double *)PyArray_GETPTR2(arr_H, i,  j1);
  //     double *h3 = (double *)PyArray_GETPTR2(arr_H, i1, j);
  //     double *h4 = (double *)PyArray_GETPTR2(arr_H, i1, j1);
  //     //double d_even = 1.0/16.0 * (*h1 + *h2 + *h3 + *h4);
  //     //double d_odd  = 1.0/16.0 * (*h1 - *h2 - *h3 + *h4);
  //     double d14 = 0.5 * (*h1 + *h4);
  //     double d23 = 0.5 * (*h2 + *h3);
  //     *h1 -= d14;
  //     *h2 -= d23;
  //     *h3 -= d23;
  //     *h4 -= d14;
  //  }
  //}
  //for (int i = 0; i < Nq; i++) {
  //  for (int j = 0; j < Nq; j++) {
  //    *(double *)PyArray_GETPTR2(arr_H, i, j) = H[i * Nq + j].real() * C;
  //  }
  //}
  //fprintf (stderr, "data copied out\n");
  ret_dict = PyDict_New();
  PyDict_SetItemString(ret_dict, "H_re", (PyObject *)arr_H_re);
  PyDict_SetItemString(ret_dict, "H_im", (PyObject *)arr_H_im);
  PyDict_SetItemString(ret_dict, "q", (PyObject *)arr_q);
  PyDict_SetItemString(ret_dict, "z_re", (PyObject *)arr_x);
  PyDict_SetItemString(ret_dict, "z_im", (PyObject *)arr_y);
  {
    PyObject *key, *value;
    Py_ssize_t pos = 0;
    while (PyDict_Next(ret_dict, &pos, &key, &value)) {
      Py_DECREF(value);
    }
  }
  //Py_DECREF(arr_q);
  fprintf (stderr, "output dict made\n");
fail:
  //fprintf (stderr, "free q = %p\n", q);
  if (q) { PyMem_Free(q); q = NULL; }
  if (z) { PyMem_Free(z); z = NULL; }
  // fprintf (stderr, "free H = %p\n", H);
  //if (H) { PyMem_Free(H); H = NULL; }
  //fprintf (stderr, "return\n");
  return ret_dict;
}


//static PyObject* meth_make_quad(PyObject *self, PyObject *args) {
//
//}

static struct PyMethodDef module_methods[] = {
    {"fourier_quad",
       meth_make_fourier_quad,
       METH_VARARGS, "Fourier quadrature"},
    {"hilbert_quad",
       meth_make_hilbert_quad,
       METH_VARARGS, "Hilbert quadrature"},
    {"hilbert_quad_z",
       meth_make_hilbert_quad_z,
       METH_VARARGS, "Hilbert quadrature for complex arg"},
    {NULL, NULL, 0, NULL},
};


extern "C" {
  static struct PyModuleDef makequad_def = {
	    PyModuleDef_HEAD_INIT,
	    "_makequad",     // name
	    "",              // doc
	    -1,              // size of per-interpreter state
	    module_methods
  };
  PyMODINIT_FUNC
  PyInit__makequad(void){
    PyObject *module, *module_dict;
    PyMethodDef *method_def; 
    module  =  PyModule_Create(&makequad_def); 
    error_obj = PyErr_NewException("_makequad.error", NULL, NULL);
    Py_XINCREF(error_obj);
    if (PyModule_AddObject(module, "error", error_obj) < 0) {
      Py_XDECREF(error_obj);
      Py_CLEAR(error_obj);
      Py_DECREF(module);
      return NULL; 
    }
    import_array(); 
    return module; 
  }
  void init_makequad() {
     PyObject *module, *module_dict;
     PyMethodDef *method_def;
     //fprintf(stderr,  "enter module init\n");
     //fprintf (stderr, "enter init\n");
     //module = Py_InitModule("_makequad", module_methods);
     //module_dict = PyModule_GetDict(module);
     //fprintf (stderr, "get dict\n");
     //error_obj = Py_BuildValue("s", "_makequad.error");
     //fprintf (stderr, "has err occurred?\n");
     //if (PyErr_Occurred()) {
     //   Py_FatalError ("cannot initialize the module _makequad");
     //}
     //import_array();
  }
}
