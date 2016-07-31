/*
 * Copyright (C) 2010 EigenSystems, LLC. All rights reserved.
 *
 * You may not use this file except in compliance with the License. A copy of the License may be
 * obtained by contacting EigenSystems, LLC.
 *
 * Without the prior written consent of EigenSystems, LLC, you may not:
 * - reproduce, distribute, or transfer the Software, or portions thereof, to any third party.
 * - sell, rent, lease, assign, or sublet the Software or portions thereof.
 * - grant rights to any other person.
 * - use the Software in violation of any United States or international law or regulation.
 *
 * Permission is granted to create a Larger Work by combining this code with other code not
 * governed by the terms of this license, provided that all copyright and proprietary notices
 * and logos within the Software files remain intact.
 *
 * You may make copies of this file for back-up purposes, provided that you reproduce its content
 * in its original form and with all proprietary notices on the back-up copy.
 *
 * Disclaimer Of Warranty
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESSED OR IMPLIED, INCLUDING,
 * BUT NOT LIMITED TO, WARRANTIES OF QUALITY, PERFORMANCE, NON-INFRINGEMENT, MERCHANTABILITY, OR
 * FITNESS FOR A PARTICULAR PURPOSE. FURTHER, EigenSystems, LLC DOES NOT WARRANT THAT THE SOFTWARE
 * OR ANY RELATED SERVICE WILL ALWAYS BE AVAILABLE.
 *
 * Limitations Of Liability
 *
 * YOU ASSUME ALL RISK ASSOCIATED WITH THE INSTALLATION AND USE OF THE SOFTWARE. IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS OF THE SOFTWARE BE LIABLE FOR CLAIMS, DAMAGES OR OTHER LIABILITY
 * ARISING FROM, OUT OF, OR IN CONNECTION WITH THE SOFTWARE. LICENSE HOLDERS ARE SOLELY RESPONSIBLE
 * FOR DETERMINING THE APPROPRIATENESS OF USE AND ASSUME ALL RISKS ASSOCIATED WITH ITS USE, INCLUDING
 * BUT NOT LIMITED TO THE RISKS OF PROGRAM ERRORS, DAMAGE TO EQUIPMENT, LOSS OF DATA OR SOFTWARE PROGRAMS,
 * OR UNAVAILABILITY OR INTERRUPTION OF OPERATIONS.
 */
#include "include/Describe.h"
#include "include/Variance.h"
/**
 * ============================================================================
 */
typedef PyObject * ( *unary_afp )( PyArrayObject *ary );
/**
 * Descriptive statistics
 */
template<typename T>
struct Describe {
	/**
	 * Apply operator
	 * @param ary - numpy ndarray
	 */
	PyObject * operator()( PyArrayObject *ary ) {
		npp::NDArray<T> array( ary );
		npp::Describe<T> func( array );
		int jN = ( array.nd() == 1 ) ? 1 : array.dim( 1 );

		tbb::parallel_reduce(tbb::blocked_range<int>( 0, jN ), func );
		return func.results();
	}
};
/**
 * Rolling window equal weighted variance of log returns
 */
template<typename T>
struct Variance_UCC {
	int		_window;
	/**
	 * Construct instance given window size
	 * @param window - window size
	 */
	Variance_UCC( int window )
		: _window( window )
	{}
	/**
	 * Apply operator
	 * @param ary - numpy ndarray
	 */
	PyObject * operator()( PyArrayObject *ary ) {
		npp::NDArray<T> array( ary );
		npp::Seq_Variance_UCC<T> seqFunc;
		npp::Variance<T, npp::Seq_Variance_UCC<T> > func(
				array,
				seqFunc,
				_window );
		tbb::parallel_for( iterationBounds( array, _window ), func );
		return func.results();
	}
};
/**
 * Rolling window equal weighted variance of log returns
 */
template<typename T>
struct Variance_EWA {
	double	_decay;
	int		_window;
	/**
	 * Construct instance given window size
	 * @param window - window size
	 */
	Variance_EWA( double decay, int window )
		: _decay( decay ), _window( window )
	{}
	/**
	 * Apply operator
	 * @param ary - numpy ndarray
	 */
	PyObject * operator()( PyArrayObject *ary ) {
		npp::NDArray<T> array( ary );
		npp::Seq_Variance_EWMA<T> seqFunc( _decay );
		npp::Variance<T, npp::Seq_Variance_EWMA<T> > func(
				array,
				seqFunc,
				_window );
		tbb::parallel_for( iterationBounds( array, _window ), func );
		return func.results();
	}
};
/**
 * Unary array function
 */
template<typename F_SHORT, typename F_INT, typename F_LONG, typename F_DOUBLE>
PyObject *unary_af( PyObject *ary, F_SHORT f_short, F_INT f_int, F_LONG f_long, F_DOUBLE f_double ) {
	try {
		PyArrayObject *array = ( PyArrayObject * ) PyArray_FROM_O( ary );
		switch( array->descr->type_num ) {
			case NPY_SHORT:
			case NPY_USHORT:
				return f_short( array );
			case NPY_UINT:
			case NPY_INT:
				return f_int( array );
			case NPY_LONG:
			case NPY_ULONG:
				return f_long( array );
			case NPY_FLOAT:
			case NPY_DOUBLE:
				return f_double( array );
			default:
				throw npp::PyException( PyExc_TypeError, "not a numeric array" );
		}
	} catch( npp::PyException & ) {
		return NULL;
	}
}
/**
 * Extension function: descriptive statistics of a 1-D or 2-D numpy array
 */
static PyObject *describe( PyObject *self, PyObject *args ) {
	PyObject	*ary;

	if ( PyArg_ParseTuple( args, "O", &ary ) )
		return unary_af(
				ary,
				Describe<short>(),
				Describe<int>(),
				Describe<long>(),
				Describe<double>() );
	return NULL;
}
/**
 * Extension function: Rolling window equal weighted variance of log returns
 */
static PyObject *variance_ucc( PyObject *self, PyObject *args ) {
	PyObject	*ary;
	int			window = 0;

	if ( PyArg_ParseTuple( args, "O|i", &ary, &window ) )
		return unary_af(
				ary,
				Variance_UCC<short>( window ),
				Variance_UCC<int>( window ),
				Variance_UCC<long>( window ),
				Variance_UCC<double>( window ) );
	return NULL;
}
/**
 * Extension function: Rolling window exponentially weighted moving variance of log returns
 */
static PyObject *variance_ewa( PyObject *self, PyObject *args ) {
	PyObject	*ary;
	double		decay;
	int			window = 0;

	if ( PyArg_ParseTuple( args, "Od|i", &ary, &decay, &window ) )
		return unary_af(
				ary,
				Variance_EWA<short>( decay, window ),
				Variance_EWA<int>( decay, window ),
				Variance_EWA<long>( decay, window ),
				Variance_EWA<double>( decay, window ) );
	return NULL;
}
/**
 * ============================================================================
 */
/**
 * Decay constant to half-life conversion
 */
static PyObject *half_life( PyObject *self, PyObject *args ) {
	double	decay;

	if ( PyArg_ParseTuple( args, "d", &decay ) )
		return Py_BuildValue( "d", npp::half_life( decay ) );
	return NULL;
}
/**
 * Half-life to decay constant conversion
 */
static PyObject *decay( PyObject *self, PyObject *args ) {
	double	half_life;

	if ( PyArg_ParseTuple( args, "d", &half_life ) )
		return Py_BuildValue( "d", npp::decay( half_life ) );
	return NULL;
}
/**
 * ============================================================================
 */

static PyMethodDef _methods[] = {
	{ "describe",		describe,		METH_VARARGS, "summary statistics" },
	{ "variance_ucc",	variance_ucc,	METH_VARARGS, "unweighted close-to-close variance" },
	{ "variance_ewa",	variance_ewa,	METH_VARARGS, "exponentially weighted moving average close-to-close variance" },
	{ "half_life",		half_life,		METH_VARARGS, "compute half life given decay factor" },
	{ "decay",			decay,			METH_VARARGS, "compute decay factor given half life" },

	{ NULL, NULL, 0, NULL }
};

extern "C" {
void initnpp( void ) {
	Py_InitModule( "npp", _methods );
    import_array();
}
}

