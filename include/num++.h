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
#ifndef NUM_PLUS_PLUS_
#define NUM_PLUS_PLUS_
#include "Python.h"
#include "numpy/arrayobject.h"
#include <math.h>
#include <limits>
#include <iostream>
#include <string>
#include <sstream>
#include <vector>
#include <tbb/blocked_range.h>
#include <tbb/parallel_reduce.h>

#include <tbb/scalable_allocator.h>
#include <tbb/cache_aligned_allocator.h>
#include <tbb/task_scheduler_init.h>
#include <tbb/tick_count.h>
#include <tbb/atomic.h>
#include <tbb/spin_mutex.h>
#include <tbb/mutex.h>
#include <tbb/concurrent_queue.h>
#include <tbb/concurrent_vector.h>
#include <tbb/concurrent_hash_map.h>
#include <tbb/blocked_range2d.h>
#include <tbb/parallel_for.h>

#ifdef max
#undef max
#endif

#ifdef min
#undef min
#endif

#if defined(WIN32) || defined(WIN64)
#define	isnan	( _isnan )
#endif
namespace npp {
 /**
 * @mainpage
 * A C++ toolkit for extending and embedding numpy
 *
 * @defgroup core numpy API Wrappers
 * @defgroup stats statistical methods
 */

/** @ingroup core
 * \brief
 * A python exception. The general convention for a C/C++ method to report an error to python is by
 * set error indicators and returning NULL (@see http://docs.python.org/c-api/exceptions.html).
 *
 * A PyException can be used to wrap a python error; and calls PyErr_SetString when destroyed; this
 * allows most error handling to be reduced to the form
 *
 * <code>
 * 	try {
 * 		...
 * 		throw PyExeption( <exc>, <message> << 1 << " more bit of info";
 * 		...
 * 	} catch( PyException & exc ) {
 * 		return NULL;
 * 	}
 * </code>
 *
 * For a list of valid exception codes, @see http://docs.python.org/c-api/exceptions.html#standard-exceptions
 */
class PyException : public virtual std::exception {
protected:
	PyObject *_exc_type;
	std::string _what;

public:
	/**
	 * Construct given exception code
	 * @param exc_type - an exception code
	 */
	PyException( PyObject *exc_type ) throw()
		: _exc_type( exc_type ) {
	};
	/**
	 * Construct given exception code and message; for the common case where the message is
	 * a string constant
	 * @param exc_type - an exception code
	 * @param what - error message
	 */
	PyException( PyObject *exc_type, const char *what ) throw()
		: _exc_type( exc_type ), _what( what ) {
	};
	/**
	 * Set the error string upon destruction
	 */
	~PyException() throw() {
		if ( ! PyErr_Occurred() && _what.length() != 0 ) {
			PyErr_SetString( _exc_type, _what.c_str() );
		}
	}
	/**
	 * Convenience operator for augmenting the error message with additional information
	 * @param msg - an argument of any datatype admissible for the stream << operator
	 */
	template<typename T> PyException & operator <<( const T & msg ) {
		std::stringstream strStrm;
		strStrm << msg;
		_what += strStrm.str();
		return *this;
	}
	/**
	 * Faster specializaton of operator << for c-strings
	 * @param msg - a c-string
	 */
	template<const char *> PyException & operator <<( const char *msg ) {
		_what += msg; return *this;
	}
	/**
	 * Faster specializaton of operator << for STL strings
	 * @param msg - an STL string
	 */
	template<const std::string &> PyException & operator <<( const std::string & msg ) {
		_what += msg; return *this;
	}
};

/** @cond */
double const EPSILON = 1e-10;
template<typename T>	NPY_TYPES npy_type()					{ return NPY_NOTYPE; }
template<> 		inline	NPY_TYPES npy_type<bool>()				{ return NPY_BOOL; }
template<>		inline	NPY_TYPES npy_type<signed char>()		{ return NPY_BYTE; }
template<>		inline	NPY_TYPES npy_type<unsigned char>() 	{ return NPY_UBYTE; }
template<>		inline	NPY_TYPES npy_type<short>()				{ return NPY_SHORT; }
template<>		inline	NPY_TYPES npy_type<unsigned short>()	{ return NPY_USHORT; }
template<>		inline	NPY_TYPES npy_type<int>()				{ return NPY_INT; }
template<>		inline	NPY_TYPES npy_type<unsigned int>()		{ return NPY_UINT; }
template<>		inline	NPY_TYPES npy_type<long>()				{ return NPY_LONG; }
template<>		inline	NPY_TYPES npy_type<unsigned long>()		{ return NPY_ULONG; }
template<>		inline	NPY_TYPES npy_type<float>()				{ return NPY_FLOAT; }
template<>		inline	NPY_TYPES npy_type<double>()			{ return NPY_DOUBLE; }
/** @endcond */

/** @ingroup core
 * \brief
 * numpy ndarray wrapper
 *
 */
template<typename T, bool BOUNDSCHECK=false>
class NDArray {
protected:
	PyArrayObject *_array;

public:
	/**
	 * Bind to an existing numpy array
	 * @param array - existing numpy array
	 */
	NDArray( PyArrayObject *array ) : _array( array ) {
		if ( BOUNDSCHECK ) {
			if ( _array == NULL )
				throw PyException( PyExc_TypeError, "NULL array reference" );

			PyArray_Descr *type = PyArray_DescrFromType( npy_type<T>() );
			int exp_kind = type->type_num;
			int got_kind = array->descr->type_num;
			Py_DECREF( type );
			if ( exp_kind != got_kind )
				throw PyException( PyExc_TypeError, "expected dtype: " )
					<< exp_kind << " , got " << got_kind;

			if ( PyArray_NDIM( _array ) > 3 )
				throw PyException( PyExc_TypeError, "# dimensions > 3: " )
					<< PyArray_NDIM( _array );
		}
	}
	/**
	 * Create a 1-D numpy array
	 * @param m - array dimension
	 */
	NDArray( int m ) {
		int dims[] = { m };
		int ndim = sizeof( dims ) / sizeof( int );
		for ( int i = 0; i < ndim; i++ ) {
			if ( dims[ i ] <= 0 )
				throw PyException( PyExc_IndexError, "invalid dimension " ) << i + 1 << ": " << m;
		}
		_array = ( PyArrayObject * ) PyArray_FromDims( ndim, dims, npy_type<T>() );
	}
	/**
	 * Create a 2-D numpy array
	 * @param m - array dimension 1
	 * @param n - array dimension 2
	 */
	NDArray( int m, int n ) {
		int dims[] = { m, n };
		int ndim = sizeof( dims ) / sizeof( int );
		for ( int i = 0; i < ndim; i++ ) {
			if ( dims[ i ] <= 0 )
				throw PyException( PyExc_IndexError, "invalid dimension " ) << i + 1 << ": " << m;
		}
		_array = ( PyArrayObject * ) PyArray_FromDims( ndim, dims, npy_type<T>() );
	}
	/**
	 * Create a 3-D numpy array
	 * @param m - array dimension 1
	 * @param n - array dimension 2
	 */
	NDArray( int m, int n, int l ) {
		int dims[] = { m, n, l };
		int ndim = sizeof( dims ) / sizeof( int );
		for ( int i = 0; i < ndim; i++ ) {
			if ( dims[ i ] <= 0 )
				throw PyException( PyExc_IndexError, "invalid dimension " ) << i + 1 << ": " << m;
		}
		_array = ( PyArrayObject * ) PyArray_FromDims( ndim, dims, npy_type<T> );
	}
	/**
	 * @return number of dimensions
	 * @param k - index
	 */
	const int nd() const { return int( PyArray_NDIM( _array ) ); }
	/**
	 * @return i'th dimension
	 * @param i - dimension index
	 */
	const int dim( int i ) const {
		if ( i < nd() )
			return int( PyArray_DIM( _array, i ) );
		else
			return 1;
	}
	/**
	 * @return reference to value at given index for a 1-D array
	 * @param i - index
	 */
	T & operator()( int i ) {
		if ( BOUNDSCHECK ) {
			if ( nd() != 1 )
				throw PyException( PyExc_IndexError, "not 1-D" );
			if ( i >= dim( 0 ) )
				throw PyException( PyExc_IndexError ) << i << " > DIM[0]: " << dim( 0 );
		}
		return *( ( T *)PyArray_GETPTR1( _array, i ) );
	}
	/**
	 * @return reference to value at given index for a 2-D array
	 * @param i - dimension 0 index
	 * @param j - dimension 1 index
	 */
	T & operator()( int i, int j ) {
		if ( BOUNDSCHECK ) {
			if ( nd() != 2 )
				throw PyException( PyExc_IndexError, "not 2-D" );
			if ( i >= dim( 0 ) )
				throw PyException( PyExc_IndexError ) << i << " >= DIM[0]: " << dim( 0 );
			if ( j >= dim( 1 ) )
				throw PyException( PyExc_IndexError ) << j << " >= DIM[1]: " << dim( 1 );
		}
		return *( ( T *)PyArray_GETPTR2( _array, i, j ) );
	}
	/**
	 * @return reference to value at given index for a 3-D array
	 * @param i - dimension 0 index
	 * @param j - dimension 1 index
	 * @param k - dimension 2 index
	 */
	T & operator()( int i, int j, int k ) {
		if ( BOUNDSCHECK ) {
			if ( nd() != 3 )
				throw PyException( PyExc_IndexError, "not 3-D" );
			if ( i >= dim( 0 ) )
				throw PyException( PyExc_IndexError ) << i << " >= DIM[0]: " << dim( 0 );
			if ( j >= dim( 1 ) )
				throw PyException( PyExc_IndexError ) << j << " >= DIM[1]: " << dim( 1 );
			if ( k >= dim( 2 ) )
				throw PyException( PyExc_IndexError ) << k << " >= DIM[2]: " << dim( 1 );
		}
		return *( ( T *)PyArray_GETPTR3( _array, i, j, k ) );
	}
	/**
	 * @return borrowed reference underlying numpy array
	 */
	PyObject *borrow() const { return ( PyObject * )_array; }
};
}
#endif // NUM_PLUS_PLUS_
