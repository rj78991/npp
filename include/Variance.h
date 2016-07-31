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
#ifndef VARIANCE_
#define VARIANCE_
#include "include/num++.h"

using namespace std;
namespace npp {
/** @ingroup stats
 * \brief
 * State container for one pass computation computation of variance
 */
struct Variance_Step {
	int n;
	double m1, m2;
	/**
	 * Initialize state
	 */
	Variance_Step()
		: n( 0 ), m1( 0 ), m2( 0 )
	{}
	/**
	 * Advance state by accumulating one data sample
	 * @param x - accumulation value
	 */
	void next( double x ) {
		double n1 = n++;
		double delta = x - m1;
		double delta_n = delta / n;

		m1 += delta_n;
		m2 += delta * delta_n * n1;
	}
};
/** @ingroup stats
 * \brief
 * Compute size of the result of a rolling window computation
 * @param array - array to iterate over
 * @param window - window size
 */
template<typename T>
int resultLength( const NDArray<T> & array, int window ) {
	if ( window == 0 )
		window = array.dim( 0 ) - 1;
	return array.dim( 0 ) - window;
}
/** @ingroup stats
 * \brief
 * Compute iteration bounds based on window
 * @param array - array to iterate over
 * @param window - window size
 */
template<typename T>
tbb::blocked_range2d<int> iterationBounds( const NDArray<T> & array, int window ) {
	int iN = resultLength( array, window );
	int jN = ( array.nd() == 1 ) ? 1 : array.dim( 1 );
	return tbb::blocked_range2d<int>( 0, iN, 0, jN );
}
/** @ingroup stats
 * \brief
 * Function object for sequentially computing unweighted variance of close-to-close returns
 */
template<typename T>
class Seq_Variance_UCC {
public:
	/**
	 * Compute variance of unweighted close-to-close returns
	 * @param array - input array of levels, one column of price history per asset
	 * @param j0 - starting column index
	 * @param jN - ending column index
	 * @param step - array of variance calculation iterator
	 * @param i0 - offset in array to start at
	 * @param N - number of samples to use
	 * @param M - offset into step vector
	 */
	void operator()(
			NDArray<T> & array,
			int j0,
			int jN,
			std::vector<Variance_Step> & step,
			int i0 = 0,
			int N = -1,
			int M = 0
	) const {
		N = ( ( N == -1 ) ? array.dim( 0 ) : 1 + min( N, array.dim( 0 ) - 1 ) );

		for( int i = i0 + 1; i < ( i0 + N ); i++ ) {
			for( int j = j0; j < jN; j++ ) {
				double x = double( array( i, j ) ) / double( array( i - 1, j ) );
				if ( ! isnan( x ) && x > EPSILON )
					step[ j - M ].next( log( x )  );
			}
		}
	}
};
/** @ingroup stats
 * Constants & conversions between decay and half life
 */
static const double LN_HALF = -0.69314718055994530941;
static const double E       =  2.71828182845904523536;
/**
 * Return half-life given decay
 * @param lambda - decay parameter
 */
double half_life( double decay ) {
	return LN_HALF / log( decay );
}
/**
 * Return decay given half-life
 * @param lambda - decay parameter
 */
static double decay( double half_life ) {
	return pow( E, ( LN_HALF / half_life ) );
}
/** @ingroup stats
 * \brief
 * Function object for variance of exponentially weighted moving average of returns
 */
template<typename T>
class Seq_Variance_EWMA {
protected:
	const double _decay;

public:
	/**
	 * Instantiate and initialize given lambda
	 * @param lambda - decay parameter
	 */
	Seq_Variance_EWMA( double decay )
		: _decay( decay )
	{}
	/**
	 * Compute variance of exponentially close-to-close returns
	 * @param array - input array of levels, one column of price history per asset
	 * @param j0 - starting column index
	 * @param jN - ending column index
	 * @param step - array of variance calculation iterator
	 * @param i0 - offset in array to start at
	 * @param N - number of samples to use
	 * @param M - offset into step vector
	 */
	void operator() (
			NDArray<T> & array,
			int j0,
			int jN,
			std::vector<Variance_Step> & step,
			int i0 = 0,
			int N = -1,
			int M = 0
	) const {
		N = ( ( N == -1 ) ? array.dim( 0 ) : 1 + min( N, array.dim( 0 ) - 1 ) );

		double weight = ( 1.0 - _decay ) * pow( _decay, N - 2 );
		for( int i = i0 + 1; i < ( i0 + N ); i++ ) {
			for( int j = j0; j < jN; j++ ) {
				double x = double( array( i, j ) ) / double( array( i - 1, j ) );
				if ( ! isnan( x ) && x > EPSILON )
					step[ j - M ].next( weight * log( x )  );
			}
			weight /= _decay;
		}
	}
};
/** @ingroup stats
 * \brief
 * Function object for variance of exponentially weighted moving average of returns
 * using an array of decay factors
 */
template<typename T>
class Seq_Variance_EWMA_v {
protected:
	static const double LN_HALF = -0.69314718055994530941;
	static const double E       =  2.71828182845904523536;
	NDArray<double> & _decay;

public:
	/**
	 * Instantiate and initialize given lambda
	 * @param lambda - decay parameter
	 */
	Seq_Variance_EWMA_v( const NDArray<double> & decay )
		: _decay( decay )
	{}
	/**
	 * Compute variance of exponentially close-to-close returns
	 * @param array - input array of levels, one column of price history per asset
	 * @param j0 - starting column index
	 * @param jN - ending column index
	 * @param step - array of variance calculation iterator
	 * @param i0 - offset in array to start at
	 * @param N - number of samples to use
	 * @param M - offset into step vector
	 */
	void operator() (
			NDArray<T> & array,
			int j0,
			int jN,
			std::vector<Variance_Step> & step,
			int i0 = 0,
			int N = -1,
			int M = 0
	) const {
		double weight[jN - j0 + 1];
		for( int j = j0; j < jN; j++ ) {
			double d = _decay( j - M );
			weight[ j - M ] = ( 1.0 - d ) * pow( d, N - 2 );
		}

		N = ( ( N == -1 ) ? array.dim( 0 ) : 1 + min( N, array.dim( 0 ) - 1 ) );
		for( int i = i0 + 1; i < ( i0 + N ); i++ ) {
			for( int j = j0; j < jN; j++ ) {
				double x = double( array( i, j ) ) / double( array( i - 1, j ) );
				if ( ! isnan( x ) && x > EPSILON )
					step[ j - M ].next( weight[ j - M ] * log( x )  );
			}
			weight /= _decay;
		}
	}
};
/** @ingroup stats
 * \brief
 * Function object for unweighted close-to-close variance of continuously compounded returns
 */
template<typename T, typename R>
class Variance {
	NDArray<T> & _array;
	R _seqCalc;
	int _window;
	PyObject *_result;

public:
	/**
	 * Instantiate and initialize
	 * @param array - input array of levels, one column of price history per asset
	 * @param returnCalc - function object to compute variance of returns
	 * @param window - moving window size
	 */
	Variance( NDArray<T> & array, R returnCalc, int window = 0 )
		: _array( array ),
		  _seqCalc( returnCalc ),
		  _window( window ) {

		if ( _array.nd() > 2 )
			throw PyException( PyExc_TypeError, "# dimensions > 2: " )
				<< _array.nd();
		if ( _window == 0 )
			_window = _array.dim( 0 );
		if ( _window < 2 || window > _array.dim( 0 ) )
			throw PyException( PyExc_IndexError, "invalid window: " )
				<< _window << ", array length: " << _array.dim( 0 );
		if ( array.nd() == 1 )
			_result = NDArray<double>( resultLength( array, window ) ).borrow();
		else
			_result = NDArray<double>( resultLength( array, window ), array.dim( 1 ) ).borrow();
	}
	/**
	 * Apply operator
	 * @param r - blocked range to operate upon
	 */
	void operator()( const tbb::blocked_range2d<int> & r ) const {
		int i0 = r.rows().begin(), iN = r.rows().end();
		int j0 = r.cols().begin(), jN = r.cols().end();
		NDArray<double> result( ( PyArrayObject *) _result );

		for( int i = i0; i < iN && iN <= _array.dim( 0 ) ; i++ ) {
			std::vector<Variance_Step> step( jN - j0 );
			_seqCalc( _array, j0, jN, step, i, _window, j0 );
			for( int j = j0; j < jN; j++ )
				result( i, j ) = step[ j - j0 ].m2 / ( step[ j - j0 ].n - 1 );
		}
	}
	/**
	 * Return final results as a 1-D numpy ndarray of variances, one element
	 * corresponding to each input column:
	 * If the input was a vector, the list contains a single element
	 */
	PyObject *results() const { return _result; }
};
}
#endif // VARIANCE_
