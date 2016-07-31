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
#ifndef DESCRIBE_
#define DESCRIBE_
#include "include/num++.h"

namespace npp {
/** @ingroup stats
 * \brief
 * State container for one pass computation of descriptive statics
 */
template<typename T>
class Describe_Step {
public:
	int n;
	T lo, hi;
	double m1, m2, m3, m4;
	/**
	 * Initialize state
	 */
	Describe_Step()
		: n( 0 ),
		  lo( std::numeric_limits<T>::max() ),
		  hi( std::numeric_limits<T>::min() ),
		  m1( 0 ),
		  m2( 0 ),
		  m3( 0 ),
		  m4( 0 )
	{}
	/**
	 * Advance state by accumulating one data sample
	 * @param x - accumulation value
	 */
	void next(T x) {
		if ( isnan( x ) )
			return;

		int n1;
		double delta, delta_n2, delta_n, term1;

		n1 = n++;

		lo = std::min<T>( x, lo );
		hi = std::max<T>( x, hi );

		delta = x - m1;
		delta_n = delta / n;
		term1 = delta * delta_n * n1;

		m1 += delta_n;
		delta_n2 = delta_n * delta_n;
		m4 += term1 * delta_n2 * ( n * n - T( 3 ) * n + T( 3 ) ) +
			  6 * delta_n2 * m2 - 4 * delta_n * m3;
		m3 += term1 * delta_n * ( n - T( 2 ) ) - T( 3 ) * delta_n * m2;
		m2 += term1;
	}
};
/** @ingroup stats
 * \brief
 * Function object for parallel, one pass computation of descriptive statics
 * of a 1- or 2-D ndarray of data. Each column of a 2-D ndarray yields a set
 * of descriptive measures (min, max and four moments).
 */
template<typename T>
class Describe {
protected:
	NDArray<T> & _array;
	std::vector<Describe_Step<T> > _step;

public:
	/**
	 * Instantiate and initialize
	 * @param array - input array
	 */
	Describe( NDArray<T> & array )
		: _array( array ) {

		if ( _array.nd() == 1 )
			new ( &_step ) std::vector<Describe_Step<T> >( 1 );
		else if ( _array.nd() == 2 )
			new ( & _step) std::vector<Describe_Step<T> >( _array.dim( 1 ) );
		else
			throw PyException( PyExc_TypeError, "# dimensions > 2: " )
				<< _array.nd();
	}
	/**
	 * Instantiate and initialize split partition
	 * @param f - function instance
	 * @param mark - placeholder type to distinguish from copy constructor
	 */
	Describe( Describe & f, tbb::split )
		: _array( f._array ), _step( f._step )
	{}
	/**
	 * Reduction operator
	 * @param r - 2-D blocked range to operate upon
	 */
	void operator()( const tbb::blocked_range<int> & r ) {
		int j0 = r.begin(), jN = r.end();

		for( int i = 0; i < _array.dim( 0 ); i++ ) {
			for( int j = j0; j < jN; j++ ) {
				_step[ j ].next( _array( i, j ) );
			}
		}
	}
	/**
	 * Combine computed results from another data partition
	 * @param f - function object with results
	 */
	void join( const Describe & f ) {
		for( size_t i = 0; i < _step.size(); i++ ) {
			if ( _step[ i ].n == 0 )
				_step[ i ] = f._step[ i ];
		}
	}
	/**
	 * Return final results as a list of
	 * 	(count, min, max, mu, variance, skewness, kurtosis)
	 * tuples, one for each column vector of the input array.
	 * If the input was a vector, the list contains a single tuple
	 */
	PyObject *results() const {
		NDArray<double> result( _step.size(), 7 );
		for( size_t i = 0; i < _step.size(); i++ ) {
			result( i, 0 ) = double( _step[ i ].n );
			result( i, 1 ) = _step[ i ].lo;
			result( i, 2 ) = _step[ i ].hi;
			result( i, 3 ) = _step[ i ].m1;
			if (  _step[ i ].n < 2 ) {
				result( i, 4 ) = 0.0;
				result( i, 5 ) = 0.0;
				result( i, 6 ) = 0.0;
			} else {
				result( i, 4 ) = _step[ i ].m2 / ( _step[ i ].n - 1 );
				result( i, 5 ) = sqrt( _step[ i ].n ) * _step[ i ].m3 / pow( _step[ i ].m2, 1.5 );
				result( i, 6 ) = ( _step[ i ].n * _step[ i ].m4 ) / ( _step[ i ].m2 * _step[ i ].m2 ) - 3.0;
			}
		}
		return result.borrow();
	}
};
}
#endif // DESCRIBE_
