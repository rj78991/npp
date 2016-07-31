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
#ifndef COVARIANCE_
#define COVARIANCE_
#include "include/num++.h"

using namespace std;
namespace npp {
/** @ingroup stats
 * \brief
 * State container for one pass computation computation of variance
 */
struct Covariance_Step {
	int n;
	double x_m1, y_m1, xy_m2;
	/**
	 * Initialize state
	 */
	Covariance_Step()
		: n( 0 ), x_m1( 0 ), y_m1( 0 ), xy_m2( 0 )
	{}
	/**
	 * Advance state by accumulating one data sample from each set
	 * @param x - accumulation value from x set
	 * @param y - accumulation value from y set
	 */
	void next( double x, double y ) {
		double n1 = n++;
		double dx = x - x_m1;
		double dy = y - y_m1;

		x_m1 += dx / n;
		y_m1 += dy / n;
		xy_m2 += ( dx * dy * self.n ) / n1;
	}
};
/** @ingroup stats
 * \brief
 * Function object for sequentially computing covariance of of close-to-close returns
 */
template<typename T>
class Seq_Covariance_UCC {
public:
	/**
	 * Compute covariance matrix of of unweighted close-to-close returns
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
			int k0,
			int kN,
			std::vector<Covariance_Step> & step,
			int i0 = 0,
			int N = -1
	) const {
		N = ( ( N == -1 ) ? array.dim( 0 ) : 1 + min( N, array.dim( 0 ) - 1 ) );
		for( int i = i0 + 1; i < i0 + N; i++ ) {
			for ( int j = j0; j < jN; j++ ) {
				for( int k = k0; k < kN; k++ ) {
					double x = array( i, j );
					double y = array( i, k );
					if ( ! isnan( x ) && x > EPSILON && ! isnan( y ) && y > EPSILON)
						step[ j, k ].next( x, y  );
					}
				}
			}
		}
	}
};
}
#endif // COVARIANCE_
