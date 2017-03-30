// pezhman foroughi Nov. 9 2011
// C++ implementation of EstimateCorr
/***************************************************************************
 Copyright (c) 2012
 MUSiiC Laboratory
 Pezhman Foroughi, Emad M Boctor eboctor@jhmi.edu
 Johns Hopkins University

 For commercial use/licensing, please contact the authors
 Please see license.txt for further information.
 ***************************************************************************/
#include "TrUE_Corr.h"

double EstimateCorr(const double Tr1[16], const double Tr2[16],
		const int ROIrect[4], const double ScaleXY[2], const double effAx,
		const double Sig[3]) {
	double DD[3];
	GetDis(Tr1, Tr2, ROIrect, ScaleXY, DD);
	double Crr = exp(
			-DD[0] / (4.0 * Sig[0] * Sig[0])
					- 1.0 / (4.0 * Sig[1] * Sig[1])
							* pow(fabs(DD[1] - effAx), 3) / (DD[1] + 0.0001)
					- DD[2] / (4 * Sig[2] * Sig[2]));
	return Crr;
}

void GetDis(const double Tr1[16], const double Tr2[16], const int ROIrect[4],
		const double ScaleXY[2], double outputDD[3]) {
	double RelT[16];
	double invTr1[16];
	// RelT = inv(Tr1)*Tr2;
	if (!gluInvertMatrix(Tr1, invTr1)) {
		invTr1[0] = invTr1[5] = invTr1[10] = invTr1[15] = 1.0;
		invTr1[1] = invTr1[2] = invTr1[3] = 0;
		invTr1[4] = invTr1[6] = invTr1[7] = 0;
		invTr1[8] = invTr1[9] = invTr1[11] = 0;
		invTr1[12] = invTr1[13] = invTr1[14] = 0;
	}
	MulMatrices(invTr1, Tr2, RelT);

	double nn[3], tt, theta;
	//%% convert to axis-angle %%%
	tt = (RelT[0] + RelT[5] + RelT[10] - 1) / 2;
	if ((tt < 1) && (tt > -1)) // make sure acos returns real number
		theta = acos(tt);
	else
		theta = 0;

	nn[0] = RelT[9] - RelT[6];
	nn[1] = RelT[2] - RelT[8];
	nn[2] = RelT[4] - RelT[1];
	double norm_nn = sqrt(nn[0] * nn[0] + nn[1] * nn[1] + nn[2] * nn[2]);
	for (int i = 0; i < 3; i++)
		nn[i] *= theta / norm_nn;
	//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

	double X1 = ScaleXY[0] * ROIrect[0];
	double X2 = ScaleXY[0] * ROIrect[1];
	double Y1 = ScaleXY[1] * ROIrect[2];
	double Y2 = ScaleXY[1] * ROIrect[3];

	// find RMS^2 of AvgD
	// DD(1): -nn(3)*Y + RelT(1,4)
	if (fabs(nn[2]) > 0.000001)   // prevent round off error
		outputDD[0] =
				-1.0 / (3.0 * nn[2]) / (Y2 - Y1)
						* (pow(-nn[2] * Y2 + RelT[3], 3)
								- pow(-nn[2] * Y1 + RelT[3], 3));
	else
		outputDD[0] = RelT[3] * RelT[3];

	// DD(2): nn(3)*X + RelT(2,4)
	if (fabs(nn[2]) > 0.000001)   // prevent round off error
		outputDD[1] = 1.0 / (3.0 * nn[2]) / (X2 - X1)
				* (pow(nn[2] * X2 + RelT[7], 3) - pow(nn[2] * X1 + RelT[7], 3));
	else
		outputDD[1] = RelT[7] * RelT[7];

	// DD(3): nn(1)*Y - nn(2)*X + RelT(3,4)
	if (fabs(nn[0] * nn[1]) > 0.0000001)   // prevent round off error
		outputDD[2] = -1.0 / (12.0 * nn[0] * nn[1]) / (X2 - X1) / (Y2 - Y1)
				* (pow(nn[0] * Y2 - nn[1] * X2 + RelT[11], 4)
						- pow(nn[0] * Y2 - nn[1] * X1 + RelT[11], 4)
						- pow(nn[0] * Y1 - nn[1] * X2 + RelT[11], 4)
						+ pow(nn[0] * Y1 - nn[1] * X1 + RelT[11], 4));
	else if (fabs(nn[1]) > 0.000001)   // prevent round off error
		outputDD[2] = -1.0 / (3.0 * nn[1]) / (X2 - X1)
				* (pow(-nn[1] * X2 + RelT[11], 3)
						- pow(-nn[1] * X1 + RelT[11], 3));
	else if (fabs(nn[0]) > 0.000001)   // prevent round off error
		outputDD[2] =
				1.0 / (3.0 * nn[0]) / (Y2 - Y1)
						* (pow(nn[0] * Y2 + RelT[11], 3)
								- pow(nn[0] * Y1 + RelT[11], 3));
	else
		outputDD[2] = RelT[11] * RelT[11];

}

bool gluInvertMatrix(const double m[16], double invOut[16]) {
	double inv[16], det;
	int i;

	inv[0] = m[5] * m[10] * m[15] - m[5] * m[11] * m[14] - m[9] * m[6] * m[15]
			+ m[9] * m[7] * m[14] + m[13] * m[6] * m[11] - m[13] * m[7] * m[10];
	inv[4] = -m[4] * m[10] * m[15] + m[4] * m[11] * m[14] + m[8] * m[6] * m[15]
			- m[8] * m[7] * m[14] - m[12] * m[6] * m[11] + m[12] * m[7] * m[10];
	inv[8] = m[4] * m[9] * m[15] - m[4] * m[11] * m[13] - m[8] * m[5] * m[15]
			+ m[8] * m[7] * m[13] + m[12] * m[5] * m[11] - m[12] * m[7] * m[9];
	inv[12] = -m[4] * m[9] * m[14] + m[4] * m[10] * m[13] + m[8] * m[5] * m[14]
			- m[8] * m[6] * m[13] - m[12] * m[5] * m[10] + m[12] * m[6] * m[9];
	inv[1] = -m[1] * m[10] * m[15] + m[1] * m[11] * m[14] + m[9] * m[2] * m[15]
			- m[9] * m[3] * m[14] - m[13] * m[2] * m[11] + m[13] * m[3] * m[10];
	inv[5] = m[0] * m[10] * m[15] - m[0] * m[11] * m[14] - m[8] * m[2] * m[15]
			+ m[8] * m[3] * m[14] + m[12] * m[2] * m[11] - m[12] * m[3] * m[10];
	inv[9] = -m[0] * m[9] * m[15] + m[0] * m[11] * m[13] + m[8] * m[1] * m[15]
			- m[8] * m[3] * m[13] - m[12] * m[1] * m[11] + m[12] * m[3] * m[9];
	inv[13] = m[0] * m[9] * m[14] - m[0] * m[10] * m[13] - m[8] * m[1] * m[14]
			+ m[8] * m[2] * m[13] + m[12] * m[1] * m[10] - m[12] * m[2] * m[9];
	inv[2] = m[1] * m[6] * m[15] - m[1] * m[7] * m[14] - m[5] * m[2] * m[15]
			+ m[5] * m[3] * m[14] + m[13] * m[2] * m[7] - m[13] * m[3] * m[6];
	inv[6] = -m[0] * m[6] * m[15] + m[0] * m[7] * m[14] + m[4] * m[2] * m[15]
			- m[4] * m[3] * m[14] - m[12] * m[2] * m[7] + m[12] * m[3] * m[6];
	inv[10] = m[0] * m[5] * m[15] - m[0] * m[7] * m[13] - m[4] * m[1] * m[15]
			+ m[4] * m[3] * m[13] + m[12] * m[1] * m[7] - m[12] * m[3] * m[5];
	inv[14] = -m[0] * m[5] * m[14] + m[0] * m[6] * m[13] + m[4] * m[1] * m[14]
			- m[4] * m[2] * m[13] - m[12] * m[1] * m[6] + m[12] * m[2] * m[5];
	inv[3] = -m[1] * m[6] * m[11] + m[1] * m[7] * m[10] + m[5] * m[2] * m[11]
			- m[5] * m[3] * m[10] - m[9] * m[2] * m[7] + m[9] * m[3] * m[6];
	inv[7] = m[0] * m[6] * m[11] - m[0] * m[7] * m[10] - m[4] * m[2] * m[11]
			+ m[4] * m[3] * m[10] + m[8] * m[2] * m[7] - m[8] * m[3] * m[6];
	inv[11] = -m[0] * m[5] * m[11] + m[0] * m[7] * m[9] + m[4] * m[1] * m[11]
			- m[4] * m[3] * m[9] - m[8] * m[1] * m[7] + m[8] * m[3] * m[5];
	inv[15] = m[0] * m[5] * m[10] - m[0] * m[6] * m[9] - m[4] * m[1] * m[10]
			+ m[4] * m[2] * m[9] + m[8] * m[1] * m[6] - m[8] * m[2] * m[5];

	det = m[0] * inv[0] + m[1] * inv[4] + m[2] * inv[8] + m[3] * inv[12];
	if (det == 0)
		return false;

	det = 1.0 / det;

	for (i = 0; i < 16; i++)
		invOut[i] = inv[i] * det;

	return true;
}

void MulMatrices(const double A[16], const double B[16], double AB[16]) {
	double sum;
	for (int i = 0; i < 4; i++)
		for (int j = 0; j < 4; j++) {
			sum = 0;
			for (int e = 0; e < 4; e++)
				sum += A[4 * i + e] * B[4 * e + j];
			AB[4 * i + j] = sum;
		}
}
