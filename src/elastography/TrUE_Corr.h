/***************************************************************************
 Pezhman Foroughi 11/9/11
 ***************************************************************************/
/***************************************************************************
 Copyright (c) 2012
 MUSiiC Laboratory
 Pezhman Foroughi, Emad M Boctor eboctor@jhmi.edu
 Johns Hopkins University

 For commercial use/licensing, please contact the authors

 Please see license.txt for further information.

 ***************************************************************************/
#include <math.h>

double EstimateCorr(const double Tr1[16], const double Tr2[16],
        const int ROIrect[4], const double ScaleXY[2], const double effAx,
        const double Sig[2]);

void GetDis(const double Tr1[16], const double Tr2[16], const int ROIrect[4],
        const double ScaleXY[2], double outputDD[3]);

bool gluInvertMatrix(const double m[16], double invOut[16]);

void MulMatrices(const double A[16], const double B[16], double AB[16]);
