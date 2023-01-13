/******************************************/
/* tp2_poisson1D_direct.c                 */
/* This file contains the main function   */
/* to solve the Poisson 1D problem        */
/******************************************/
#include "../include/lib_poisson1D.h"
#include "atlas_headers.h"
# include<time.h>
int main(int argc,char *argv[])
/* ** argc: Nombre d'arguments */
/* ** argv: Valeur des arguments */
{
  int ierr;
  int jj;
  int nbpoints, la;
  int ku, kl, kv, lab;
  int *ipiv;
  int info;
  int NRHS;
  double T0, T1;
  double *RHS, *EX_SOL, *X;
  double **AAB;
  double *AB;

  double temp, relres;

  NRHS=1;
  nbpoints=10;
  la=nbpoints-2;
  T0=-5.0;
  T1=5.0;

  printf("--------- Poisson 1D ---------\n\n");
  RHS=(double *) malloc(sizeof(double)*la);
  EX_SOL=(double *) malloc(sizeof(double)*la);
  X=(double *) malloc(sizeof(double)*la);

  set_grid_points_1D(X, &la);
  set_dense_RHS_DBC_1D(RHS,&la,&T0,&T1);
  set_analytical_solution_DBC_1D(EX_SOL, X, &la, &T0, &T1);

  write_vec(RHS, &la, "DATA/DIRECT/RHS/RHS.dat");
  write_vec(EX_SOL, &la, "DATA/DIRECT/SOL/EX_SOL.dat");
  write_vec(X, &la, "DATA/DIRECT/X_grid.dat");

  kv=1;
  ku=1;
  kl=1;
  lab=kv+kl+ku+1;



  AB = (double *) malloc(sizeof(double)*lab*la);

clock_t start, end;
double execution_time;


// DGBMV and DGBTRF DGBTRS to find RHS
    set_GB_operator_colMajor_poisson1D(AB, &lab, &la, &kv);
    write_GB_operator_colMajor_poisson1D(AB, &lab, &la, "DATA/DIRECT/AB/AB.dat");

    // double* VECTOR_dgbm
    cblas_dgbmv(CblasColMajor,CblasConjTrans,la,la,kl,ku,1.0,AB+1,lab,EX_SOL,1,0.0,RHS,1);
    write_vec(RHS, &la, "DATA/DIRECT/RHS/RHS_dgbmv.dat");
    start = clock();

    printf("Solution with LAPACK\n");
    /* LU Factorization */
    info=0;
    ipiv = (int *) calloc(la, sizeof(int));
    LAPACK_dgbtrf(&la, &la, &kl, &ku, AB, &lab, ipiv, &info);
    LU_Facto( AB, &lab, &la, &kv);


  /* LU for tridiagonal matrix  (can replace dgbtrf_) */
    ierr = dgbtrftridiag(&la, &la, &kl, &ku, AB, &lab, ipiv, &info);
    write_GB_operator_colMajor_poisson1D(AB, &lab, &la, "DATA/DIRECT/AB/AB_dgbtrftridiag.dat");
    /*
      write_GB_operator_colMajor_poisson1D(AB, &lab, &la, "LU.dat");
    */
    /* Solution (Triangular) */
    if (info==0){
    //  ierr = dgbtrs_("N", &la, &kl, &ku, &NRHS, AB, &lab, ipiv, RHS, &la, &info);
        LAPACK_dgbtrs("N", &la, &kl, &ku, &NRHS, AB, &lab, ipiv, RHS, &la, &info);
        write_vec(RHS, &la, "DATA/DIRECT/RHS/RHS_dgbtrs.dat");
        if (info!=0){printf("\n INFO DGBTRS = %d\n",info);}
    }else{
      printf("\n INFO = %d\n",info);
    }

end = clock();
execution_time = ((double)(end - start))/CLOCKS_PER_SEC;


printf("Execution time of DGBTRF and DGBTRS %f seconds",execution_time);

start = clock();

    // SOLUTION WITH DGBSV

    free(AB);
    free(RHS);
    AB = (double *) malloc(sizeof(double)*lab*la);
    RHS=(double *) malloc(sizeof(double)*la);

    set_GB_operator_colMajor_poisson1D(AB, &lab, &la, &kv);
    set_dense_RHS_DBC_1D(RHS,&la,&T0,&T1);


  /* It can also be solved with dgbsv (dgbtrf+dgbtrs) */
    LAPACK_dgbsv(&la, &kl, &ku, &NRHS, AB, &lab, ipiv, RHS, &la, &info);
    write_GB_operator_colMajor_poisson1D(AB, &lab, &la, "AB_dgbsv.dat");
    write_xy(RHS, X, &la, "DATA/DIRECT/RHS/RHS_dgbsv.dat");

end = clock();
execution_time = ((double)(end - start))/CLOCKS_PER_SEC;
printf("\nExecution time of DGBSV %f seconds",execution_time);


  /* Relative forward error */
  temp = cblas_ddot(la, RHS, 1, RHS,1);
  temp = sqrt(temp);
  cblas_daxpy(la, -1.0, RHS, 1, EX_SOL, 1);
  relres = cblas_ddot(la, EX_SOL, 1, EX_SOL,1);
  relres = sqrt(relres);
  relres = relres / temp;

  printf("\nThe relative forward error is relres = %e\n",relres);

  free(RHS);
  free(EX_SOL);
  free(X);
  free(AB);
  printf("\n\n--------- End -----------\n");
}
