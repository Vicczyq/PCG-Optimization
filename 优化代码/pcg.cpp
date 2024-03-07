#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <crts.h>
#include "pcg.h"
typedef struct{
	int * row_off;
  int * cols;
  int rows;
  double * data;
	double * pre_mat_val;
	double * preD;
	double * x;
	double * Ax;
	double * r;
	double * z;
	double * source;
	double * residual;
	double normfactor;
	double tolerance;
	int maxIter;
	int *iter;
  double *init_residual;
  int now_time;
  int mesh_num;
  double *diag;
}Csr;

typedef struct{
  int *offset_1;
  int *offset_2;
  double *lower;
  double *upper;
  double *data;
  int faces;
  int *cols;
  int *lPtr;
  int *uPtr;
}pre_data;

typedef struct{
    int rows;
    int data_size;
    int *row_off;
    int *cols;
    double *data;
    double *diag;
}CsrDiaMatrix;
extern "C" void slave_pcg(Csr *para);
extern "C" void slave_pre(pre_data *para);
int now_time = 0;
int mesh_num = 1;
CsrDiaMatrix csrdia_matrix;
PCG pcg;
Precondition pre;
PCGReturn pcg_solve(const LduMatrix &ldu_matrix, double *source,double *x, int maxIter, double tolerance, double normfactor) {

	//计算公式  M-1Ax=M-1B
	//矩阵规模44181、58908和88362。

    int iter = 0;
    double init_residual;
          pcg.x = x;//解向量，表示线性方程组的解
          pcg.source = source;//源向量或右端向量，表示线性方程组的右端向量。

          csrdia_matrix.rows = ldu_matrix.cells;//行数
          csrdia_matrix.data_size = 2*ldu_matrix.faces;//计算csr数据长度，即矩阵大小
          csrdia_matrix.row_off = (int *)malloc((csrdia_matrix.rows + 1)*sizeof(int));//普通矩阵每行起始位置（i，0）之前有多少个非零元素的row_off
          csrdia_matrix.cols = (int *)malloc(csrdia_matrix.data_size*sizeof(int));//每个非零元素在普通矩阵中的列标（从0开始）
          csrdia_matrix.data = (double *)malloc(csrdia_matrix.data_size*sizeof(double));//非零元素取值
          csrdia_matrix.diag = ldu_matrix.diag;
          int row, col;
          int *offset_1 = (int *)malloc(csrdia_matrix.data_size*sizeof(int));
          int *offset_2 = (int *)malloc(csrdia_matrix.data_size*sizeof(int));
          
          int *tmp = (int *)malloc((csrdia_matrix.rows + 1)*sizeof(int));
          
          memset(csrdia_matrix.row_off,0,(csrdia_matrix.rows + 1)*sizeof(int));
  
          for(int i = 0; i < ldu_matrix.faces; i++){
              row	= ldu_matrix.uPtr[i];
              col = ldu_matrix.lPtr[i];
              csrdia_matrix.row_off[row+1]++;
              csrdia_matrix.row_off[col+1]++;
          }
          
          for(int i=0 ; i<ldu_matrix.cells; i++){
              csrdia_matrix.row_off[i+1] += csrdia_matrix.row_off[i];
          }
          
          memcpy(&tmp[0], &csrdia_matrix.row_off[0], (ldu_matrix.cells + 1)*sizeof(int));
          
          for(int i = 0; i < ldu_matrix.faces; i++ ){
              row = ldu_matrix.uPtr[i];
              col = ldu_matrix.lPtr[i];
              offset_1[i] = tmp[row]++;
              offset_2[i] = tmp[col]++;
          }
          
          pre_data para_1={
                            offset_1,
                            offset_2,
                            ldu_matrix.lower,
                            ldu_matrix.upper,
                            csrdia_matrix.data,
                            ldu_matrix.faces,
                            csrdia_matrix.cols,
                            ldu_matrix.lPtr,
                            ldu_matrix.uPtr,
          };
          Csr para = {
              csrdia_matrix.row_off,
              csrdia_matrix.cols,
              csrdia_matrix.rows,
              csrdia_matrix.data,
              pre.pre_mat_val,
              pre.preD,
              pcg.x,
              pcg.Ax,
              pcg.r, 
              pcg.z,
              pcg.source,
              &pcg.residual,
              normfactor,
              tolerance,
              maxIter,
              &iter,
              &init_residual,
              now_time,
              mesh_num,
              csrdia_matrix.diag
               };
            CRTS_init();
            athread_spawn(slave_pre,&para_1);
            for(int i=0;i*33 < ldu_matrix.faces;i++)
            {
                csrdia_matrix.data[offset_1[i]] = ldu_matrix.lower[i];
                csrdia_matrix.data[offset_2[i]] = ldu_matrix.upper[i];
            }
            for(int i=0;i*33<ldu_matrix.faces;i++)
            {
              csrdia_matrix.cols[offset_1[i]] = ldu_matrix.lPtr[i];
              csrdia_matrix.cols[offset_2[i]] = ldu_matrix.uPtr[i];
            }
            athread_join();
            athread_spawn(slave_pcg,&para);
            free(tmp);
            free(offset_1);
            free(offset_2);
            if(++now_time==200)
            {
              now_time=0;
              mesh_num++;
            } 
       	athread_join();
        athread_halt();
        
  INFO("PCG: init residual = %e, final residual = %e, iterations: %d\n", init_residual, pcg.residual, iter);
 
      free(csrdia_matrix.cols);
      free(csrdia_matrix.data);
      free(csrdia_matrix.row_off);
    PCGReturn pcg_return;
    pcg_return.residual = pcg.residual;
    pcg_return.iter = iter;
    return pcg_return;
}
