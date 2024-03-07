#include <slave.h>
#include "pcg_def.h"
#include <stdbool.h>
#include <math.h>
#include <crts.h>
#include <simd.h>
#define likely(x) __builtin_expect(!!(x), 1)   
#define unlikely(x) __builtin_expect(!!(x), 0)

__thread_local crts_rply_t DMARply = 0;
__thread_local unsigned int DMARplyCount = 0;
__thread_local crts_rply_t Rply = 0;
__thread_local unsigned int RplyCount = 0;
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

__thread_local_share double P_s[88362] __attribute__((aligned(64)));
__thread_local_share double Z_s[88362] __attribute__((aligned(64)));

__thread_local int Col_s[9800];
__thread_local double Y_s[9800];
__thread_local int ROW_OFF_s[1500] __attribute__((aligned(32)));
__thread_local double X_s[1500] __attribute__((aligned(64)));
__thread_local int id,len,saddr,faddr,start,num_s;

void slave_pcg(Csr * para)
{
	Csr para_s;
	CRTS_dma_iget(&para_s,para,sizeof(Csr),&DMARply);
  DMARplyCount++;
  CRTS_dma_wait_value(&DMARply,DMARplyCount);	 
  if(para_s.now_time==0)
  {  
     id = CRTS_tid;

 	   len = para_s.rows/64;
 	   saddr = len*id;
 	   if(id == 63)len+=para_s.rows%64;
     faddr = saddr + len;

  }   
      int iter = 0;
		  double sumprod=0,sumprod_old=0,beta=0,alpha=0;
     double vec[16];
      int init_j;  
      double Ax_s[len];
	   	double source[len];
      double preD[len];
      double Diag[len];
      double Buffer[len];//缓冲区
      int end,i,j;
	    double residual,temp;
      CRTS_dma_iget(ROW_OFF_s,para_s.row_off + saddr, len*sizeof(int)+4,&DMARply);
      DMARplyCount++;
      CRTS_dma_wait_value(&DMARply,DMARplyCount);	 
      start = ROW_OFF_s[0];
      num_s = ROW_OFF_s[len] - start;
      

CRTS_dma_iget(source,para_s.source + saddr, len*sizeof(double),&Rply);
CRTS_dma_iget(Y_s, para_s.data + start, num_s*sizeof(double),&DMARply);
CRTS_dma_iget(Col_s, para_s.cols + start, num_s*sizeof(int),&DMARply); 
CRTS_dma_get(Diag,para_s.diag + saddr, len * sizeof(double));
DMARplyCount+=2;
RplyCount++;

		for(i = 0; i < len ; i++)
		{
			preD[i] = 1.0/Diag[i];
		}
 
		j = start; 
		residual=0;
  
    for(i=0;i < len ; i++)
    {
      Ax_s[i]=Diag[i]*para_s.x[i+saddr];//注意对x优化，这里是正经的计算，其实时间就差0.1s左右
    }
CRTS_dma_wait_value(&DMARply,DMARplyCount);
CRTS_dma_iget(X_s,para_s.x+saddr,len*sizeof(double),&DMARply);
DMARplyCount++;
   
   	for(i= 0;i < len; i++)
	  {
        end = ROW_OFF_s[i+1];
    		temp=0;
    		for(; j < end; j++)
    		{
          temp += para_s.x[Col_s[j-start]] * Y_s[j-start];
    		}
    		Ax_s[i] += temp;
	  }
     
    CRTS_dma_wait_value(&Rply,RplyCount);
     
     for(i=0;i<len;i++)//更新r,空间原因,继续用source保存
     {
       source[i]-=Ax_s[i];
     }

     for(i=0;i<len;i++)//计算初始残差
     {
       residual+=fabs(source[i]);
     }
 
		athread_redurt(&residual,&residual,1,athread_double,OP_add,&temp,1);  
    *para_s.init_residual=residual;
		

		if(fabs(residual / para_s.normfactor) >para_s.tolerance)
		{
			do{
				if(unlikely(iter == 0))
				{
		    	   for(i = saddr; i < faddr; i++)
		  			{
		  			 	Z_s[i] = source[i-saddr] * preD[i-saddr];
		  			}
					j = start;
		 			sumprod = 0; 	    		      
					CRTS_ssync_array();
		                   	      		
					for(i = 0; i < len; i++)
					{
				      end =ROW_OFF_s[i+1];
        			temp = 0;           
              init_j = j;
              for(;j<end;j++)
              {
                vec[j-init_j] = Z_s[Col_s[j-start]];
              }
			        for(j=init_j; j < end; j++)
			        {
			            temp += vec[j-init_j] * Y_s[j-start]; 
			        }
              temp = (source[i] - temp) * preD[i];
              P_s[i+saddr] = temp;
              sumprod += temp * source[i];
              Ax_s[i] = temp * Diag[i];//计算Ax_s对角线的数据
				   }
                 j = start;
        
					athread_redurt(&sumprod,&sumprod,1,athread_double,OP_add,&temp,1);  
          CRTS_dma_wait_value(&DMARply,DMARplyCount);//确认X_s异步传输iget完成
				}
				else 
				{			
          sumprod_old = sumprod;	
   	      for(i = 0; i<len; i++)
				 	{
				 	 	Z_s[i+saddr] = source[i] * preD[i];
				 	}
     	    j = start;
				 	CRTS_ssync_array();
 				 	//j = start;
	         
				    for(i = 0; i < len; i++)
				    {
                end =ROW_OFF_s[i+1];
				        temp = 0;
                init_j = j;
                for(;j<end;j++)
                {
                  vec[j-init_j] = Z_s[Col_s[j-start]];
                }
				        for(j=init_j; j < end; j++)
				        {
				            temp += vec[j-init_j] * Y_s[j-start]; 
				        }
				        Ax_s[i] = temp;
				    }
    				sumprod=0;
		        for(i = 0; i< len; i++)
  					{
              Buffer[i] = (source[i] - Ax_s[i]) * preD[i];
              sumprod += Buffer[i] * source[i];
   	        }
       
					athread_redurt(&sumprod,&sumprod,1,athread_double,OP_add,&temp,1);
					beta = sumprod/sumprod_old;
					
					for(i=0; i < len; i++){
							P_s[i+saddr] = Buffer[i] + beta * P_s[i+saddr];
              Ax_s[i] = P_s[i+saddr] * Diag[i];//计算Ax_s对角线的数据
					    Z_s[i+saddr] = Buffer[i];
          }
          j = start;	
          CRTS_ssync_array();
          //j = start;		
				}	  	
		  		//=======================csr_spmv=========================//
         /* 
		 		  for(i= 0; i<len; i++)
		    	{
		       	  Ax_s[i] = P_s[i+saddr] * Diag[i];
		     	}
        */
    	 	
                      
				  for(i= 0;i < len; i++)
				  {
              end = ROW_OFF_s[i+1];
              temp=0;
              init_j = j;
              
              for(;j<end;j++)
              {
                vec[j-init_j] = P_s[Col_s[j-start]];
              }
              
    					for(j=init_j ; j < end; j++)
    					{
    						temp += vec[j-init_j] * Y_s[j-start];
    					}
					Ax_s[i] += temp;
				}
				//=============================================================//
				//=======================gsumprod==============================//
   	      alpha=0;
		    	for(i=0; i < len; i++)
			    {
				    alpha += Ax_s[i] * P_s[i+saddr];
				  }
           residual=0;  
		       athread_redurt(&alpha,&alpha,1,athread_double,OP_add,&temp,1);
			     alpha = sumprod / alpha;
        
          
			   //residual=0;
           for(i = 0; i < len; i++) {   
		           source[i] = source[i] - alpha * Ax_s[i];
		           residual += fabs(source[i]);
               X_s[i] += alpha * P_s[i+saddr];//计算X
		       }
			   	athread_redurt(&residual,&residual,1,athread_double,OP_add,&temp,1);

			}while(++iter < para_s.maxIter  && (residual / para_s.normfactor) >=para_s.tolerance);
			
		}
		*para_s.residual = residual;
		*para_s.iter = iter;
    
/*
    //检测从核处理的X的正误,去掉注释即可启用
    //在if里填写需要检测的参数即可，now_time:检测时间步,范围0~199,mesh_num:检测矩阵号，范围1~3,id:检测X对应从核id,范围0~63
      if(para_s.now_time==1&&para_s.mesh_num==1&&id==0)
      {
          for(i=0;i<len;i++)printf("slaveCore_id:%d,X[%d]=%.16lf\n",id,i+saddr,X_s[i]);//打印形式id:0,X[0]=413223681.3848046600000000
          DMARplyCount++;
    		  CRTS_dma_wait_value(&DMARply,DMARplyCount);
      }
*/
}


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

void slave_pre(pre_data * para)
{
   int faces = para->faces;
   int n = faces/66;
   int st = (CRTS_tid+2) * n;
   if(CRTS_tid==63)n+=faces%66;
   int ed = st + n;
   int i;
   for(i=st;i<ed;i++)
   { 
     para->data[para->offset_1[i]] = para->lower[i];
     para->data[para->offset_2[i]] = para->upper[i];     
   }
   for(i=st;i<ed;i++)
   {
     para->cols[para->offset_1[i]] = para->lPtr[i];
     para->cols[para->offset_2[i]] = para->uPtr[i];
   }
}