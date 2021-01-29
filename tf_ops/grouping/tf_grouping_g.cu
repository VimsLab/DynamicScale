#include <cstdio>
#include <omp.h>
#include <ctime>
#include <cstring> // memset
#include <cstdlib> // rand, RAND_MAX
#include <cmath> // sqrtf
#include <vector>
#include <algorithm>
#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cusolverDn.h>
#include <iostream>
#include <cstdlib>
#include <device_launch_parameters.h>
#include "helper_cuda.h"
#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <set>
#include <unistd.h>
#include <sys/time.h>
#include <map>
#include <assert.h>


int row = 0;
int col = 0;
using namespace std;
const int max_iter = 1000;
// test symmetric matrices


/* ---------------------------------------------------------------- */
//
// the following functions come from here:
//
// https://people.sc.fsu.edu/~jburkardt/cpp_src/jacobi_eigenvalue/jacobi_eigenvalue.cpp
//
// attributed to j. burkardt, FSU
// they are unmodified except to add __host__ __device__ decorations
//
//****************************************************************************80
__device__ void r8mat_diag_get_vector(int n, float a[], float v[])
{
  int i;

  for ( i = 0; i < n; i++ )
  {
    v[i] = a[i+i*n];
  }

  return;
}
//****************************************************************************80
__device__ void r8mat_identity(int n, float a[])
{
  int i;
  int j;
  int k;

  k = 0;
  for ( j = 0; j < n; j++ )
  {
    for ( i = 0; i < n; i++ )
    {
      if ( i == j )
      {
        a[k] = 1.0;
      }
      else
      {
        a[k] = 0.0;
      }
      k = k + 1;
    }
  }

  return;
}
//****************************************************************************80
__device__ void jacobi_eigenvalue(int n, float a[], int it_max, float v[], float d[], int &it_num, int &rot_num)
{
  float *bw;
  float c;
  float g;
  float gapq;
  float h;
  int i;
  int j;
  int k;
  int l;
  int m;
  int p;
  int q;
  float s;
  float t;
  float tau;
  float term;
  float termp;
  float termq;
  float theta;
  float thresh;
  float w;
  float *zw;

  r8mat_identity ( n, v );

  r8mat_diag_get_vector ( n, a, d );

  bw = new float[n];
  zw = new float[n];

  for ( i = 0; i < n; i++ )
  {
    bw[i] = d[i];
    zw[i] = 0.0;
  }
  it_num = 0;
  rot_num = 0;

  while ( it_num < it_max )
  {
    it_num = it_num + 1;
//
//  The convergence threshold is based on the size of the elements in
//  the strict upper triangle of the matrix.
//
    thresh = 0.0;
    for ( j = 0; j < n; j++ )
    {
      for ( i = 0; i < j; i++ )
      {
        thresh = thresh + a[i+j*n] * a[i+j*n];
      }
    }

    thresh = sqrt ( thresh ) / ( float ) ( 4 * n );

    if ( thresh == 0.0 )
    {
      break;
    }

    for ( p = 0; p < n; p++ )
    {
      for ( q = p + 1; q < n; q++ )
      {
        gapq = 10.0 * fabs ( a[p+q*n] );
        termp = gapq + fabs ( d[p] );
        termq = gapq + fabs ( d[q] );
//
//  Annihilate tiny offdiagonal elements.
//
        if ( 4 < it_num &&
             termp == fabs ( d[p] ) &&
             termq == fabs ( d[q] ) )
        {
          a[p+q*n] = 0.0;
        }
//
//  Otherwise, apply a rotation.
//
        else if ( thresh <= fabs ( a[p+q*n] ) )
        {
          h = d[q] - d[p];
          term = fabs ( h ) + gapq;

          if ( term == fabs ( h ) )
          {
            t = a[p+q*n] / h;
          }
          else
          {
            theta = 0.5 * h / a[p+q*n];
            t = 1.0 / ( fabs ( theta ) + sqrt ( 1.0 + theta * theta ) );
            if ( theta < 0.0 )
            {
              t = - t;
            }
          }
          c = 1.0 / sqrt ( 1.0 + t * t );
          s = t * c;
          tau = s / ( 1.0 + c );
          h = t * a[p+q*n];
//
//  Accumulate corrections to diagonal elements.
//
          zw[p] = zw[p] - h;                 
          zw[q] = zw[q] + h;
          d[p] = d[p] - h;
          d[q] = d[q] + h;

          a[p+q*n] = 0.0;
//
//  Rotate, using information from the upper triangle of A only.
//
          for ( j = 0; j < p; j++ )
          {
            g = a[j+p*n];
            h = a[j+q*n];
            a[j+p*n] = g - s * ( h + g * tau );
            a[j+q*n] = h + s * ( g - h * tau );
          }

          for ( j = p + 1; j < q; j++ )
          {
            g = a[p+j*n];
            h = a[j+q*n];
            a[p+j*n] = g - s * ( h + g * tau );
            a[j+q*n] = h + s * ( g - h * tau );
          }

          for ( j = q + 1; j < n; j++ )
          {
            g = a[p+j*n];
            h = a[q+j*n];
            a[p+j*n] = g - s * ( h + g * tau );
            a[q+j*n] = h + s * ( g - h * tau );
          }
//
//  Accumulate information in the eigenvector matrix.
//
          for ( j = 0; j < n; j++ )
          {
            g = v[j+p*n];
            h = v[j+q*n];
            v[j+p*n] = g - s * ( h + g * tau );
            v[j+q*n] = h + s * ( g - h * tau );
          }
          rot_num = rot_num + 1;
        }
      }
    }

    for ( i = 0; i < n; i++ )
    {
      bw[i] = bw[i] + zw[i];
      d[i] = bw[i];
      zw[i] = 0.0;
    }
  }
//
//  Restore upper triangle of input matrix.
//
  for ( j = 0; j < n; j++ )
  {
    for ( i = 0; i < j; i++ )
    {
      a[i+j*n] = a[j+i*n];
    }
  }
//
//  Ascending sort the eigenvalues and eigenvectors.
//
  for ( k = 0; k < n - 1; k++ )
  {
    m = k;
    for ( l = k + 1; l < n; l++ )
    {
      if ( d[l] < d[m] )
      {
        m = l;
      }
    }

    if ( m != k )
    {
      t    = d[m];
      d[m] = d[k];
      d[k] = t;
      for ( i = 0; i < n; i++ )
      {
        w        = v[i+m*n];
        v[i+m*n] = v[i+k*n];
        v[i+k*n] = w;
      }
    }
  }

  delete [] bw;
  delete [] zw;

  return;
}

void initialize_matrix(int mat_id, int n, float *mat, float *v){

  for (int i = 0; i < n*n; i++) *(v+(mat_id*n*n)+i) = mat[i];
}

// end of FSU code
/* ---------------------------------------------------------------- */

__global__ void je(int num_matr, int n, float *a, int it_max, float *v, float *d){

  int idx = threadIdx.x+blockDim.x*blockIdx.x;
  int it_num;
  int rot_num;
  if (idx < num_matr){
    jacobi_eigenvalue(n, a+(idx*n*n), it_max, v+(idx*n*n), d+(idx*n), it_num, rot_num);
  }
}

// input: radius (1), nsample (1), xyz1 (b,n,3), xyz2 (b,m,3)
// output: idx (b,m,nsample), pts_cnt (b,m)
__global__ void query_ball_point_gpu(int b, int n, int m, float radius, int nsample, const float *xyz1, const float *xyz2, int *idx, int *pts_cnt) {
    int batch_index = blockIdx.x;
    xyz1 += n*3*batch_index;
    xyz2 += m*3*batch_index;
    idx += m*nsample*batch_index;
    pts_cnt += m*batch_index; // counting how many unique points selected in local region

    int index = threadIdx.x;
    int stride = blockDim.x;
    
    for (int j=index;j<m;j+=stride) {
        int cnt = 0;
        for (int k=0;k<n;++k) {
            if (cnt == nsample)
                break; // only pick the FIRST nsample points in the ball
            float x2=xyz2[j*3+0];
            float y2=xyz2[j*3+1];
            float z2=xyz2[j*3+2];
            float x1=xyz1[k*3+0];
            float y1=xyz1[k*3+1];
            float z1=xyz1[k*3+2];
    	    float d=max(sqrtf(((x2-x1)*(x2-x1))+((y2-y1)*(y2-y1))+((z2-z1)*(z2-z1))),1e-20f);
            if (d<=1) {
                if (cnt==0) { // set ALL indices to k, s.t. if there are less points in ball than nsample, we still have valid (repeating) indices
                    for (int l=0;l<nsample;++l)
                        idx[j*nsample+l] = j;
                }
                idx[j*nsample+cnt] = k;
                cnt+=1;
            }
        }
        pts_cnt[j] = cnt;
    }
}

// input: e1 (1), e2(1), e3(1), nsample (1), xyz1 (b,n,3), xyz2 (b,m,3)
// output: idx (b,m,nsample), pts_cnt (b,m), v(b,m,9), d(b,m,3), dist(b,m,nsample,1)
__global__ void query_ellipsoid_point_gpu(int b, int n, int m, int c, float e1, float e2, float e3, int nsample, const float *ingroup_xyz1, const float *ingroup_xyz2, int *ingroup_idx, int *ingroup_pts_cnt, float *ingroup_out, float *ingroup_cva, float *v, float *d, float *dist) {

  int batch_index = blockIdx.x;
  ingroup_xyz1 += n*3*batch_index;    //n is the 1024 points
  ingroup_xyz2 += m*3*batch_index;    //m are the points obtained by FPS 256 or 512
  ingroup_idx += m*nsample*batch_index;
  dist += m*nsample*batch_index;
  ingroup_pts_cnt += m*batch_index; // counting how many unique points selected in local region
  ingroup_out += m*nsample*c*batch_index;
  ingroup_cva+=m*c*c*batch_index;
  v+=m*c*c*batch_index;
  d+=m*c*batch_index;

  int index = threadIdx.x;
  int stride = blockDim.x;
  float cc = e3*e3;

  for (int j=index;j<m;j+=stride) {
    int cnt = 0;
    float xc=ingroup_xyz2[j*3+0];
    float yc=ingroup_xyz2[j*3+1];
    float zc=ingroup_xyz2[j*3+2];

    for (int k=0;k<n;++k) {
      if (cnt == nsample)
        break; // only pick the FIRST nsample points in the ellipsoid
      float x1=ingroup_xyz1[k*3+0];
      float y1=ingroup_xyz1[k*3+1];
      float z1=ingroup_xyz1[k*3+2];
      float spoint[3];

      spoint[0]=x1-xc;
      spoint[1]=y1-yc;
      spoint[2]=z1-zc;

      float xx = spoint[0];
      float yy = spoint[1];
      float zz = spoint[2];

      float d1=max(sqrtf((xx*xx/cc)+(yy*yy/cc)+(zz*zz/cc)),1e-20f);
      if (d1<=1.0) {
        if (cnt==0) { // set ALL indices to k, s.t. if there are less points in ellipsoid than nsample, we still have valid (repeating) indices
          for (int l=0;l<nsample;++l){
            ingroup_idx[j*nsample+l] = j;
            dist[j*nsample+l] = 0.0;
          }
        }
        ingroup_idx[j*nsample+cnt] = k;
        dist[j*nsample+cnt] = d1;
                cnt+=1;
      }
    }
    ingroup_pts_cnt[j] = cnt;
    //grouping points from the ball query.
    for (int k=0;k<nsample;++k) {
      int ii = ingroup_idx[j*nsample+k];
      for (int l=0;l<c;++l) {
        ingroup_out[j*nsample*c+k*c+l] = ingroup_xyz1[ii*c+l];
      }
    }
    //from the grouped points pick unique points
    float *Matrix=(float *)malloc(sizeof(float)*ingroup_pts_cnt[j]*c);
    float *tMatrix=(float *)malloc(sizeof(float)*ingroup_pts_cnt[j]*c);
        
    int flag=0;
    if(ingroup_pts_cnt[j]>=3){//&&ingroup_pts_cnt[j]<(nsample/2)) {
      for(int k=0;k<ingroup_pts_cnt[j];k++){
        int ii = ingroup_idx[j*nsample+k];
        for (int l=0;l<c;++l) {
          Matrix[l+3*k] = ingroup_xyz1[ii*c+l];
        }
        if(ingroup_xyz1[ii*c+0]==0 && ingroup_xyz1[ii*c+1]==0 && ingroup_xyz1[ii*c+2]==0){
          flag=1;
        }
      }

      if(flag!=1){
        //find mean of unique points
        float means[3];
        float d2;
        means[0]=means[1]=means[2]=0.0;               
        for (int up=0;up<ingroup_pts_cnt[j];up++){
          means[0]+=Matrix[up*c+0];
          means[1]+=Matrix[up*c+1];
          means[2]+=Matrix[up*c+2];                   
        }
        means[0]=means[0]/ingroup_pts_cnt[j];
        means[1]=means[1]/ingroup_pts_cnt[j];
        means[2]=means[2]/ingroup_pts_cnt[j];
        //distance between mean of unique points and the centroid point
        d2=sqrtf((means[0]-xc)*(means[0]-xc)+(means[1]-yc)*(means[1]-yc)+(means[2]-zc)*(means[2]-zc));

        //covariance adjustment
        if (d2 >= e1/4.0){
          //if more points are on one side of the centroid
          for(int up=0;up<ingroup_pts_cnt[j];up++){
            //subtract centroid from the points
            Matrix[c*up]=Matrix[c*up]-xc;
            Matrix[c*up+1]=Matrix[c*up+1]-yc;
            Matrix[c*up+2]=Matrix[c*up+2]-zc;
          }
        }else{
          for(int up=0;up<ingroup_pts_cnt[j];up++){
            // subtract mean from the points
            Matrix[c*up]=Matrix[c*up]-means[0];
            Matrix[c*up+1]=Matrix[c*up+1]-means[1];
            Matrix[c*up+2]=Matrix[c*up+2]-means[2];
          }
        }
        //transpose points matrix
        for(int tpt=0;tpt<c;tpt++){
          for(int tup=0;tup<ingroup_pts_cnt[j];tup++){
            tMatrix[tpt+c*tup]=Matrix[tpt+c*tup];
          }
        }
        //calculate covariance matrix
        float *covm=(float *)malloc(sizeof(float)*c*c);
        for(int t3=0;t3<c;t3++){
          for(int tn=0;tn<c;tn++){
            covm[tn+t3*c] = 0.0;
            for(int n3=0;n3<ingroup_pts_cnt[j];n3++){
              covm[tn+t3*c]+=tMatrix[t3+c*n3]*Matrix[tn+n3*c];
            }
            ingroup_cva[j*c*c+tn+t3*c]=covm[tn+t3*c]/(ingroup_pts_cnt[j]-1);
          }
        }
        free(covm);
      }
    }
    free(Matrix);
    free(tMatrix);
    int it_num;
    int rot_num;

    if((ingroup_pts_cnt[j]>=3)){ //Eigendecomposition 
      jacobi_eigenvalue(c, ingroup_cva+(j*c*c), max_iter, v+(j*c*c), d+(j*c), it_num, rot_num);

    int cnt = ingroup_pts_cnt[j];
    float newaa = 2*(*(d+(c*j)+2))/(e3*e3);

    while(newaa<0.7){
      newaa = 2*newaa;
    }
    if(newaa > 1.0){
      newaa = 0.9;
    }
    float newbb = newaa/2;
    float newcc = newbb;
    for (int k=0;k<n;++k) {
      if (cnt == nsample)
        break; // only pick the FIRST nsample points in the ellipsoid
      float x1=ingroup_xyz1[k*3+0];
      float y1=ingroup_xyz1[k*3+1];
      float z1=ingroup_xyz1[k*3+2];
      float spoint[3];
      float rspoint[3];
      //centering the neighborhood point at the centroid
      spoint[0]=x1-xc;
      spoint[1]=y1-yc;
      spoint[2]=z1-zc;
      //rotating input points
      rspoint[0] = ((*(v+(c*c*j)+6)))*spoint[0]+((*(v+(c*c*j)+7)))*spoint[1]+((*(v+(c*c*j)+8)))*spoint[2];
      rspoint[1] = ((*(v+(c*c*j)+3)))*spoint[0]+((*(v+(c*c*j)+4)))*spoint[1]+((*(v+(c*c*j)+5)))*spoint[2];
      rspoint[2] = ((*(v+(c*c*j)+0)))*spoint[0]+((*(v+(c*c*j)+1)))*spoint[1]+((*(v+(c*c*j)+2)))*spoint[2];

	    float xx = rspoint[0];
      float yy = rspoint[1];
      float zz = rspoint[2];
      //second querying - oriented and scaled ellipsoid
      float d3=max(sqrtf(((xx*xx)/(newaa*newaa))+((yy*yy)/(newbb*newbb))+((zz*zz)/(newcc*newcc))),1e-20f);
      //union of both query points
      if (d3<=1.0) {
        int kflag=0;
        for(int kk=0;kk<nsample;kk++){
          if (ingroup_idx[j*nsample+kk]==k){
            kflag=1;
            break;
          }
        }
        if (kflag!=1){
          ingroup_idx[j*nsample+cnt] = k;
          dist[j*nsample+cnt] = d3; //Euclidean distance between centroid and point in the neighborhood
          cnt+=1;
        }
      }
    }
    ingroup_pts_cnt[j] = cnt;
    }
  }
}

// input: points (b,n,c), idx (b,m,nsample)
// output: out (b,m,nsample,c)
__global__ void group_point_gpu(int b, int n, int c, int m, int nsample, const float *points, const int *idx, float *out) {
    int batch_index = blockIdx.x;
    points += n*c*batch_index;
    idx += m*nsample*batch_index;
    out += m*nsample*c*batch_index;

    int index = threadIdx.x;
    int stride = blockDim.x;
    
    for (int j=index;j<m;j+=stride) {
        for (int k=0;k<nsample;++k) {
            int ii = idx[j*nsample+k];
            for (int l=0;l<c;++l) {
                out[j*nsample*c+k*c+l] = points[ii*c+l];
            }
        }
    }
}

// input: grad_out (b,m,nsample,c), idx (b,m,nsample), 
// output: grad_points (b,n,c)
__global__ void group_point_grad_gpu(int b, int n, int c, int m, int nsample, const float *grad_out, const int *idx, float *grad_points) {
    int batch_index = blockIdx.x;
    idx += m*nsample*batch_index;
    grad_out += m*nsample*c*batch_index;
    grad_points += n*c*batch_index;

    int index = threadIdx.x;
    int stride = blockDim.x;

    for (int j=index;j<m;j+=stride) {
        for (int k=0;k<nsample;++k) {
            int ii = idx[j*nsample+k];
            for (int l=0;l<c;++l) {
                 atomicAdd(&grad_points[ii*c+l], grad_out[j*nsample*c+k*c+l]);
            }
        }
    }
}

// input: k (1), distance matrix dist (b,m,n)
// output: idx (b,m,n), dist_out (b,m,n)
// only the top k results within n are useful
__global__ void selection_sort_gpu(int b, int n, int m, int k, const float *dist, int *outi, float *out) {
    int batch_index = blockIdx.x;
    dist+=m*n*batch_index;
    outi+=m*n*batch_index;
    out+=m*n*batch_index;

    int index = threadIdx.x;
    int stride = blockDim.x;

    // copy from dist to dist_out
    for (int j=index;j<m;j+=stride) {
        for (int s=0;s<n;++s) {
            out[j*n+s] = dist[j*n+s];
            outi[j*n+s] = s;
        }
    }

    float *p_dist;
    for (int j=index;j<m;j+=stride) {
        p_dist = out+j*n;
        // selection sort for the first k elements
        for (int s=0;s<k;++s) {
            int min=s; 
            // find the min
            for (int t=s+1;t<n;++t) {
                if (p_dist[t]<p_dist[min]) {
                    min = t;
                }
            }
            // swap min-th and i-th element
            if (min!=s) {
                float tmp = p_dist[min];
                p_dist[min] = p_dist[s];
                p_dist[s] = tmp;
                int tmpi = outi[j*n+min];
                outi[j*n+min] = outi[j*n+s];
                outi[j*n+s] = tmpi;
            }
        }
    }
}

void queryBallPointLauncher(int b, int n, int m, float radius, int nsample, const float *xyz1, const float *xyz2, int *idx, int *pts_cnt) {
    query_ball_point_gpu<<<b,256>>>(b,n,m,radius,nsample,xyz1,xyz2,idx,pts_cnt);
    //cudaDeviceSynchronize();
}
void queryEllipsoidPointLauncher(int b, int n, int m, int c, float e1, float e2, float e3, int nsample, const float *ingroup_xyz1, const float *ingroup_xyz2, int *ingroup_idx, int *ingroup_pts_cnt, float *ingroup_out, float *ingroup_cva, float *v, float *d, float *dist) {
    query_ellipsoid_point_gpu<<<b,128>>>(b,n,m,c,e1,e2,e3,nsample,ingroup_xyz1,ingroup_xyz2,ingroup_idx,ingroup_pts_cnt,ingroup_out,ingroup_cva,v,d,dist);
    //cudaDeviceSynchronize();
}
void selectionSortLauncher(int b, int n, int m, int k, const float *dist, int *outi, float *out) {
    selection_sort_gpu<<<b,256>>>(b,n,m,k,dist,outi,out); 
    //cudaDeviceSynchronize();
}
void groupPointLauncher(int b, int n, int c, int m, int nsample, const float *points, const int *idx, float *out){
    group_point_gpu<<<b,256>>>(b,n,c,m,nsample,points,idx,out);
    //cudaDeviceSynchronize();
}
void groupPointGradLauncher(int b, int n, int c, int m, int nsample, const float *grad_out, const int *idx, float *grad_points){
    group_point_grad_gpu<<<b,256>>>(b,n,c,m,nsample,grad_out,idx,grad_points);
    //group_point_grad_gpu<<<1,1>>>(b,n,c,m,nsample,grad_out,idx,grad_points);
    //cudaDeviceSynchronize();
}

