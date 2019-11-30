#include <iostream>
#include <fstream>
#include <math.h>
#include <time.h>
#include <stdlib.h>
#include <string>
#include <vector>
using namespace std;

/* We may want these to be 16 bit floats, pointers to GPU memory, etc.,
 * so let's abstract this all away.  */
typedef double ScalarType;
typedef ScalarType* MatrixType;
typedef ScalarType* VectorType;


/* ======================================== Globals ================================== */

//Number of NN layers
const size_t L=4;

//Number of neurons +1 in each layer. n[0]-1 is the number of inputs, n[L-1]-1 is the number of outputs.
size_t n[]={11,11,11,2};

//Outputs. Each o[l] is a vector with n[l] rows and o[0]=1 (this allows us to calculate weights more easily)
//We store the outputs, which is sigma(activation), because this is the thing we do vector and matrix operations to.
VectorType o[L];

//The weights. each w[l] has n[l] rows and n[l+1] columns. The first column is always (1,0,0,0...).
//Note that w[0] propagates from the 0th layer to the 1st layer.
MatrixType w[L-1];
MatrixType DeltaW[L-1]; //capital because it's an upper case Delta!

//Used for backpropagation.
VectorType delta[L];

std::vector<ScalarType> allData;

/* ======================================== Helper Functions ================================== */

//Activation function. This cannot be changed without also changing the backward phase.
//( it's only for this sigma that o'(x)=o(x)*(1-o(x)) )
ScalarType sigma(ScalarType x){
    return 1.0/(1.+exp(-x));
}

MatrixType constructMatrix(size_t N, size_t M){
    return new ScalarType[N*M];
}

MatrixType constructVector(size_t N){
    return new ScalarType[N];
}

ScalarType unitRand(){
    return (1.0*rand())/RAND_MAX;
}

/* a is an nA by nB matrix
 * b is a nB by nC matrix
 * returns a pointer to a new nA by nC matrix. */
MatrixType matrixMatrixMultiply(MatrixType a, MatrixType b, size_t nA, size_t nB, size_t nC){
    MatrixType c=constructMatrix(nA,nC);
    for(size_t i=0;i<nA;i++){
        for(size_t j=0;j<nC;j++){
            c[i*nC+j]=0;
            for(size_t k=0;k<nB;k++){
                //c_ij=a_ik*b_kj (repeated index summation)
                c[i*nC+j]+=a[i*nB+k]*b[k*nC+j];
            }
        }
    }
    return c;
}

/* a is an nA by nB matrix
 * b is a nB by nC matrix
 * returns a pointer to a new nA by nC matrix. */
MatrixType matrixMatrixAdd(MatrixType a, MatrixType b, size_t nA, size_t nB){
    MatrixType c=constructMatrix(nA,nB);
    for(size_t i=0;i<nA;i++){
        for(size_t j=0;j<nB;j++){
            c[i*nB+j]=a[i*nB+j]+b[i*nB+j];
        }
    }
    return c;
}

/* a is an nA by nB matrix
 * b is a nB vector
 * returns a pointer to a new nA vector. */
VectorType matrixVectorMultiply(MatrixType a, VectorType v, size_t nA, size_t nB){
    MatrixType c=constructVector(nA);
    for(size_t i=0;i<nA;i++){
        c[i]=0;
        for(size_t j=0;j<nB;j++){
            //c_i=a_ij*b_j (repeated index summation)
            c[i]+=a[i*nB+j]*v[j];
        }
    }
    return c;
}
/* v is an nA vector
 * b is a nA by nB vector
 * returns a pointer to a new nB vector. */
VectorType vectorMatrixMultiply(VectorType v, MatrixType b, size_t nA, size_t nB){
    MatrixType c=constructVector(nB);
    for(size_t j=0;j<nB;j++){
        c[j]=0;
        for(size_t i=0;i<nA;i++){
            //c_j=v_i*b_ij (repeated index summation)
            c[j]+=v[i]*b[i*nB+j];
        }
    }
    return c;
}

/* ======================================== Neural Network Initialization ================================== */

void constructWeights(){
    //initialize the weights. each w[l] has n[l] rows and n[l+1] columns.
    for(size_t l=0;l<L-1;l++){
        w[l]=constructMatrix(n[l],n[l+1]);
        for(size_t i=0;i<n[l];i++){
            for(size_t j=0;j<n[l+1];j++){
                ScalarType &elem=w[l][i*n[l+1]+j];
                if(j==0 ) {
                    if(i==0)
                        elem=1;
                    else 
                        elem=0;
                } else {
                    elem=(unitRand()*2.0-1.0)/sqrt(n[l+1]-1.0);
                }
            }
        }
    }
}
void deleteWeights(){
    for(size_t l=0;l<L-1;l++){
        delete[] w[l];
    }
}

//Initializes o and delta
void constructOutputsAndDelta(){
    for(size_t l=0;l<L;l++){
        o[l]=constructVector(n[l]);
        delta[l]=constructVector(n[l]);
    }
}
//Initializes o and delta
void deleteOutputsAndDelta(){
    for(size_t l=0;l<L;l++){
        delete[] o[l];
        delete[] delta[l];
    }
}

void hitWithSigma(VectorType v, size_t N){
    for(size_t i=1;i<N;i++){
        v[i]=sigma(v[i]);
    }
    v[0]=1;
}
void zeroStepPhase(){
    for(size_t l=0;l<L-1;l++){
        for(size_t i=0;i<n[l];i++){
            for(size_t j=0;j<n[l+1];j++){
                DeltaW[l][i*n[l+1]+j]=0;
            }
        }
    }
}
void constructStepPhase(){
    for(size_t l=0;l<L-1;l++){
        DeltaW[l]=constructMatrix(n[l],n[l+1]);
    }
    zeroStepPhase();
}
void deleteStepPhase(){
    for(size_t l=0;l<L-1;l++){
        delete[] DeltaW[l];
    }
}


/* ======================================== Forward prop and backprop ================================== */


/* Preconditions: 
 *      w_ij^l has been initialized,
 *      o_0^0=1 
 *      o_i^0 has been initialized to the correct input data.
 * */
void forwardPhase(){
    for(size_t l=0;l<L-1;l++){
        VectorType tmp=o[l+1];
        o[l+1]=vectorMatrixMultiply(o[l],w[l],n[l],n[l+1]);
        hitWithSigma(o[l+1],n[l+1]);
        o[l+1][0]=1;
        delete[] tmp;
    }
}

/* Preconditions: 
 *      w_ij^l has been initialized,
 *      o_i^l has been filled via the forwardPhase function
 *      delta_i^l has been allocated and delta_i^{L-1} has been initialized to the correct values.
 * */
void backwardPhase(){
    for(int l=L-2;l>=0;l--){
        VectorType tmp=delta[l];
        delta[l]=matrixVectorMultiply(w[l],delta[l+1],n[l],n[l+1]);
        for(int i=0;i<n[l];i++){
            delta[l][i]*=o[l][i]*(1-o[l][i]);
        }
        delete[] tmp;
    }
}

/* Preconditions: 
 *      w_ij^l has been initialized,
 *      o_i^l has been filled via the forwardPhase function
 *      delta_i^l has been filled via the backwardPhase function
 * */
void stepPhaseAdd(){
    for(size_t l=0;l<L-1;l++){
        for(size_t i=0;i<n[l];i++){
            for(size_t j=0;j<n[l+1];j++){
                DeltaW[l][i*n[l+1]+j]+=delta[l+1][j]*o[l][i];
            }
        }
    }
}
//Add the values to the step phase.
void stepPhaseCommit(double alpha){
    for(size_t l=0;l<L-1;l++){
        for(size_t i=0;i<n[l];i++){
            for(size_t j=0;j<n[l+1];j++){
                w[l][i*n[l+1]+j]+=(-alpha)*DeltaW[l][i*n[l+1]+j];
            }
        }
    }
}

/* ============================= Data Loading + saving ========================== */

/*
ScalarType * loadData(std::string fname){

    
}*/


int loadPrimesCSV(){
    ifstream datafile("primes.csv");
    allData=vector<ScalarType>(1024*13);
    if(!datafile.is_open()){
        cout<<"Error: couldn't open primes.csv."<<endl;
        return 1;
    }
    
    char comma;
    size_t index=0;
    for (;;) {          /* loop continually */
        if(index>=1024*13)
            break;
        for(int i=0;i<12;i++){
            datafile >> allData[index]; index++;
            datafile >> comma;
        }
        datafile>>allData[index]; index++;

        datafile.ignore (128, '\n');
        if (datafile.fail() || datafile.eof())   
            break;

    }
    cout<<"Successfully read "<<index<<" entries from primes.csv."<<endl;
    datafile.close();
    return 0;
}


double naiveStep(double alpha){
    double error=0;
    zeroStepPhase();

    for(int nn=0;nn<1024;nn++){
        //Initialize the inputs
        for(size_t i=0;i<11;i++){
            o[0][i]=allData[nn*13+i];
        }
        forwardPhase();
        //cout<<"Output should be 1: "<<o[L-1][1]<<endl;

        error+=(o[L-1][1]-allData[nn*13+12])*(o[L-1][1]-allData[nn*13+12]);
        for(size_t j=0;j<2;j++){
            delta[L-1][j]=o[L-1][j]*(1-o[L-1][j])*(o[L-1][j]-allData[nn*13+11+j]);
        }

        backwardPhase();

        stepPhaseAdd();
    }

    //cout<<DeltaW[0][0]<<endl;
    stepPhaseCommit(alpha);
    return error;
}

double computeTotalError(){
    double error=0;
    for(int n=0;n<1024;n++){
        for(int i=0;i<11;i++){
            o[0][i]=allData[n*13+i];
        }
        forwardPhase();
        error+=(o[L-1][1]-allData[n*13+12])*(o[L-1][1]-allData[n*13+12]);
    }
    return error;
}
double clamp(double x,double min, double max){
    if(x<min)
        return min;
    if(x>max)
        return max;
    return x;
}

int main(){
    srand (time(NULL));

    constructWeights();
    constructOutputsAndDelta();
    constructStepPhase();


    if(loadPrimesCSV())
        return 1;


    double e=computeTotalError();
    //cout<<"Total error with random weights: "<<e<<endl;
    //cin.get();
    double error=naiveStep(0.02);
    double alpha=0.02;
    double max_rate=0.05;
    double min_rate=0.00001;
    double mult=1.05;
    int incrctr=0;
    for(int k=0;k<500000;k++){

        double newError=naiveStep(alpha);
        if(abs(newError-error)>10){
            alpha=clamp(alpha/(mult*mult*mult),min_rate,max_rate);
            incrctr=0;
        }
        else if(abs(newError-error)>1){
            alpha=clamp(alpha/(mult),min_rate,max_rate);
            incrctr=0;
        }
        else if(abs(newError-error)<0.1){
            incrctr+=1;
            if(incrctr>20){
                alpha=clamp(alpha*mult,min_rate,max_rate);
            }
        }
        if(k>180000)
            alpha=0.001;
            
        if(k%5000==0)
            cout<<"Total error after training for "<<k<<" steps: "<<(newError)<<", learning rate: "<<alpha<<endl;
        error=newError;
    }
    for(int l=0;l<L-1;l++){
        cout<<"w["<<l<<"] is a "<<n[l]<<" by "<<n[l+1]<<" matrix."<<endl;
        for(int i=0;i<n[l];i++){
            for(int j=0;j<n[l+1];j++){
                cout<<w[l][i*n[l+1]+j]<<" ";
            }
            cout<<endl;
        }
    }
    for(int nn=0;nn<1024;nn++){
        cout<<"NN output for number ";
        for(size_t i=0;i<11;i++){
            o[0][i]=allData[nn*13+i];
            if(i>0)
                cout<<o[0][i];
        }
        cout<<" is: ";
        forwardPhase();
        cout<<o[L-1][1]<<" when it should be "<<allData[nn*13+12]<<"."<<endl;
    }



    deleteStepPhase();
    deleteOutputsAndDelta();
    deleteWeights();
    return 0;

}

