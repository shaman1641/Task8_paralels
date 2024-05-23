#include <iostream>
#include <boost/program_options.hpp>
#include <cmath>
#include <memory>
#include <algorithm>
#include <fstream>
#include <iomanip>
#include <chrono>
namespace opt = boost::program_options;

#include <cuda_runtime.h>
#include <cub/cub.cuh>

#define maximum_t 32

// cuda unique_ptr
template<typename T>
using cuda_unique_ptr = std::unique_ptr<T,std::function<void(T*)>>;

// new
template<typename T>
T* cuda_new(size_t size)
{
    T *d_ptr;
    cudaMalloc((void **)&d_ptr, sizeof(T) * size);
    return d_ptr;
}

// delete
template<typename T>
void cuda_delete(T *dev_ptr)
{
    cudaFree(dev_ptr);
}
cudaStream_t* cuda_new_stream()
{
    cudaStream_t* stream = new cudaStream_t;
    cudaStreamCreate(stream);
    return stream;
}

void cuda_delete_stream(cudaStream_t* stream)
{
    cudaStreamDestroy(*stream);
    delete stream;
}

cudaGraph_t* cuda_new_graph()
{
    cudaGraph_t* graph = new cudaGraph_t;
    return graph;
}

void cuda_delete_graph(cudaGraph_t* graph)
{
    cudaGraphDestroy(*graph);
    delete graph;
}

cudaGraphExec_t* cuda_new_graph_exec()
{
    cudaGraphExec_t* graphExec = new cudaGraphExec_t;
    return graphExec;
}

void cuda_delete_graph_exec(cudaGraphExec_t* graphExec)
{
    cudaGraphExecDestroy(*graphExec);
    delete graphExec;
}



void swapMatrices(double* &prevmatrix, double* &curmatrix) {
    double* temp = prevmatrix;
    prevmatrix = curmatrix;
    curmatrix = temp;
    
}

__global__ void computeOneStepOnGPU(double *A, double *B, int size) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    if (j == 0 || i == 0 || i >= size-1 || j >= size-1)
        return;

    B[i*size+j]  = 0.25 * (A[i*size+j+1] + A[i*size+j-1] + A[(i-1)*size+j] + A[(i+1)*size+j]);
}

__global__ void gpuMatrixSubtractionAndErr(double *A, double *B,double *C, int size) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    if (j == 0 || i == 0 || i >= size-1 || j >= size-1)
        return;
    C[i*size + j] = fabs(A[i*size+j] - B[i*size+j]);
}


double linearInterpolation(double x, double x1, double y1, double x2, double y2) {
    return y1 + ((x - x1) * (y2 - y1) / (x2 - x1));
}

void initMatrix(std::unique_ptr<double[]> &arr ,int N){
    for (size_t i = 0; i < N*N-1; i++){
            arr[i] = 0;
    }
    arr[0] = 10.0;
    arr[N-1] = 20.0;
    arr[(N-1)*N + (N-1)] = 30.0;
    arr[(N-1)*N] = 20.0;
    for (size_t i = 1; i < N-1; i++){
        arr[0*N+i] = linearInterpolation(i,0.0,arr[0],N-1,arr[N-1]);
        arr[i*N+0] = linearInterpolation(i,0.0,arr[0],N-1,arr[(N-1)*N]);
        arr[i*N+(N-1)] = linearInterpolation(i,0.0,arr[N-1],N-1,arr[(N-1)*N + (N-1)]);
        arr[(N-1)*N+i] = linearInterpolation(i,0.0,arr[(N-1)*N],N-1,arr[(N-1)*N + (N-1)]);
    }
}

int main(int argc, char const *argv[]) {
    cudaSetDevice(2);
     // парсим аргументы
    opt::options_description desc("опции");
    desc.add_options()
        ("accuracy",opt::value<double>()->default_value(1e-6),"точность")
        ("cellsCount",opt::value<int>()->default_value(256),"размер матрицы")
        ("iterCount",opt::value<int>()->default_value(1000000),"количество операций")
        ("help","помощь")
    ;

    opt::variables_map vm;

    opt::store(opt::parse_command_line(argc, argv, desc), vm);
    if (vm.count("help")) {
        std::cout << desc << "\n";
        return 1;
    }
    opt::notify(vm);

    
    // и это всё было только ради того чтобы спарсить аргументы.......

    int N = vm["cellsCount"].as<int>();
    double accuracy = vm["accuracy"].as<double>();
    int countIter = vm["iterCount"].as<int>();
    


    cuda_unique_ptr<cudaStream_t> stream(cuda_new_stream(),cuda_delete_stream);
    cuda_unique_ptr<cudaGraph_t>graph(cuda_new_graph(),cuda_delete_graph);
    cuda_unique_ptr<cudaGraphExec_t>g_instance(cuda_new_graph_exec(),cuda_delete_graph_exec);


    double error =1.0;
    int iter = 0;

    std::unique_ptr<double[]> A(new double[N*N]);
    std::unique_ptr<double[]> Anew(new double[N*N]);
    std::unique_ptr<double[]> B(new double[N*N]);

    initMatrix(std::ref(A),N);
    initMatrix(std::ref(Anew),N);
    
    double* curmatrix = A.get();
    double* prevmatrix = Anew.get();
    double* error_Matrix = B.get();

    cuda_unique_ptr<double> A_GPU_NEW_ptr(cuda_new<double>(N*N),cuda_delete<double>);
    cuda_unique_ptr<double> A_GPU_ptr(cuda_new<double>(N*N),cuda_delete<double>);
    cuda_unique_ptr<double> error_GPU_Matrix_ptr(cuda_new<double>(N*N),cuda_delete<double>);
    cuda_unique_ptr<double>error_GPU_ptr(cuda_new<double>(1),cuda_delete<double>);

    double* A_GPU_NEW = A_GPU_NEW_ptr.get();
    double* A_GPU = A_GPU_ptr.get();
    double* error_GPU_Matrix = error_GPU_Matrix_ptr.get();
    double* error_GPU = error_GPU_ptr.get();


    cudaMemcpy(A_GPU_NEW,curmatrix,N*N*sizeof(double),cudaMemcpyHostToDevice);
    cudaMemcpy(A_GPU,prevmatrix,N*N*sizeof(double),cudaMemcpyHostToDevice);
    cudaMemcpy(error_GPU_Matrix,error_Matrix,N*N*sizeof(double),cudaMemcpyHostToDevice);


    size_t tmp_size = 0;
    double* tmp = NULL; 
    cub::DeviceReduce::Max(tmp,tmp_size,A_GPU,error_GPU,N*N);
    cuda_unique_ptr<double>tmp_ptr(cuda_new<double>(tmp_size),cuda_delete<double>);
    tmp = tmp_ptr.get();


    dim3 threads_in_block = dim3(32, 32);
    dim3 blocks_in_grid((N + threads_in_block.x - 1) / threads_in_block.x, (N + threads_in_block.y - 1) / threads_in_block.y);


   


    int update_matrix = 100;

    auto start = std::chrono::high_resolution_clock::now();

    cudaStreamBeginCapture(*stream, cudaStreamCaptureModeGlobal);


        for(int i = 0; i < update_matrix-1; i ++){
            computeOneStepOnGPU<<<blocks_in_grid, threads_in_block,0,*stream>>>(A_GPU,A_GPU_NEW,N);
            swapMatrices(A_GPU,A_GPU_NEW);
        }
        computeOneStepOnGPU<<<blocks_in_grid, threads_in_block,0, *stream>>>(A_GPU, A_GPU_NEW, N);
        gpuMatrixSubtractionAndErr<<<blocks_in_grid, threads_in_block,0, *stream>>>(A_GPU, A_GPU_NEW,error_GPU_Matrix, N);
        swapMatrices(A_GPU,A_GPU_NEW);
        // вычисление ошибки
        cub::DeviceReduce::Max(tmp, tmp_size, error_GPU_Matrix, error_GPU, (N*N), *stream);
    

    cudaStreamEndCapture(*stream, graph.get());


    cudaGraphInstantiate(g_instance.get(), *graph, NULL, NULL, 0);
    while(error > accuracy && iter < countIter){

        cudaGraphLaunch(*g_instance, *stream);
        cudaMemcpy(&error, error_GPU, sizeof(double), cudaMemcpyDeviceToHost);
        iter += update_matrix;
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;
    double time_s = double(std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count())/1000;
    std::cout<<"time: " << time_s<<" error: "<<error << " iterarion: " << iter<<std::endl;
    if(N < 15){
        cudaMemcpy(curmatrix, A_GPU, sizeof(double)*N*N, cudaMemcpyDeviceToHost);
        for(int i = 0; i < N; i++) {
            for(int j = 0; j < N; j++)
                std::cout << A[i*N + j] << ' ';
            std::cout << std::endl;
        }
    }

    return 0;
}
