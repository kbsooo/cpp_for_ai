# GPU 프로그래밍 (CUDA)

## 서론: GPU 병렬 컴퓨팅의 필요성과 CPU vs GPU 아키텍처  
현대 AI/ML 응용에서는 행렬 연산, 이미지 처리 등 대규모 병렬 연산이 필수적입니다. CPU는 소수의 강력한 코어(통상 2–16개)를 갖추고 있으며 직렬 작업에 최적화되어 있습니다. 반면 GPU는 수천 개의 단순 코어를 탑재하여 **병렬 처리**에 특화된 장치입니다 ([Understanding Parallel Computing: GPUs vs CPUs Explained Simply with role of CUDA | DigitalOcean](https://www.digitalocean.com/community/tutorials/parallel-computing-gpu-vs-cpu-with-cuda#:~:text=Whereas%2C%20a%20graphics%20processing%20unit,designed%20to%20handle%20certain%20tasks)) ([CUDA Refresher: The CUDA Programming Model | NVIDIA Technical Blog](https://developer.nvidia.com/blog/cuda-refresher-cuda-programming-model/#:~:text=The%20host%20is%20the%20CPU,memory%20likewise%20called%20device%20memory)). 예를 들어, 최상위 소비자용 CPU가 16코어인 데 비해 최신 GPU(NVIDIA RTX 4090)의 CUDA 코어 수는 16,384개에 달하며, 엔비디아 H100은 18,432개 CUDA 코어를 갖추고 있습니다 ([Understanding Parallel Computing: GPUs vs CPUs Explained Simply with role of CUDA | DigitalOcean](https://www.digitalocean.com/community/tutorials/parallel-computing-gpu-vs-cpu-with-cuda#:~:text=Let%E2%80%99s%20take%20a%20look%20at,amounts%20of%20computations%20in%20parallel)). 각 GPU 코어는 CPU 코어보다 단순하지만 대규모 동시 연산을 필요로 하는 ML 학습·추론 작업에서는 GPU의 병렬 처리 능력이 압도적입니다 ([Understanding Parallel Computing: GPUs vs CPUs Explained Simply with role of CUDA | DigitalOcean](https://www.digitalocean.com/community/tutorials/parallel-computing-gpu-vs-cpu-with-cuda#:~:text=Whereas%2C%20a%20graphics%20processing%20unit,designed%20to%20handle%20certain%20tasks)) ([Understanding Parallel Computing: GPUs vs CPUs Explained Simply with role of CUDA | DigitalOcean](https://www.digitalocean.com/community/tutorials/parallel-computing-gpu-vs-cpu-with-cuda#:~:text=Let%E2%80%99s%20take%20a%20look%20at,amounts%20of%20computations%20in%20parallel)).  

GPU는 원래 그래픽 렌더링을 위해 개발되었으나, 현재는 일반 연산에도 광범위하게 활용됩니다. CPU와 GPU 간에는 **아키텍처 차이**가 있습니다. CPU는 적은 수의 고성능 코어와 복잡한 제어 유닛, 대용량 캐시를 갖고 복잡한 연산과 다중 작업을 처리합니다. GPU는 수백–수천 개의 단순 연산 코어(Streaming Multiprocessor)와 대역폭이 넓은 메모리를 갖추어, 동일한 연산을 여러 데이터에 동시에 적용하는 데 매우 효율적입니다 ([Understanding Parallel Computing: GPUs vs CPUs Explained Simply with role of CUDA | DigitalOcean](https://www.digitalocean.com/community/tutorials/parallel-computing-gpu-vs-cpu-with-cuda#:~:text=Whereas%2C%20a%20graphics%20processing%20unit,designed%20to%20handle%20certain%20tasks)) ([Understanding Parallel Computing: GPUs vs CPUs Explained Simply with role of CUDA | DigitalOcean](https://www.digitalocean.com/community/tutorials/parallel-computing-gpu-vs-cpu-with-cuda#:~:text=Let%E2%80%99s%20take%20a%20look%20at,amounts%20of%20computations%20in%20parallel)). 이러한 특성 덕분에 대규모 행렬 곱셈, 합성곱 연산 등 병렬화 가능한 작업을 CPU보다 훨씬 빠르게 처리할 수 있습니다.

**참고:** 병렬 컴퓨팅 작업은 데이터를 작은 조각으로 분할하여 동시에 처리하기 때문에, GPU의 많은 코어가 이 데이터를 병렬 처리할 때 큰 이점을 얻습니다. 

## CUDA 프로그래밍 모델  
CUDA는 NVIDIA GPU를 위한 병렬 프로그래밍 플랫폼으로, 호스트(CPU)와 디바이스(GPU) 간 계산을 분담할 수 있게 해줍니다 ([CUDA Refresher: The CUDA Programming Model | NVIDIA Technical Blog](https://developer.nvidia.com/blog/cuda-refresher-cuda-programming-model/#:~:text=The%20host%20is%20the%20CPU,memory%20likewise%20called%20device%20memory)). CUDA 프로그래밍의 핵심 개념은 **호스트와 디바이스**입니다. `host`는 시스템의 CPU와 메인 메모리, `device`는 GPU와 GPU 메모리를 의미합니다 ([CUDA Refresher: The CUDA Programming Model | NVIDIA Technical Blog](https://developer.nvidia.com/blog/cuda-refresher-cuda-programming-model/#:~:text=The%20host%20is%20the%20CPU,memory%20likewise%20called%20device%20memory)). CUDA 프로그램 실행을 위한 기본 절차는 다음과 같습니다 ([CUDA Refresher: The CUDA Programming Model | NVIDIA Technical Blog](https://developer.nvidia.com/blog/cuda-refresher-cuda-programming-model/#:~:text=To%20execute%20any%20CUDA%20program%2C,there%20are%20three%20main%20steps)): 
1. **데이터 전송(Host→Device):** 입력 데이터를 CPU 메모리에서 GPU 메모리로 복사합니다.  
2. **커널 실행(Kernel Launch):** GPU에서 실행할 커널 함수를 로드하여 실행합니다. 커널은 `__global__` 한정자를 붙인 함수로 정의하며, GPU 상에서 병렬로 실행됩니다 ([CUDA Refresher: The CUDA Programming Model | NVIDIA Technical Blog](https://developer.nvidia.com/blog/cuda-refresher-cuda-programming-model/#:~:text=Every%20CUDA%20kernel%20starts%20with,in%20variables)) ([CUDA Refresher: The CUDA Programming Model | NVIDIA Technical Blog](https://developer.nvidia.com/blog/cuda-refresher-cuda-programming-model/#:~:text=A%20group%20of%20threads%20is,Figure%202)).  
3. **결과 전송(Device→Host):** 계산 결과를 GPU 메모리에서 CPU 메모리로 복사합니다.  

CUDA 커널은 **병렬 쓰레드**를 사용하여 동시에 여러 데이터 요소를 처리합니다. **쓰레드 계층 구조**는 다음과 같습니다 ([CUDA Refresher: The CUDA Programming Model | NVIDIA Technical Blog](https://developer.nvidia.com/blog/cuda-refresher-cuda-programming-model/#:~:text=A%20group%20of%20threads%20is,Figure%202)) ([CUDA Refresher: The CUDA Programming Model | NVIDIA Technical Blog](https://developer.nvidia.com/blog/cuda-refresher-cuda-programming-model/#:~:text=CUDA%20defines%20built,built%203D%20variable%20called%20%60blockIdx)):  
- **그리드(Grid):** 커널 호출 시 지정하는 전체 쓰레드 집합입니다.  
- **블록(Block):** 그리드는 여러 개의 블록으로 나뉩니다. 각 블록은 GPU의 한 SM(Streaming Multiprocessor)에서 실행됩니다 ([CUDA Refresher: The CUDA Programming Model | NVIDIA Technical Blog](https://developer.nvidia.com/blog/cuda-refresher-cuda-programming-model/#:~:text=A%20group%20of%20threads%20is,Figure%202)).  
- **쓰레드(Thread):** 블록은 여러 개의 쓰레드로 구성되며, 각 쓰레드는 동일 커널의 병렬 인스턴스를 의미합니다.  

CUDA는 쓰레드와 블록을 식별하기 위한 내장 변수들을 제공합니다 ([CUDA Refresher: The CUDA Programming Model | NVIDIA Technical Blog](https://developer.nvidia.com/blog/cuda-refresher-cuda-programming-model/#:~:text=CUDA%20defines%20built,built%203D%20variable%20called%20%60blockIdx)):  
- `threadIdx.{x,y,z}`: 블록 내에서의 쓰레드 인덱스  
- `blockIdx.{x,y,z}`: 그리드에서 블록의 인덱스  
- `blockDim.{x,y,z}`: 블록 크기(축별 쓰레드 수)  
예를 들어 `int idx = blockIdx.x * blockDim.x + threadIdx.x;` 형태로 각 쓰레드에 고유 인덱스를 부여할 수 있습니다. 이처럼 3차원 인덱싱을 사용하여 행렬이나 3D 볼륨 데이터에 자연스럽게 접근할 수 있습니다 ([CUDA Refresher: The CUDA Programming Model | NVIDIA Technical Blog](https://developer.nvidia.com/blog/cuda-refresher-cuda-programming-model/#:~:text=CUDA%20defines%20built,built%203D%20variable%20called%20%60blockIdx)). 블록 당 최대 쓰레드 수는 아키텍처에 따라 제한(예: 1024개)되며, `__syncthreads()` 함수를 통해 동일 블록 내 쓰레드 동기화가 가능합니다 ([CUDA Refresher: The CUDA Programming Model | NVIDIA Technical Blog](https://developer.nvidia.com/blog/cuda-refresher-cuda-programming-model/#:~:text=Here%20are%20a%20few%20noticeable,points)).  

```cpp
// CUDA 커널 예제: 벡터 덧셈
__global__ void vectorAdd(const float* A, const float* B, float* C, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x; 
    if (i < N) {
        C[i] = A[i] + B[i];
    }
}
```
위 예제에서 커널 함수 `vectorAdd`는 인자로 받은 세 개의 벡터를 요소별로 더합니다. `__global__` 한정자를 통해 GPU에서 실행됨을 표시하고, 인자로 벡터와 크기를 받습니다. `blockIdx.x * blockDim.x + threadIdx.x` 계산으로 쓰레드별 고유 인덱스를 구해 해당 요소에 접근합니다. 

## GPU 메모리 구조  
GPU에는 여러 계층의 메모리 공간이 존재합니다. 주요 메모리 공간은 **글로벌(Global) 메모리, 공유(Shared) 메모리, 상수(Constant) 메모리, 레지스터(Registers)** 입니다. 

- **글로벌 메모리(Global memory):** GPU의 주 메모리 공간으로, CPU와 GPU 모두 접근 가능합니다 ([GPU Programming: Registers, Global, and Local Memory](https://carpentries-incubator.github.io/lesson-gpu-programming/global_local_memory.html#:~:text=Global%20memory%20can%20be%20considered,is%20also%20slower%20to%20access)). 크기가 크지만 접근 속도가 상대적으로 느립니다. 글로벌 메모리는 디바이스 메모리이며, 호스트가 할당한 데이터(예: `cudaMalloc`으로 할당된 메모리)는 기본적으로 글로벌 메모리에 위치합니다 ([GPU Programming: Registers, Global, and Local Memory](https://carpentries-incubator.github.io/lesson-gpu-programming/global_local_memory.html#:~:text=The%20vectors%20,are%20stored%20in%20global%20memory)). 모든 쓰레드에서 읽고 쓸 수 있지만, 쓰레드 간 일관성이 보장되지 않으므로 커널 종료 시점까지 변경 사항이 확정됩니다 ([GPU Programming: Registers, Global, and Local Memory](https://carpentries-incubator.github.io/lesson-gpu-programming/global_local_memory.html#:~:text=Global%20memory%20is%20accessible%20by,any%20value%20in%20global%20memory)).  
- **공유 메모리(Shared memory):** 각 블록마다 할당되는 온칩 메모리입니다. 블록 내 모든 쓰레드가 공유하여 빠르게 데이터를 주고받을 수 있으며, GPU 내 캐시처럼 사용됩니다. 공유 메모리는 글로벌 메모리보다 훨씬 빠르며(비교적 100배 낮은 대기 시간) ([Using Shared Memory in CUDA C/C++ | NVIDIA Technical Blog](https://developer.nvidia.com/blog/using-shared-memory-cuda-cc/#:~:text=Because%20it%20is%20on,Shared%20memory%20is%20allocated)), 스레드 협업(데이터 교환, 병렬 합성 등)에 활용됩니다. 다만, 뱅크 충돌(bank conflict)이 발생하면 성능이 저하될 수 있으므로 주의해야 합니다 ([Using Shared Memory in CUDA C/C++ | NVIDIA Technical Blog](https://developer.nvidia.com/blog/using-shared-memory-cuda-cc/#:~:text=Because%20it%20is%20on,Shared%20memory%20is%20allocated)).  
- **상수 메모리(Constant memory):** 읽기 전용 캐시 메모리로, 모든 블록의 쓰레드가 접근 가능합니다 ([GPU Programming: Constant Memory](https://carpentries-incubator.github.io/lesson-gpu-programming/constant_memory.html#:~:text=Constant%20memory%20is%20a%20read,this%20works%20with%20an%20example)). 크기는 작지만 캐시를 통해 모든 쓰레드에 데이터를 방송(broadcast)하므로 동일 상수값을 여러 쓰레드가 사용할 때 유리합니다. CUDA 코드에서는 `__constant__` 한정자로 선언하며 전역 변수로 사용합니다 ([GPU Programming: Constant Memory](https://carpentries-incubator.github.io/lesson-gpu-programming/constant_memory.html#:~:text=Constant%20memory%20is%20a%20read,this%20works%20with%20an%20example)).  
- **레지스터(Registers):** 각 쓰레드에 할당되는 가장 빠른 온칩 메모리입니다 ([GPU Programming: Registers, Global, and Local Memory](https://carpentries-incubator.github.io/lesson-gpu-programming/global_local_memory.html#:~:text=Registers%20are%20fast%20on,executed%20by%20the%20computing%20cores)). 연산 중 사용되는 스칼라 변수 등이 레지스터에 저장되며, 쓰레드 간 공유되지 않습니다 ([GPU Programming: Registers, Global, and Local Memory](https://carpentries-incubator.github.io/lesson-gpu-programming/global_local_memory.html#:~:text=Registers%20are%20local%20to%20a,the%20execution%20of%20a%20thread)). 레지스터는 런타임 동안만 데이터가 유지되고, 한정된 수의 레지스터를 효율적으로 사용하여 데이터 재사용을 최적화해야 성능에 도움이 됩니다 ([GPU Programming: Registers, Global, and Local Memory](https://carpentries-incubator.github.io/lesson-gpu-programming/global_local_memory.html#:~:text=Registers%20are%20the%20fastest%20memory,improve%20performance%20in%20future%20episodes)).  

메모리 접근 최적화 기법으로는 **메모리 공동접속(coalescing)**과 **공유 메모리 활용**이 있습니다. 전역 메모리는 디바이스 DRAM에 위치하므로, 인접 쓰레드가 인접 주소를 접근하도록 하면 메모리 트랜잭션이 병합(coalesced)되어 대역폭을 효율적으로 사용할 수 있습니다 ([How to Access Global Memory Efficiently in CUDA C/C++ Kernels | NVIDIA Technical Blog](https://developer.nvidia.com/blog/how-access-global-memory-efficiently-cuda-c-kernels/#:~:text=Grouping%20of%20threads%20into%20warps,0)). 한 워프(32개 쓰레드 단위) 내의 연속된 주소 접근이 이상적입니다. 또한 글로벌 메모리 접근 시 스트라이드가 크면 성능이 떨어지므로, 필요한 경우 **공유 메모리**를 이용해 데이터를 미리 로드하고 인접 접근 패턴으로 변경하면 성능이 향상됩니다 ([Using Shared Memory in CUDA C/C++ | NVIDIA Technical Blog](https://developer.nvidia.com/blog/using-shared-memory-cuda-cc/#:~:text=Because%20it%20is%20on,managed%20data%20caches)) ([How to Access Global Memory Efficiently in CUDA C/C++ Kernels | NVIDIA Technical Blog](https://developer.nvidia.com/blog/how-access-global-memory-efficiently-cuda-c-kernels/#:~:text=Grouping%20of%20threads%20into%20warps,0)). 

```cpp
// CUDA 커널 예제: 행렬 전치 (공유 메모리 최적화 예)
__global__ void matrixTransposeShared(const float* A, float* B, int N) {
    __shared__ float tile[32][32];
    int x = blockIdx.x * 32 + threadIdx.x;
    int y = blockIdx.y * 32 + threadIdx.y;
    if (x < N && y < N) {
        tile[threadIdx.y][threadIdx.x] = A[y * N + x];
    }
    __syncthreads();
    x = blockIdx.y * 32 + threadIdx.x;  // 블록 좌표 반전
    y = blockIdx.x * 32 + threadIdx.y;
    if (x < N && y < N) {
        B[y * N + x] = tile[threadIdx.x][threadIdx.y];
    }
}
```
위 예제는 공유 메모리(`tile`)를 사용해 전치 작업을 수행합니다. 먼저 글로벌 메모리 `A`에서 데이터를 타일 형태로 공유 메모리에 로드한 뒤, 쓰레드 동기화 후에 전치된 인덱스로 데이터를 다시 글로벌 메모리 `B`에 저장합니다. 이렇게 하면 글로벌 메모리 접근의 연속성을 확보하여 성능을 높일 수 있습니다.

## 데이터 전송 최적화  
CPU와 GPU 간 데이터 전송은 성능에 큰 영향을 미칩니다. 기본적으로 `cudaMemcpy` 함수를 사용하여 호스트와 디바이스 메모리 간 동기식 복사를 수행합니다. 이때 **페이지 잠김(pinned)** 메모리를 사용하면 전송 속도와 효율이 개선됩니다. 일반 힙 메모리(pageable memory) 대신 `cudaMallocHost()`로 할당한 페이지 잠금 메모리는 DMA(Direct Memory Access)를 통해 GPU로 직접 전송할 수 있어, 추가 버퍼 복사 없이 빠른 데이터 전송과 CPU/GPU 연산의 중첩(오버랩)이 가능합니다 ([Does cudaMemcpyAsync require pinned memory? - CUDA Programming and Performance - NVIDIA Developer Forums](https://forums.developer.nvidia.com/t/does-cudamemcpyasync-require-pinned-memory/40411#:~:text=,memory%20copy%3F)) ([cuda - Effect of using page-able memory for asynchronous memory copy? - Stack Overflow](https://stackoverflow.com/questions/14093601/effect-of-using-page-able-memory-for-asynchronous-memory-copy#:~:text=Optionally%2C%20if%20the%20call%20is,block%20the%20calling%20host%20thread)). 예를 들어, 비동기 전송(`cudaMemcpyAsync`)을 사용하려면 호스트 메모리를 반드시 페이지 잠금 메모리로 할당해야 진정한 비동기(커널 실행과 전송 동시 진행)가 가능합니다 ([Does cudaMemcpyAsync require pinned memory? - CUDA Programming and Performance - NVIDIA Developer Forums](https://forums.developer.nvidia.com/t/does-cudamemcpyasync-require-pinned-memory/40411#:~:text=,memory%20copy%3F)) ([cuda - Effect of using page-able memory for asynchronous memory copy? - Stack Overflow](https://stackoverflow.com/questions/14093601/effect-of-using-page-able-memory-for-asynchronous-memory-copy#:~:text=Optionally%2C%20if%20the%20call%20is,block%20the%20calling%20host%20thread)). 만약 페이지 잠금이 아닌 메모리에서 비동기 복사를 시도하면, CUDA 런타임이 임시로 페이지 잠금 버퍼로 복사한 후 전송하므로 실제 GPU와 커널 연산 간 중첩이 일어나지 않습니다 ([cuda - Effect of using page-able memory for asynchronous memory copy? - Stack Overflow](https://stackoverflow.com/questions/14093601/effect-of-using-page-able-memory-for-asynchronous-memory-copy#:~:text=Optionally%2C%20if%20the%20call%20is,block%20the%20calling%20host%20thread)).  

CUDA **스트림(stream)**과 **이벤트(event)**는 비동기 작업과 동시 실행을 관리하는 도구입니다. 스트림은 작업 실행 순서를 지정하는 큐로, 기본 스트림(default stream)과 사용자 정의 스트림이 있습니다 ([How to Overlap Data Transfers in CUDA C/C++ | NVIDIA Technical Blog](https://developer.nvidia.com/blog/how-overlap-data-transfers-cuda-cc/#:~:text=CUDA%20Streams)). 하나의 스트림 내에서는 작업이 순차 실행되지만, 서로 다른 스트림 간에는 가능하면 작업이 병렬로 실행됩니다 ([How to Overlap Data Transfers in CUDA C/C++ | NVIDIA Technical Blog](https://developer.nvidia.com/blog/how-overlap-data-transfers-cuda-cc/#:~:text=A%20stream%20in%20CUDA%20is,they%20can%20even%20run%20concurrently)). 예를 들어, 별도의 스트림을 생성하여 `cudaMemcpyAsync`와 커널 런치를 해당 스트림에서 실행하면, 이론적으로 데이터 전송과 커널 실행을 오버랩할 수 있습니다. CUDA 이벤트는 GPU 작업 완료 여부를 확인하거나 타이밍을 측정할 때 사용됩니다. `cudaEventRecord(event, stream)`으로 이벤트를 기록하고 `cudaEventSynchronize(event)`로 이벤트 완료를 대기하면, 특정 시점까지의 스트림 작업 완료를 보장할 수 있습니다 ([CUDA Series: Streams and Synchronization | by Dmitrij Tichonov | Medium](https://medium.com/@dmitrijtichonov/cuda-series-streams-and-synchronization-873a3d6c22f4#:~:text=cudaEventSynchronize)).  

```cpp
// 스트림과 페이지드 메모리 활용 예제
cudaStream_t stream;
cudaStreamCreate(&stream);
float *h_A, *h_B; 
cudaMallocHost(&h_A, N*sizeof(float)); // 페이지 잠금 호스트 메모리
cudaMallocHost(&h_B, N*sizeof(float));
float *d_A, *d_B;
cudaMalloc(&d_A, N*sizeof(float));
cudaMalloc(&d_B, N*sizeof(float));
// 비동기 전송 및 커널 실행
cudaMemcpyAsync(d_A, h_A, N*sizeof(float), cudaMemcpyHostToDevice, stream);
kernel<<<grid, block, 0, stream>>>(d_A, d_B);
cudaMemcpyAsync(h_B, d_B, N*sizeof(float), cudaMemcpyDeviceToHost, stream);
// 호스트에서 모든 작업 완료 대기
cudaStreamSynchronize(stream);
```

## GPU 특화 라이브러리 활용  
CUDA 환경에서는 다양한 GPU 최적화 라이브러리를 제공합니다. 대표적으로 **cuBLAS**는 GPU 최적화 BLAS 라이브러리로, 행렬·벡터 연산(GEMM, GEMV 등)을 고성능으로 수행합니다. 예를 들어, 단일 정밀도 행렬 곱셈 `C = A * B`를 위해 다음과 같이 `cublasSgemm` 함수를 호출할 수 있습니다.

```cpp
cublasHandle_t handle;
cublasCreate(&handle);
float alpha = 1.0f, beta = 0.0f;
cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, M, N, K,
            &alpha, A_d, M, B_d, K, &beta, C_d, M);
```

위 코드에서 `cublasSgemm`은 GPU 메모리 `A_d`, `B_d`, `C_d`에 저장된 행렬을 인자로 받아 연산을 수행합니다. cuBLAS는 내부적으로 고효율 알고리즘과 텐서 코어를 활용하여 성능을 극대화합니다.  

**cuDNN**은 딥러닝 연산에 특화된 NVIDIA의 라이브러리로, 합성곱(convolution), 활성화 함수, 풀링(pooling) 등의 연산을 GPU에서 최적화하여 제공합니다 ([Convolutions with cuDNN – Peter Goldsborough](http://www.goldsborough.me/cuda/ml/cudnn/c++/2017/10/01/14-37-23-convolutions_with_cudnn/#:~:text=CuDNN%20is%20a%20closed,more%20work%20than%20you%E2%80%99d%20expect)). 예를 들어, 합성곱 연산을 수행할 때는 cuDNN 핸들(`cudnnHandle_t`)을 생성하고, 입력 텐서 및 필터에 대한 디스크립터를 설정한 뒤 `cudnnConvolutionForward` 함수를 호출합니다.

```cpp
cudnnHandle_t cudnn;
cudnnCreate(&cudnn);
// 텐서 및 필터 디스크립터 설정 (예: NCHW 포맷)
cudnnTensorDescriptor_t inputDesc, outputDesc;
cudnnFilterDescriptor_t filterDesc;
cudnnConvolutionDescriptor_t convDesc;
cudnnSetTensor4dDescriptor(inputDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                           batchSize, inChannels, inHeight, inWidth);
cudnnSetFilter4dDescriptor(filterDesc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
                           outChannels, inChannels, kernelH, kernelW);
cudnnSetConvolution2dDescriptor(convDesc, padH, padW, strideH, strideW,
                                dilationH, dilationW, CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT);
// 합성곱 알고리즘 선택
cudnnConvolutionForward(cudnn, &alpha,
                        inputDesc, inputDev, filterDesc, filterDev,
                        convDesc, algo, workspace, workspaceSize,
                        &beta, outputDesc, outputDev);
```

위와 같이 cuDNN을 사용하면 복잡한 합성곱 계산을 직접 구현할 필요 없이, 간단한 API 호출로 GPU 최적화된 연산을 수행할 수 있습니다. 주요 딥러닝 프레임워크(TensorFlow, PyTorch 등)는 내부적으로 cuDNN을 사용하여 GPU 가속 합성곱 연산을 수행합니다 ([Convolutions with cuDNN – Peter Goldsborough](http://www.goldsborough.me/cuda/ml/cudnn/c++/2017/10/01/14-37-23-convolutions_with_cudnn/#:~:text=CuDNN%20is%20a%20closed,more%20work%20than%20you%E2%80%99d%20expect)). 

## 실제 예제  
### 벡터 덧셈 커널  
다음은 CUDA를 이용한 간단한 벡터 덧셈 예제입니다.  

```cpp
#include <cuda_runtime.h>
#include <iostream>

__global__ void vectorAdd(const float* A, const float* B, float* C, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        C[i] = A[i] + B[i];
    }
}

int main() {
    const int N = 1<<20;
    size_t bytes = N * sizeof(float);
    // 호스트 메모리 할당 및 초기화
    float *h_A = (float*)malloc(bytes), *h_B = (float*)malloc(bytes), *h_C = (float*)malloc(bytes);
    for (int i = 0; i < N; i++) {
        h_A[i] = float(i);
        h_B[i] = float(2*i);
    }
    // 디바이스 메모리 할당
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, bytes);
    cudaMalloc(&d_B, bytes);
    cudaMalloc(&d_C, bytes);
    // 데이터 전송 (Host → Device)
    cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice);
    // 커널 실행
    int blockSize = 256;
    int gridSize = (N + blockSize - 1) / blockSize;
    vectorAdd<<<gridSize, blockSize>>>(d_A, d_B, d_C, N);
    cudaDeviceSynchronize();
    // 결과 전송 (Device → Host)
    cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost);
    // 결과 확인
    std::cout << "C[0] = " << h_C[0] << ", C[N-1] = " << h_C[N-1] << std::endl;
    // 메모리 해제
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    free(h_A); free(h_B); free(h_C);
    return 0;
}
```

위 코드에서 `vectorAdd` 커널은 각 쓰레드가 하나의 배열 요소를 처리합니다. 적절한 블록/그리드 크기를 설정하여 모든 요소를 덮도록 합니다. 이 예제를 NVCC로 컴파일하고 실행하면 벡터 덧셈 결과를 확인할 수 있습니다.

### 행렬 곱셈 커널  
다음은 간단한 나이브 행렬 곱셈 예제입니다. 성능 향상을 위해 공유 메모리를 사용할 수도 있지만, 여기서는 이해를 위해 직관적인 2중 중첩 루프 버전을 보여줍니다.  

```cpp
#include <cuda_runtime.h>
#include <iostream>

__global__ void matMul(const float* A, const float* B, float* C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < N && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < N; ++k) {
            sum += A[row * N + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

int main() {
    const int N = 512;
    size_t bytes = N * N * sizeof(float);
    float *h_A = (float*)malloc(bytes), *h_B = (float*)malloc(bytes), *h_C = (float*)malloc(bytes);
    // 행렬 초기화 (예: 모든 원소를 1로)
    for (int i = 0; i < N*N; i++) {
        h_A[i] = 1.0f;
        h_B[i] = 2.0f;
    }
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, bytes); cudaMalloc(&d_B, bytes); cudaMalloc(&d_C, bytes);
    cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice);
    dim3 blockSize(16, 16);
    dim3 gridSize((N + blockSize.x - 1) / blockSize.x,
                  (N + blockSize.y - 1) / blockSize.y);
    matMul<<<gridSize, blockSize>>>(d_A, d_B, d_C, N);
    cudaDeviceSynchronize();
    cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost);
    std::cout << "C[0] = " << h_C[0] << std::endl;  // 모든 원소 합 (1*2*N)
    // 메모리 해제
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    free(h_A); free(h_B); free(h_C);
    return 0;
}
```

이 코드는 크기가 512×512인 행렬 곱셈을 수행합니다. `matMul` 커널에서 각 쓰레드는 하나의 출력 원소 \(C[row][col]\)를 계산합니다. 이 버전은 복잡한 공유 메모리 최적화를 사용하지 않으므로 메모리 병목이 발생할 수 있습니다. 성능을 높이기 위해서는 **블록 타일링**, **공유 메모리** 활용 등을 고려해야 합니다.

## 성능 주의사항  
CUDA 커널 성능 최적화를 위해 다음과 같은 점을 고려해야 합니다.  

- **메모리 병목**: 전역 메모리 접근이 잦을 경우 대역폭 한계에 걸릴 수 있습니다. 연속적이지 않은 접근(스트라이드 접근)이나 비효율적 메모리 패턴을 피하고, 가능한 한 **메모리 공동접속(coalescing)**을 유도해야 합니다 ([How to Access Global Memory Efficiently in CUDA C/C++ Kernels | NVIDIA Technical Blog](https://developer.nvidia.com/blog/how-access-global-memory-efficiently-cuda-c-kernels/#:~:text=Grouping%20of%20threads%20into%20warps,0)). 또 공유 메모리에 자주 쓰고 읽을 때 **뱅크 충돌**이 발생하면 병목이 심해지므로, 인덱스를 조정하여 충돌을 최소화해야 합니다 ([Using Shared Memory in CUDA C/C++ | NVIDIA Technical Blog](https://developer.nvidia.com/blog/using-shared-memory-cuda-cc/#:~:text=Because%20it%20is%20on,Shared%20memory%20is%20allocated)).  
- **Occupancy(점유도)**: GPU의 SM당 활성 워프 수를 의미하는 점유도가 지나치게 낮으면 코어 자원이 놀게 되어 성능이 저하될 수 있습니다. 블록 크기, 레지스터 수, 공유 메모리 사용량 등을 조정해 점유도를 높이는 것이 중요합니다. 다만, 점유도가 높다고 항상 최적은 아니므로 작업 특성과 데이터 종속성을 고려해야 합니다.  
- **출력 병목**: 커널의 출력이 적거나 데이터 전송이 비효율적이면 GPU의 계산 능력이 낭비될 수 있습니다. CPU와 GPU 간 전송 비용 대비 효율을 항상 고려하십시오.  
- **연산 대역폭 불균형**: 단순 연산 대비 메모리 접근이나 기타 대기 시간이 많으면 연산 유닛이 놀 수 있으므로, 연산-메모리 비율을 감안하여 최적화해야 합니다.  

## 연습 문제  
- 벡터 덧셈 커널 및 스칼라 곱셈(kernel) 프로그램을 작성해 보세요. 각 쓰레드가 배열의 요소 하나를 처리하도록 하며, 적절한 그리드/블록 크기를 선택하십시오.  
- 비동기 데이터 전송을 활용하여 커널을 실행해 보세요. 페이지드 메모리와 `cudaMemcpyAsync`, `cudaStream_t`를 사용하여 CPU와 GPU 연산이 오버랩되도록 구성해 보세요.  
- 간단한 행렬 곱셈 커널을 작성해 보세요. 그리드와 블록 크기에 따른 성능 변화를 측정하고, 공유 메모리를 이용해 최적화를 시도해 보세요.  