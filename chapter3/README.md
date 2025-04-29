# 3장. 수치 연산과 자료구조

이 장에서는 벡터와 행렬 같은 자료구조를 설계하여 수치 계산을 효율적으로 처리하는 방법을 다룬다. **메모리 연속성**(row-major vs column-major)을 고려한 벡터/행렬 클래스 설계, **메모리 풀**과 **버퍼 재사용**을 통한 메모리 관리 기법, 그리고 **행렬 곱셈 알고리즘**의 캐시 최적화(블록킹 기법)를 학습한다. 또한 고성능 C++ 선형대수 라이브러리인 **Eigen**, **Armadillo**, **xtensor**, **Boost.uBLAS**의 특징과 사용법을 비교해 본다. 이들 라이브러리는 공통적으로 템플릿 기반으로 구현되어 있으며, *지연평가(lazy evaluation)*와 *브로드캐스팅* 등을 지원해 복잡한 표현식도 효율적으로 계산할 수 있다. 본 장의 주요 내용은 다음과 같다.

- **벡터/행렬 클래스 설계**: 메모리 연속성을 고려한 배열 배치(행 우선 vs 열 우선)와 인덱싱 방법.  
- **메모리 풀/버퍼 재사용**: 미리 할당된 메모리 블록을 이용한 동적 할당 비용 절감 기법.  
- **효율적인 행렬 곱셈**: 기본 곱셈 알고리즘과 *블록킹(tiling)* 최적화 기법을 통한 캐시 친화적 구현.  
- **선형대수 라이브러리 비교**: Eigen, Armadillo, xtensor, Boost.uBLAS의 개요 및 사용 예제.  

각 개념에 대한 이론 설명과 함께 완전한 동작 예제 코드를 제공한다. 주요 절 끝에는 자기 학습을 위한 **연습문제**를 포함한다.  

## 벡터/행렬 클래스 설계

고성능 행렬 연산을 위해서는 데이터의 **메모리 연속성**을 고려해야 한다. 컴퓨터는 연속 메모리를 효율적으로 처리하므로, 2차원 행렬도 1차원 배열에 저장할 때 행 우선(row-major) 또는 열 우선(column-major) 방식으로 배치할 수 있다. *행 우선*에서는 같은 행(row)의 요소들이 연속 저장되고, *열 우선*에서는 같은 열(column)의 요소들이 연속 저장된다. 즉 C/C++ 다차원 배열은 기본적으로 행 우선 방식으로 저장된다 ([Row- and column-major order - Wikipedia](https://en.wikipedia.org/wiki/Row-_and_column-major_order#:~:text=The%20difference%20between%20the%20orders,using%20this%20approach%20are%20effectively)). 반면 Fortran이나 BLAS 라이브러리 일부는 열 우선 방식을 사용한다. 어떤 방식을 택하든 **연속적인 메모리 접근**이 중요하다. 같은 행 내 데이터를 순차적으로 읽으면 CPU 캐시가 이점을 얻어 빠르게 처리되지만, 열 방향으로 큰 간격을 두고 접근하면 캐시 미스가 증가하여 성능이 떨어진다 ([Row- and column-major order - Wikipedia](https://en.wikipedia.org/wiki/Row-_and_column-major_order#:~:text=traversing%20an%20array%20because%20modern,such%20as%20%2054%2C%20accessing)).

예를 들어, 행우선 방식으로 3×4 행렬 `A`를 저장하면 메모리에는 `[A(0,0), A(0,1), A(0,2), A(0,3), A(1,0), ...]` 순으로 연속 저장된다. 열우선 방식으로 저장하면 `[A(0,0), A(1,0), A(2,0), A(0,1), ...]` 식으로 배치된다. 이를 코드로 확인해 보자. 아래 예제는 3×4 크기의 행렬을 설계하는 클래스이다. `rowMajor` 플래그로 저장 방식을 선택할 수 있으며, `(i,j)` 인덱싱 연산자가 메모리 오프셋을 계산해 준다. 또한 `raw_data()` 메서드로 내부 저장 버퍼를 출력하여 실제 메모리 배치를 확인할 수 있다.  

```cpp
#include <iostream>
#include <vector>

template<typename T>
class Matrix {
private:
    int rows, cols;
    bool rowMajor;          // 행우선(true) 또는 열우선(false)
    std::vector<T> data;
public:
    Matrix(int r, int c, bool row_major = true)
        : rows(r), cols(c), rowMajor(row_major), data(r*c) {}
    // (i,j) 원소 접근
    T& operator()(int i, int j) {
        if (rowMajor)
            return data[i*cols + j];   // 행우선: i행의 j열
        else
            return data[j*rows + i];   // 열우선: j열의 i행
    }
    // 내부 데이터 버퍼 접근 (디버깅용)
    const std::vector<T>& raw_data() const { return data; }
};

int main() {
    Matrix<double> A(3, 4, true);   // 행우선
    Matrix<double> B(3, 4, false);  // 열우선
    for(int i=0;i<3;i++){
        for(int j=0;j<4;j++){
            A(i,j) = i*4 + j;   // 0,1,2,...,11 순으로 저장
            B(i,j) = i*4 + j;
        }
    }
    std::cout<<"A data: ";
    for(double x : A.raw_data()) std::cout<<x<<' ';
    std::cout<<"\nB data: ";
    for(double x : B.raw_data()) std::cout<<x<<' ';
    std::cout<<std::endl;
    return 0;
}
```

위 코드에서 `A`는 행우선, `B`는 열우선으로 저장된다. 실행 결과를 보면:  
```
A data: 0 1 2 3 4 5 6 7 8 9 10 11 
B data: 0 4 8 1 5 9 2 6 10 3 7 11 
```  
`A`는 0부터 11까지 순서대로 저장되었지만, `B`는 같은 행 값들이 떨어져 저장되어 있음을 알 수 있다. 실제 연산에서는 행우선이나 열우선 중 하나로 일관되게 저장하고, 메모리 접근 패턴에 맞추어 인덱싱해야 캐시 효율을 높일 수 있다.  

벡터 클래스도 비슷한 원리로 설계할 수 있다. 단순한 1차원 벡터는 `std::vector<T>` 등을 사용하면 되지만, 고정 크기 벡터는 템플릿을 이용한 배열로 구현할 수 있다. 예를 들어:  

```cpp
template<int N, typename T>
class StaticVector {
    T data[N];
public:
    T& operator[](int i) { return data[i]; }
    const T& operator[](int i) const { return data[i]; }
};
```  

`StaticVector<3,double>` 같이 사용하면 컴파일 시 크기가 고정되어 빠르게 접근할 수 있다.  

- **벡터/행렬 클래스 설계 시 고려 사항**:  
  - *메모리 레이아웃*: 행우선과 열우선 중 시스템과 알고리즘에 맞는 것을 선택.  
  - *연속성 유지*: 데이터를 연속 배열에 저장하여 캐시 효율 극대화 ([Row- and column-major order - Wikipedia](https://en.wikipedia.org/wiki/Row-_and_column-major_order#:~:text=The%20difference%20between%20the%20orders,using%20this%20approach%20are%20effectively)).  
  - *템플릿 활용*: 행렬 크기나 데이터 타입을 컴파일 시점에 결정하도록 구현하여 성능 향상.  

### 연습문제

- 직접 2차원 행렬 클래스를 구현하라. 행우선과 열우선 모드를 선택할 수 있도록 하고, `(i,j)` 연산자를 오버로드하여 올바른 원소를 반환하도록 하라.  
- `Matrix` 클래스를 이용해 5×5 행렬을 만들어 연산을 테스트해 보고, `rowMajor` 플래그를 바꾸었을 때 내부 메모리 순서가 어떻게 바뀌는지 확인해 보라.  
- 연속 메모리 접근이 캐시 성능에 미치는 영향을 실험해 보라. 예를 들어, 행우선 배열을 열 단위로 순회하는 코드와 행 단위로 순회하는 코드의 실행 시간을 비교하라.  

## 메모리 풀과 버퍼 재사용

동적 메모리 할당(`new`/`delete` 또는 `malloc`/`free`)은 런타임 비용이 크고, 빈번한 할당/해제는 메모리 단편화를 초래할 수 있다. **메모리 풀(Memory Pool)**은 미리 큰 메모리 블록을 할당해 두고, 고정 크기 블록 단위로 메모리를 제공함으로써 할당 비용을 절감하고 단편화를 줄이는 기법이다 ([What is a Memory Pool? | GeeksforGeeks](https://www.geeksforgeeks.org/what-is-a-memory-pool/#:~:text=A%20memory%20pool%2C%20also%20known,that%20offers%20a%20number%20of)) ([What is a Memory Pool? | GeeksforGeeks](https://www.geeksforgeeks.org/what-is-a-memory-pool/#:~:text=,Improved%20performance%20and%20stability)). 즉 메모리 풀은 *미리 할당된 고정 크기 블록들의 집합*으로, 필요할 때마다 이 풀에서 블록을 꺼내 쓰고, 해제할 때는 다시 풀에 반환한다. 미리 할당된 블록들은 연속 메모리로 구성되어 있으므로 할당/해제 속도가 빠르다.  

메모리 풀의 장점은 *할당/해제 속도가 빠르고, 메모리 단편화를 줄여 예측 가능한 메모리 사용이 가능*하다는 점이다 ([What is a Memory Pool? | GeeksforGeeks](https://www.geeksforgeeks.org/what-is-a-memory-pool/#:~:text=,Improved%20performance%20and%20stability)). 예를 들어 물리 연산이나 게임 서버처럼 짧은 시간에 많은 개체를 생성/파괴하는 환경에서 유용하다. 단점은 *다양한 크기의 할당에 유연하지 못하고*, 구현 복잡도가 증가하며, 사용자가 메모리 해제 시 실수하면 누수(leak)가 발생할 수 있다는 점이다 ([What is a Memory Pool? | GeeksforGeeks](https://www.geeksforgeeks.org/what-is-a-memory-pool/#:~:text=Disadvantages%20of%20memory%20pools)).  

아래 예제는 간단한 메모리 풀 클래스이다. `MemoryPool`는 내부에 정수만 저장하는 풀(pool of `int`)을 구현했다. `allocate()`는 풀에서 빈 블록을 꺼내어 주소를 반환하고, `deallocate()`는 반환된 포인터를 다시 풀로 되돌린다. 여기서는 **POD 형식**(특수 생성자가 필요 없는 타입)의 객체에만 적용하도록 한다. 실제 객체를 저장하려면 placement new와 명시적 소멸자 호출을 추가해야 하지만, 이 예제에서는 생략하였다.  

```cpp
#include <iostream>
#include <vector>

template<typename T>
class MemoryPool {
private:
    struct Node { Node* next; };
    Node* head;
    std::vector<Node*> chunks;
    size_t chunkSize;

    // 새로운 메모리 블록(chunk) 할당
    void allocateBlock() {
        Node* block = new Node[chunkSize];
        chunks.push_back(block);
        for (size_t i = 0; i < chunkSize; ++i) {
            block[i].next = head;
            head = &block[i];
        }
    }

public:
    MemoryPool(size_t size = 1024) : head(nullptr), chunkSize(size) {
        allocateBlock();  // 초기 블록 할당
    }
    T* allocate() {
        if (!head) allocateBlock();  // 빈 블록이 없으면 추가 할당
        Node* node = head;
        head = head->next;
        return reinterpret_cast<T*>(node);
    }
    void deallocate(T* ptr) {
        Node* node = reinterpret_cast<Node*>(ptr);
        node->next = head;
        head = node;
    }
    ~MemoryPool() {
        // 할당된 모든 블록 해제
        for (Node* block : chunks) {
            delete[] block;
        }
    }
};

int main(){
    MemoryPool<int> pool(5);   // 크기 5 블록씩 할당
    int* a = pool.allocate();
    int* b = pool.allocate();
    *a = 10; *b = 20;
    std::cout << *a << ", " << *b << std::endl;
    // 메모리 해제(풀로 반환)
    pool.deallocate(a);
    pool.deallocate(b);
    // 재사용: 이전에 a, b가 사용하던 메모리를 다시 받음
    int* c = pool.allocate();
    std::cout << *c << " (재사용된 메모리 내용)" << std::endl;
    return 0;
}
```

위 코드에서 `a`, `b`에 할당된 메모리는 해제 후 풀로 반환되었으므로 `c = pool.allocate()` 시 `a` 또는 `b`가 사용하던 메모리 주소가 재활용된다. 출력 결과의 마지막 줄에 이상한 값이 나오는 것은 초기화되지 않은 메모리를 재사용했기 때문이다. 메모리 풀을 사용할 때는 재활용된 블록에 객체를 새로 생성해야 하며, 필요한 경우 해당 위치에 placement new를 통해 생성자를 호출해야 한다.  

- **메모리 풀/버퍼 설계 시 고려 사항**:  
  - *고정 블록 크기* 또는 *다중 크기 지원*: 단일 크기냐, 다양한 크기냐에 따라 설계가 달라진다.  
  - *풀 초기화 방식*: 한 번에 큰 청크를 할당하거나, 점진적으로 늘려가는 방법이 있다 ([Optimizing C++ Memory Management with a Custom Memory Pool - DEV Community](https://dev.to/pigeoncodeur/optimizing-c-memory-management-with-a-custom-memory-pool-1o3b#:~:text=1,future%20use%2C%20ensuring%20efficient%20reallocation)).  
  - *객체 생성/소멸*: POD가 아닌 객체라면 `new(ptr) T()`와 `ptr->~T()`를 이용해 생성자/소멸자를 직접 호출.  

### 연습문제

- 위 `MemoryPool` 코드를 확장하여 **객체 풀(Object Pool)**을 구현하라. 즉, 클래스 템플릿을 통해 사용자 정의 객체에 대해 생성자 호출과 소멸자 호출을 지원하게 수정하라. (힌트: placement new와 명시적 소멸자 호출 사용)  
- 프로그램이 다양한 크기의 메모리를 요청하도록 변경해 보라. 예를 들어, `MemoryPool<char>`로 다양한 크기의 바이트 배열을 할당/해제하면서 분기-접합(first-fit, best-fit 등) 알고리즘을 적용해보라.  
- 스마트 포인터(`std::unique_ptr`)나 컨테이너와 함께 메모리 풀을 사용하도록 `std::allocator`를 구현해 보라. 직접 `std::allocator_traits` 특성을 만족하도록 한다.  

## 효율적인 행렬 곱셈

행렬 곱셈은 머신러닝과 과학 계산에서 빈번히 사용되는 연산이며, 시간 복잡도는 일반적으로 \(O(n^3)\)이다. 기본적인 3중 반복문을 사용한 알고리즘은 이해하기 쉽지만, 메모리 접근 패턴이 비효율적일 수 있다. 예를 들어 전형적인 `i-k-j` 순서의 알고리즘은 다음과 같다: 

```cpp
for (int i = 0; i < n; i++)
  for (int k = 0; k < n; k++) {
    double v = A[i][k];
    for (int j = 0; j < n; j++)
      C[i][j] += v * B[k][j];
  }
```

이때 `B[k][j]`의 인덱스는 행(i) 고정, 열(k)가 빠르게 변하는 형태로 메모리를 건너뛰며 접근한다. 이런 방식은 CPU 캐시의 공간적 지역성을 잘 활용하지 못해 **캐시 미스(cache miss)**를 많이 발생시킨다 ([matrix multiplication - Cache friendly method to multiply two matrices - Stack Overflow](https://stackoverflow.com/questions/13312625/cache-friendly-method-to-multiply-two-matrices#:~:text=Basically%2C%20navigating%20the%20memory%20fastly,access%20index%20of%20the%20matrices)). 실제로 대량의 행렬 곱셈에서는 메모리 계층(캐시와 RAM) 사이의 데이터 이동이 병목이 되기 쉽다. 

이를 해결하는 대표적인 방법이 **블록킹(blocking 또는 tiling)** 기법이다. 블록킹은 큰 행렬을 작은 블록(submatrix) 단위로 나누어 곱셈하는 방식이다. 예를 들어 \(n\times n\) 행렬을 \(b\times b\) 크기 블록으로 분할하고, 각 블록끼리 곱해서 결과를 누적한다. 이렇게 하면 한 번에 작게 로드된 블록이 캐시 안에 오래 머물러 재사용성이 높아진다. 블록 사이즈 \(b\)가 크면 클수록 계산 대비 메모리 접근 비율(연산 집약도, computational intensity)이 증가하여 성능이 향상된다 ([Blocked Matrix Multiplication | Malith Jayaweera](https://malithjayaweera.com/2020/07/blocked-matrix-multiplication/#:~:text=Therefore%2C%20the%20computational%20intensity%20is%2C,b%20for%20large%20n)). 즉, 연산(flop) 수 대비 메모리 접근 수 비율 \(q\approx b\)로 증가하므로, 충분히 큰 블록을 사용하면 캐시 히트를 줄이고 벡터화 등의 이점을 극대화할 수 있다 ([Blocked Matrix Multiplication | Malith Jayaweera](https://malithjayaweera.com/2020/07/blocked-matrix-multiplication/#:~:text=Therefore%2C%20the%20computational%20intensity%20is%2C,b%20for%20large%20n)).

아래 예제는 단순 행렬 곱셈과 블록 단위 행렬 곱셈의 구현이다. `multiplyNaive()`는 3중 반복문을 사용한 기본 알고리즘이고, `multiplyBlocked()`는 겉 3중 반복문을 블록 단위로 순회하도록 개선한 형태이다. 예제에서는 `std::vector<double>`로 행렬을 1차원 배열로 표현하였다.  

```cpp
#include <iostream>
#include <vector>
#include <algorithm>
using Mat = std::vector<double>;

// Naive matrix multiply (C = A * B)
void multiplyNaive(const Mat& A, const Mat& B, Mat& C, int n) {
    for(int i=0; i<n; i++){
        for(int k=0; k<n; k++){
            double v = A[i*n + k];
            for(int j=0; j<n; j++){
                C[i*n + j] += v * B[k*n + j];
            }
        }
    }
}

// Blocked matrix multiply (block size = Bsize)
void multiplyBlocked(const Mat& A, const Mat& B, Mat& C, int n, int Bsize) {
    for(int i0=0; i0<n; i0+=Bsize){
        for(int j0=0; j0<n; j0+=Bsize){
            for(int k0=0; k0<n; k0+=Bsize){
                // 한 블록 내 연산
                for(int i=i0; i<std::min(i0+Bsize,n); i++){
                    for(int k=k0; k<std::min(k0+Bsize,n); k++){
                        double v = A[i*n + k];
                        for(int j=j0; j<std::min(j0+Bsize,n); j++){
                            C[i*n + j] += v * B[k*n + j];
                        }
                    }
                }
            }
        }
    }
}

int main() {
    int n = 4;
    Mat A(n*n), B(n*n), C1(n*n), C2(n*n);
    // 행렬 초기화
    for(int i=0;i<n*n;i++){ A[i] = i+1; B[i] = i+1; }
    multiplyNaive(A, B, C1, n);
    multiplyBlocked(A, B, C2, n, 2);
    std::cout << "C1: ";
    for(double x: C1) std::cout << x << ' ';
    std::cout << "\nC2: ";
    for(double x: C2) std::cout << x << ' ';
    std::cout << std::endl;
    return 0;
}
```

위 코드는 \(4\times4\) 행렬을 예로 들어 **나이브 곱셈**과 **블록 곱셈**의 결과를 비교한다. 출력된 `C1`과 `C2`는 동일해야 한다(실행 결과 둘 다 `90 100 110 120 202 228 254 280 314 356 398 440 426 484 542 600`으로 계산됨). 실제 큰 행렬에서는 블록킹을 통해 캐시 효율이 개선되어 실행 속도가 빨라진다. 물론 최적 블록 크기는 하드웨어 캐시 크기에 따라 달라지며, 지나치게 작은 블록은 오히려 오버헤드를 증가시킬 수 있다.  

- **행렬 곱셈 최적화 포인트**:  
  - *메모리 접근 패턴*: 가능한 한 연속 메모리에 접근하도록 루프 순서를 조정한다 ([matrix multiplication - Cache friendly method to multiply two matrices - Stack Overflow](https://stackoverflow.com/questions/13312625/cache-friendly-method-to-multiply-two-matrices#:~:text=Basically%2C%20navigating%20the%20memory%20fastly,access%20index%20of%20the%20matrices)).  
  - *블록킹*: 행렬을 \(b\times b\) 블록으로 나누어 계산해 캐시에 적합하게 데이터 재사용성을 높인다 ([Blocked Matrix Multiplication | Malith Jayaweera](https://malithjayaweera.com/2020/07/blocked-matrix-multiplication/#:~:text=Therefore%2C%20the%20computational%20intensity%20is%2C,b%20for%20large%20n)).  
  - *SIMD 벡터화*: 블록 내 연산을 SIMD 명령어로 병렬 처리하여 연산 집약도를 높인다(예: AVX 명령어 사용).  

### 연습문제

- 작은 크기의 행렬(예: \(8\times8\))에 대해 나이브 곱셈과 블록 곱셈의 실행 시간을 측정해 보라. 다양한 블록 크기(\(b=2,4,8\) 등)에 대해 성능 차이를 분석하라.  
- 일반적인 \(i\)-\(j\)-\(k\) 순서, \(i\)-\(k\)-\(j\) 순서 등 루프 순서를 바꾸어 보면서 캐시 성능 차이를 실험해 보라.  
- SIMD 명령어를 사용하여 행렬 곱셈을 최적화해 보라. (예: Intel AVX 또는 ARM NEON). 컴파일러 벡터화 옵션과 `std::simd`(C++20)를 사용하여 구현해보고 성능을 비교하라.  

## C++ 선형대수 라이브러리

Eigen, Armadillo, xtensor, Boost.uBLAS는 모두 C++용 선형대수 라이브러리이다. 각 라이브러리는 특징과 장단점이 다르므로 목적에 맞게 선택해야 한다. 

- **Eigen**은 C++ 템플릿 기반의 선형대수 라이브러리로, 행렬과 벡터 연산, 분해, 솔버 등을 지원한다 ([Eigen](https://eigen.tuxfamily.org/index.php?title=Main_Page#:~:text=Eigen%20is%20a%20C%2B%2B%20template,numerical%20solvers%2C%20and%20related%20algorithms)). 모든 표준 수치 타입과 복소수, 사용자 정의 타입을 지원하며, 고정 크기(fixed-size)와 동적 크기(dynamic-size) 행렬을 모두 제공한다. 표현 템플릿(Expression Templates)을 사용하여 복합 연산 시 임시 객체 생성을 줄이고 지연 평가를 수행하므로 성능이 우수하다 ([Eigen](https://eigen.tuxfamily.org/index.php?title=Main_Page#:~:text=,4%20MIPS%20MSA%20with)). 또한 SSE/AVX 같은 SIMD 명령어를 활용한 벡터화 연산을 지원하여 대형 행렬 연산 시 높은 성능을 낸다. Eigen은 헤더 온리(header-only)로 제공되며, 사용자는 빌드할 필요 없이 `#include <Eigen/Dense>`만 하면 된다. 사용 예를 살펴보자:  

```cpp
#include <Eigen/Dense>
#include <iostream>
using namespace Eigen;

int main() {
    MatrixXd A = MatrixXd::Random(3, 3);
    MatrixXd B = MatrixXd::Random(3, 3);
    MatrixXd C = A * B;  // 행렬 곱셈
    std::cout << "Eigen result:\n" << C << std::endl;
    return 0;
}
```

위 예제에서 `MatrixXd`는 동적 크기(double형) 행렬을 나타내며, `A*B` 연산이 백그라운드에서 최적화된 고성능 코드로 처리된다. Eigen은 짧은 문법으로 복잡한 연산을 깔끔하게 표현할 수 있다.

- **Armadillo**는 MATLAB 유사 문법을 제공하는 C++ 선형대수 라이브러리이다 ([Armadillo: C++ library for linear algebra & scientific computing](https://arma.sourceforge.net/#:~:text=,speed%20and%20ease%20of%20use)). 내부적으로 LAPACK/BLAS이나 OpenBLAS, Intel MKL 등 고성능 라이브러리를 호출하여 연산 속도를 높인다. 고수준 API를 제공하여 벡터·행렬 생성, 분해, 통계 함수 등을 쉽게 사용할 수 있다. Armadillo도 표현 템플릿을 사용하여 연산을 최적화하며, 연관된 OpenMP 멀티스레딩을 통해 병렬처리를 지원한다. 사용 예시:  

```cpp
#include <armadillo>
using namespace arma;

int main() {
    mat A = randu<mat>(3,3);
    mat B = randu<mat>(3,3);
    mat C = A * B;
    C.print("Armadillo result:");
    return 0;
}
```

`randu<mat>(3,3)`은 0~1 난수로 채워진 3×3 행렬을 생성한다. `C.print()`는 행렬을 출력하며, 내부적으로 고성능 BLAS 연산이 수행된다 ([Armadillo: C++ library for linear algebra & scientific computing](https://arma.sourceforge.net/#:~:text=,to%20increase%20speed%20and%20efficiency)).

- **xtensor**는 NumPy 스타일의 다차원 배열(N-dimensional array)을 지원하는 라이브러리이다. C++ 템플릿으로 구현되어 있으며, 브로드캐스팅(Broadcasting)과 지연 평가를 제공한다 ([Introduction — xtensor  documentation](https://xtensor.readthedocs.io/#:~:text=xtensor%20is%20a%20C%2B%2B%20library,dimensional%20array%20expressions)). 파이썬 NumPy와 유사한 문법을 사용하며, C++ STL과 같은 반복자 및 컨테이너 기반 API를 제공한다. `xtensor`는 순차적인 다차원 배열을 백엔드로 사용하며, Python 버퍼 프로토콜을 통해 NumPy 배열과 연동할 수 있다. 사용 예시:  

```cpp
#include <iostream>
#include <xtensor/xarray.hpp>
#include <xtensor/xio.hpp>
#include <xtensor-blas/xlinalg.hpp>
using namespace xt;

int main() {
    xarray<double> A = {{1.0, 2.0}, {3.0, 4.0}};
    xarray<double> B = {{5.0, 6.0}, {7.0, 8.0}};
    auto C = linalg::dot(A, B);  // 행렬 곱셈
    std::cout << "xtensor result:\n" << C << std::endl;
    return 0;
}
```

`xarray`는 가변 크기(numpy `ndarray`와 유사) 배열이고, `linalg::dot` 함수는 BLAS 스타일의 행렬 곱셈을 수행한다(단, `xtensor-blas` 모듈 필요). xtensor는 브로드캐스팅과 유사 함수(ufunc)를 지원하여 다차원 배열 연산을 파이썬과 유사한 방식으로 간결히 표현할 수 있다 ([Introduction — xtensor  documentation](https://xtensor.readthedocs.io/#:~:text=xtensor%20is%20a%20C%2B%2B%20library,dimensional%20array%20expressions)).

- **Boost.uBLAS**는 Boost 라이브러리의 일부로, BLAS 수준의 벡터 및 행렬 연산을 제공하는 템플릿 라이브러리이다 ([Boost Basic Linear Algebra - 1.88.0](https://www.boost.org/libs/numeric/ublas/doc/index.htm#:~:text=uBLAS%20is%20a%20C%2B%2B%20template,code%20generation%20via%20expression%20templates)). `boost::numeric::ublas::matrix`와 같은 타입을 사용하며, 연산자는 오버로딩되어 벡터 덧셈, 내적, 행렬곱 등을 지원한다. 다만 기본 버전에서는 일반적인 BLAS 수준만 제공하므로, 고성능 연산을 위해서는 Intel MKL 등 외부 라이브러리와 연동하거나 `prod()` 함수로 명시적 행렬곱을 수행해야 한다. 사용 예제:  

```cpp
#include <iostream>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/io.hpp>
using namespace boost::numeric::ublas;

int main() {
    matrix<double> A(2,2), B(2,2);
    A(0,0)=1; A(0,1)=2; A(1,0)=3; A(1,1)=4;
    B(0,0)=5; B(0,1)=6; B(1,0)=7; B(1,1)=8;
    matrix<double> C = prod(A, B);  // 행렬 곱
    std::cout << "uBLAS result:\n" << C << std::endl;
    return 0;
}
```

`prod(A,B)` 함수는 uBLAS에서 행렬곱을 수행하는 함수이며, 결과 `C`는 2×2 행렬이 된다. Boost.uBLAS는 템플릿 기반이지만 고급 최적화는 Eigen이나 Armadillo보다 제한적일 수 있다.  

이 네 라이브러리의 특징을 정리하면 다음과 같다:  

- **Eigen**: 헤더 온리로 설치가 쉽고, 표현 템플릿과 SIMD 최적화로 빠르다 ([Eigen](https://eigen.tuxfamily.org/index.php?title=Main_Page#:~:text=Eigen%20is%20a%20C%2B%2B%20template,numerical%20solvers%2C%20and%20related%20algorithms)) ([Eigen](https://eigen.tuxfamily.org/index.php?title=Main_Page#:~:text=,4%20MIPS%20MSA%20with)). 고정 크기 행렬과 동적 크기 행렬을 모두 지원하며 API가 직관적이다.  
- **Armadillo**: MATLAB과 유사한 문법과 풍부한 함수(분해, 통계 등)를 제공한다 ([Armadillo: C++ library for linear algebra & scientific computing](https://arma.sourceforge.net/#:~:text=,speed%20and%20ease%20of%20use)) ([Armadillo: C++ library for linear algebra & scientific computing](https://arma.sourceforge.net/#:~:text=,to%20increase%20speed%20and%20efficiency)). 내부적으로 BLAS/LAPACK을 호출하므로 대형 행렬 연산에서 성능이 높다.  
- **xtensor**: 파이썬/NumPy 스타일의 사용성을 제공하며, n차원 배열과 브로드캐스팅을 지원한다 ([Introduction — xtensor  documentation](https://xtensor.readthedocs.io/#:~:text=xtensor%20is%20a%20C%2B%2B%20library,dimensional%20array%20expressions)). C++와 Python 간의 데이터 공유가 가능하여, Python 생태계와 연동하기 좋다.  
- **Boost.uBLAS**: Boost에 포함되어 언제든 사용할 수 있고, BLAS 수준의 기본 연산을 제공한다 ([Boost Basic Linear Algebra - 1.88.0](https://www.boost.org/libs/numeric/ublas/doc/index.htm#:~:text=uBLAS%20is%20a%20C%2B%2B%20template,code%20generation%20via%20expression%20templates)). 쓰기 간편하지만 대형 연산에서는 성능이 다소 떨어질 수 있다.  

### 연습문제

- Eigen과 Armadillo를 이용해 행렬 덧셈, 곱셈, 전치 연산을 구현해 보라. 직접 만든 `Matrix` 클래스 버전과 연산 결과와 속도를 비교하라.  
- xtensor를 설치하여 NumPy 스타일로 다차원 배열을 생성하고, 브로드캐스팅 및 `linalg::dot`으로 행렬곱 예제를 작성해 보라.  
- Boost.uBLAS로 선형 방정식 \(Ax=b\)를 풀어보라. (예: LU 분해 또는 역행렬을 이용)  
- 네 라이브러리의 단위 행렬 생성, 역행렬 계산, 고유값 분해 등의 API를 살펴보고 각 라이브러리의 장단점을 정리하라.  

**참고 자료:** 메모리 정렬과 캐시 최적화 ([Row- and column-major order - Wikipedia](https://en.wikipedia.org/wiki/Row-_and_column-major_order#:~:text=The%20difference%20between%20the%20orders,using%20this%20approach%20are%20effectively)) ([matrix multiplication - Cache friendly method to multiply two matrices - Stack Overflow](https://stackoverflow.com/questions/13312625/cache-friendly-method-to-multiply-two-matrices#:~:text=Basically%2C%20navigating%20the%20memory%20fastly,access%20index%20of%20the%20matrices)), 메모리 풀 개념 ([What is a Memory Pool? | GeeksforGeeks](https://www.geeksforgeeks.org/what-is-a-memory-pool/#:~:text=A%20memory%20pool%2C%20also%20known,that%20offers%20a%20number%20of)) ([What is a Memory Pool? | GeeksforGeeks](https://www.geeksforgeeks.org/what-is-a-memory-pool/#:~:text=,Improved%20performance%20and%20stability)), Eigen 공식 문서 ([Eigen](https://eigen.tuxfamily.org/index.php?title=Main_Page#:~:text=Eigen%20is%20a%20C%2B%2B%20template,numerical%20solvers%2C%20and%20related%20algorithms)) ([Eigen](https://eigen.tuxfamily.org/index.php?title=Main_Page#:~:text=,4%20MIPS%20MSA%20with)), Armadillo 소개 ([Armadillo: C++ library for linear algebra & scientific computing](https://arma.sourceforge.net/#:~:text=,speed%20and%20ease%20of%20use)) ([Armadillo: C++ library for linear algebra & scientific computing](https://arma.sourceforge.net/#:~:text=,to%20increase%20speed%20and%20efficiency)), xtensor 소개 ([Introduction — xtensor  documentation](https://xtensor.readthedocs.io/#:~:text=xtensor%20is%20a%20C%2B%2B%20library,dimensional%20array%20expressions)), Boost.uBLAS 문서 ([Boost Basic Linear Algebra - 1.88.0](https://www.boost.org/libs/numeric/ublas/doc/index.htm#:~:text=uBLAS%20is%20a%20C%2B%2B%20template,code%20generation%20via%20expression%20templates)).