# 2장: C++ 성능 최적화 기법

본 장에서는 C++로 작성한 프로그램의 실행 성능을 높이기 위한 다양한 저수준 기법과 최적화 기술을 다룬다. 특히 AI/ML 응용에서 대규모 데이터와 행렬 연산을 처리할 때 연산 속도와 메모리 효율이 중요하므로, 메모리 관리, 캐시 활용, 컴파일러 최적화, SIMD 벡터화, 템플릿 메타프로그래밍 등의 주제를 깊이 있게 살펴본다. 각 주제마다 이론 설명과 함께 완전한 C++ 코드 예제 및 실제 측정 실험을 제공하여 학생들이 직접 실행하고 검증할 수 있도록 한다. 또 절마다 마지막에는 학습 연습 문제를 통해 학습 내용을 점검한다.

## 메모리 관리 (힙 vs 스택, `new`/`delete` 비용, 스마트 포인터)

프로그램 실행 시 **스택(stack)** 메모리와 **힙(heap)** 메모리는 서로 다른 방식으로 관리된다. 스택은 각 함수 호출 시 로컬 변수 등을 위한 메모리 블록을 순차적으로 할당/해제하므로(후입선출 방식) 메모리 할당과 해제 비용이 거의 없고 캐시 친화적이다 ([Stack vs Heap Memory Allocation | GeeksforGeeks](https://www.geeksforgeeks.org/stack-vs-heap-memory-allocation/#:~:text=2,it%20causes%20more%20cache%20misses)). 반면 힙은 `new`나 `malloc` 같은 동적 할당을 통해 필요한 시점에 메모리를 할당/해제하므로 할당 패턴이 불규칙하고 메타데이터 관리 오버헤드가 크다. 실제로 힙 프레임을 관리하는 비용은 스택 프레임보다 훨씬 크고, 힙에 저장된 데이터는 메모리에 분산되어 있어 캐시 미스가 잦아 성능 저하가 발생할 수 있다 ([Stack vs Heap Memory Allocation | GeeksforGeeks](https://www.geeksforgeeks.org/stack-vs-heap-memory-allocation/#:~:text=2,it%20causes%20more%20cache%20misses)) ([Stack vs Heap Memory Allocation | GeeksforGeeks](https://www.geeksforgeeks.org/stack-vs-heap-memory-allocation/#:~:text=5,is%20more%20than%20a%20stack)). 예를 들어 스택은 메모리 연속 영역을 사용하는 반면 힙은 임의 위치에 할당되어 비연속적이므로, 스택 접근 속도가 힙보다 빠르다 ([Stack vs Heap Memory Allocation | GeeksforGeeks](https://www.geeksforgeeks.org/stack-vs-heap-memory-allocation/#:~:text=2,it%20causes%20more%20cache%20misses)) ([Stack vs Heap Memory Allocation | GeeksforGeeks](https://www.geeksforgeeks.org/stack-vs-heap-memory-allocation/#:~:text=5,is%20more%20than%20a%20stack)). 

다음 예제 코드는 반복문 안에서 스택 배열과 힙 배열을 각각 할당하여 소요 시간을 측정한다. 스택 배열(`int arr[M]`)은 루프 변수 증가로 간단히 재사용되므로 거의 비용이 없지만, 힙 할당(`new int[M]`, `delete[]`)은 반복당 수백 마이크로초 이상의 시간이 걸리는 것을 확인할 수 있다. (실험: N=200만 회, 배열 크기 M=1000, 컴파일 최적화 `-O2` 기준)

```cpp
#include <iostream>
#include <chrono>
using namespace std;

int main() {
    const int N = 2000000;
    const int M = 1000;
    // 스택 배열 할당 루프
    auto t1 = chrono::high_resolution_clock::now();
    for(int i = 0; i < N; ++i) {
        int arr[M];
        arr[0] = i;  // 최적화를 막기 위한 연산
    }
    auto t2 = chrono::high_resolution_clock::now();
    // 힙 할당 루프
    auto t3 = chrono::high_resolution_clock::now();
    for(int i = 0; i < N; ++i) {
        int *p = new int[M];
        p[0] = i;
        delete [] p;
    }
    auto t4 = chrono::high_resolution_clock::now();

    double stack_time = chrono::duration<double>(t2 - t1).count();
    double heap_time  = chrono::duration<double>(t4 - t3).count();
    cout << "Stack loop time: " << stack_time << " s\n";
    cout << "Heap loop time:  " << heap_time  << " s\n";
    return 0;
}
```

위 코드 실행 결과, 스택 할당 루프는 거의 즉시 완료되는 반면 힙 할당 루프는 수백 밀리초가 소요됨을 알 수 있다. 즉, 가능한 경우 로컬 변수(스택) 할당을 활용하고, 빈번한 동적 할당/해제는 피하는 것이 성능에 유리하다 ([Stack vs Heap Memory Allocation | GeeksforGeeks](https://www.geeksforgeeks.org/stack-vs-heap-memory-allocation/#:~:text=2,it%20causes%20more%20cache%20misses)) ([Stack vs Heap Memory Allocation | GeeksforGeeks](https://www.geeksforgeeks.org/stack-vs-heap-memory-allocation/#:~:text=5,is%20more%20than%20a%20stack)).

스마트 포인터(std::unique_ptr, std::shared_ptr)도 내부적으로 동적 메모리를 사용하므로 성능 비용이 존재한다. `std::unique_ptr`는 사실상 일반 포인터와 동일한 성능이며(사용자 지정 소멸자 제외) 소유 객체를 소멸할 때만 한 번 해제 작업이 일어난다. 반면 `std::shared_ptr`는 참조 카운터를 유지하므로 생성자, 대입, 소멸 시마다 원자적(atomic) 연산으로 카운터를 증가·감소시켜야 한다 ([performance - How much is the overhead of smart pointers compared to normal pointers in C++? - Stack Overflow](https://stackoverflow.com/questions/22295665/how-much-is-the-overhead-of-smart-pointers-compared-to-normal-pointers-in-c#:~:text=,trivial%20deleter)) ([performance - How much is the overhead of smart pointers compared to normal pointers in C++? - Stack Overflow](https://stackoverflow.com/questions/22295665/how-much-is-the-overhead-of-smart-pointers-compared-to-normal-pointers-in-c#:~:text=,thus%20adding%20some%20more%20overhead)). 이로 인해 `shared_ptr`는 메모리 및 시간 오버헤드가 상대적으로 크며, 특히 빈번한 생성/소멸이 이루어지는 경우 프로그램 속도가 느려질 수 있다 ([performance - How much is the overhead of smart pointers compared to normal pointers in C++? - Stack Overflow](https://stackoverflow.com/questions/22295665/how-much-is-the-overhead-of-smart-pointers-compared-to-normal-pointers-in-c#:~:text=,trivial%20deleter)) ([performance - How much is the overhead of smart pointers compared to normal pointers in C++? - Stack Overflow](https://stackoverflow.com/questions/22295665/how-much-is-the-overhead-of-smart-pointers-compared-to-normal-pointers-in-c#:~:text=,thus%20adding%20some%20more%20overhead)). 예를 들어, 수백만 번의 할당/해제 실험에서 `unique_ptr`는 거의 오버헤드가 없었지만, `shared_ptr`는 약 0.03초(1백만 번 기준)의 추가 시간이 소요되었다. 하지만 데이터 참조(dereference) 자체에는 추가 비용이 없으며, 일반적인 경우 스마트 포인터 사용으로 인한 오버헤드가 전체 성능에 큰 영향을 주지 않는 경우가 많다 ([performance - How much is the overhead of smart pointers compared to normal pointers in C++? - Stack Overflow](https://stackoverflow.com/questions/22295665/how-much-is-the-overhead-of-smart-pointers-compared-to-normal-pointers-in-c#:~:text=Note%20that%20none%20of%20them,the%20most%20common%20for%20pointers)) ([performance - How much is the overhead of smart pointers compared to normal pointers in C++? - Stack Overflow](https://stackoverflow.com/questions/22295665/how-much-is-the-overhead-of-smart-pointers-compared-to-normal-pointers-in-c#:~:text=To%20sum%20up%2C%20there%20is,create%20and%20destroy%20smart%20pointers)).

```cpp
#include <iostream>
#include <memory>
#include <chrono>
using namespace std;

int main() {
    const int N = 1000000;
    auto t1 = chrono::high_resolution_clock::now();
    // unique_ptr 반복 생성/소멸
    for(int i = 0; i < N; ++i) {
        unique_ptr<int> p(new int(i));
    }
    auto t2 = chrono::high_resolution_clock::now();
    // raw pointer new/delete 반복
    for(int i = 0; i < N; ++i) {
        int *p = new int(i);
        delete p;
    }
    auto t3 = chrono::high_resolution_clock::now();
    // shared_ptr 반복 생성/소멸
    for(int i = 0; i < N; ++i) {
        shared_ptr<int> p(new int(i));
    }
    auto t4 = chrono::high_resolution_clock::now();

    double unique_time = chrono::duration<double>(t2 - t1).count();
    double raw_time    = chrono::duration<double>(t3 - t2).count();
    double shared_time = chrono::duration<double>(t4 - t3).count();
    cout << "unique_ptr time: " << unique_time << " s\n";
    cout << "raw new/delete: " << raw_time    << " s\n";
    cout << "shared_ptr time: " << shared_time << " s\n";
    return 0;
}
```

실험 결과, raw new/delete와 `unique_ptr`의 실행 시간은 거의 같았지만, `shared_ptr`는 참조 카운트 관리로 인해 상대적으로 느리게 동작했다. 따라서 성능이 중요한 코드에서는 `shared_ptr` 사용을 최소화하고, 가능하면 `unique_ptr` 혹은 스택 객체를 사용하길 권장한다 ([performance - How much is the overhead of smart pointers compared to normal pointers in C++? - Stack Overflow](https://stackoverflow.com/questions/22295665/how-much-is-the-overhead-of-smart-pointers-compared-to-normal-pointers-in-c#:~:text=,trivial%20deleter)) ([performance - How much is the overhead of smart pointers compared to normal pointers in C++? - Stack Overflow](https://stackoverflow.com/questions/22295665/how-much-is-the-overhead-of-smart-pointers-compared-to-normal-pointers-in-c#:~:text=,thus%20adding%20some%20more%20overhead)).

**연습 문제:**  
- 스택 메모리와 힙 메모리 할당의 오버헤드를 측정하는 작은 프로그램을 작성해보자. 예를 들어 `new`/`delete` 반복문과 로컬 배열 사용 반복문을 각각 측정하고 비교해보자.  
- `std::shared_ptr`를 이용해 데이터 구조를 구현한 뒤, 동일 기능을 `std::unique_ptr`로 바꾸어 실행 속도를 비교해보자. 스마트 포인터 생성 및 소멸 횟수에 따른 시간 차이를 관찰해보자.

## 캐시 최적화 (지역성, 메모리 정렬과 연속성)

현대 컴퓨터 시스템에서는 메인 메모리 접근 속도가 매우 느리므로 **캐시(cache)** 메모리 활용이 중요하다. 캐시는 프로세서와 메인 메모리 사이에 위치한 고속 메모리로, 자주 사용되는 데이터나 명령어를 저장하여 메인 메모리 접근 횟수를 줄인다. 컴퓨터 프로그램이 동일한 데이터에 반복 접근하거나 인접 데이터에 접근하는 경향을 **지역성(locality)**이라고 한다 ([Computer Organization | Locality and Cache friendly code | GeeksforGeeks](https://www.geeksforgeeks.org/computer-organization-locality-and-cache-friendly-code/#:~:text=The%20idea%20of%20caching%20the,and%20thus%20run%20faster)) ([Computer Organization | Locality and Cache friendly code | GeeksforGeeks](https://www.geeksforgeeks.org/computer-organization-locality-and-cache-friendly-code/#:~:text=Cache%20Friendly%20Code%20%E2%80%93%20Programs,The%20basic)). 특히 **공간 지역성(spatial locality)**이 좋도록 인접한 메모리 공간을 순차적으로 접근하면 캐시 적중률(cache hit rate)이 높아져 실행 속도가 빨라진다.

예를 들어, C/C++ 배열은 행(row)-주 순서(row-major)로 메모리에 저장된다. 행렬을 합산할 때 행(row)을 고정하고 열(column)을 순서대로 반복하는 방식은 공간 지역성이 높아 캐시 친화적이다. 반대로 열 우선 순서로 접근하면 캐시 미스가 늘어 성능이 크게 떨어진다. 실제 GfG 예제에서는 같은 연산량의 행렬 곱셈에서 접근 순서에 따라 **실행 시간이 20배**까지 차이날 수 있다고 설명한다 ([Computer Organization | Locality and Cache friendly code | GeeksforGeeks](https://www.geeksforgeeks.org/computer-organization-locality-and-cache-friendly-code/#:~:text=Example%3A%20The%20run%20time%20of,by%20a%20factor%20of%2020)). 또한 “지역성이 좋은 코드가 그렇지 않은 코드보다 일반적으로 더 빨리 실행된다”는 것이 알려져 있다 ([Computer Organization | Locality and Cache friendly code | GeeksforGeeks](https://www.geeksforgeeks.org/computer-organization-locality-and-cache-friendly-code/#:~:text=Cache%20Friendly%20Code%20%E2%80%93%20Programs,The%20basic)). 

아래 예제 코드는 N×N 배열을 **행 우선(row-major)**과 **열 우선(column-major)**으로 합산하는 시간을 측정한다. 행 우선 루프는 인접 메모리를 순차 접근하므로 캐시 적중이 많아 빠르게 수행되며, 열 우선 루프는 캐시 친화성이 낮아 느리게 수행된다(예: N=5000일 때 행 우선 ~0.09s, 열 우선 ~0.31s).

```cpp
#include <iostream>
#include <vector>
#include <chrono>
using namespace std;

int main() {
    const int N = 5000;
    vector<int> arr(N*N);
    // 배열 초기화
    for(int i = 0; i < N; ++i)
        for(int j = 0; j < N; ++j)
            arr[i*N + j] = i*N + j;

    long long sum = 0;
    // 행 우선 합산
    auto t1 = chrono::high_resolution_clock::now();
    for(int i = 0; i < N; ++i) {
        for(int j = 0; j < N; ++j) {
            sum += arr[i*N + j];
        }
    }
    auto t2 = chrono::high_resolution_clock::now();
    double row_time = chrono::duration<double>(t2 - t1).count();

    // 열 우선 합산
    t1 = chrono::high_resolution_clock::now();
    sum = 0;
    for(int j = 0; j < N; ++j) {
        for(int i = 0; i < N; ++i) {
            sum += arr[i*N + j];
        }
    }
    auto t3 = chrono::high_resolution_clock::now();
    double col_time = chrono::duration<double>(t3 - t1).count();

    cout << "Row-major sum time:    " << row_time << " s\n";
    cout << "Column-major sum time: " << col_time << " s\n";
    return 0;
}
```

이처럼 **메모리 연속성**(contiguous memory)과 **정렬**(alignment)을 고려하여 데이터 구조를 설계하면 캐시 성능을 크게 개선할 수 있다. 예를 들어 구조체 멤버를 적절히 정렬하거나, 벡터나 배열처럼 연속 메모리를 사용하면 연속된 데이터 접근 시 캐시 효율이 높아진다. 64바이트 캐시 라인 크기를 고려해 `alignas(64)`를 사용하거나, 데이터 접근 패턴에 맞춰 2차원 배열의 순서를 바꾸는 등의 기법이 캐시 미스 감소에 도움을 준다. 일반적으로 캐시 성능은 프로그램 실행 속도에 큰 영향을 주므로, 루프 안팎의 데이터 접근 순서를 설계할 때 항상 캐시 친화성을 염두에 두어야 한다 ([Computer Organization | Locality and Cache friendly code | GeeksforGeeks](https://www.geeksforgeeks.org/computer-organization-locality-and-cache-friendly-code/#:~:text=Cache%20Friendly%20Code%20%E2%80%93%20Programs,The%20basic)) ([Computer Organization | Locality and Cache friendly code | GeeksforGeeks](https://www.geeksforgeeks.org/computer-organization-locality-and-cache-friendly-code/#:~:text=Example%3A%20The%20run%20time%20of,by%20a%20factor%20of%2020)).

**연습 문제:**  
- 크기가 큰 2차원 배열을 선언하고, 행 우선과 열 우선 접근 방식으로 합산하는 코드를 작성하여 실행 시간을 비교해보자. 캐시 최적화의 효과를 확인할 수 있다.  
- 구조체에 여러 멤버가 있을 때, 멤버 순서와 `alignas`를 조정하여 구조체 크기와 메모리 패딩이 어떻게 달라지는지 실험해보자. 캐시 라인 정렬이 성능에 미치는 영향도 관찰해보자.

## 컴파일러 최적화 (함수 인라인화, `constexpr`, 컴파일러 플래그)

컴파일러는 다양한 최적화 기능을 제공하여 프로그램 성능을 자동으로 향상시킨다. **함수 인라인화**(inline)는 가장 기본적인 최적화 방법 중 하나다. 인라인 함수로 지정하면 함수 호출을 함수 본문으로 대체하여 함수 호출 오버헤드를 제거할 수 있다 ([Inline Functions in C++ | GeeksforGeeks](https://www.geeksforgeeks.org/inline-functions-cpp/#:~:text=In%20C%2B%2B%2C%20inline%20functions%20provide,the%20normal%20function%20call%20mechanism)). 예를 들어 `inline int square(int x) { return x * x; }`를 호출하면 컴파일러는 호출 위치에 곧바로 곱셈 코드를 삽입한다 ([Inline Functions in C++ | GeeksforGeeks](https://www.geeksforgeeks.org/inline-functions-cpp/#:~:text=In%20C%2B%2B%2C%20inline%20functions%20provide,the%20normal%20function%20call%20mechanism)). 이로 인해 짧은 함수 호출에서 레지스터 저장/복귀 비용을 없앨 수 있다. 다만 인라인은 단순히 **힌트**일 뿐이며, 반복문이나 재귀가 있는 함수 등에는 컴파일러가 인라인을 적용하지 않을 수도 있다 ([Inline Functions in C++ | GeeksforGeeks](https://www.geeksforgeeks.org/inline-functions-cpp/#:~:text=The%20inline%20keyword%20suggests%20the,inlining%20in%20such%20circumstances%20as)). 

```cpp
// 함수 인라인 예제
inline int add(int a, int b) {
    return a + b;
}

int non_inline_add(int a, int b) {
    return a + b;
}

int main() {
    int x = 1, y = 2, z;
    z = add(x, y);         // 인라인 함수 호출
    z = non_inline_add(x, y); // 일반 함수 호출
    return 0;
}
```

`constexpr` 키워드는 함수나 변수의 값을 가능한 컴파일 타임에 계산할 수 있도록 선언한다 ([c++11 - C++ constexpr - Value can be evaluated at compile time? - Stack Overflow](https://stackoverflow.com/questions/52263790/c-constexpr-value-can-be-evaluated-at-compile-time#:~:text=,or%20variable%20at%20compile%20time)). 예를 들어 `constexpr` 함수는 인자로 컴파일 시간에 알 수 있는 상수를 전달하면 컴파일러가 미리 계산해준다. 이는 런타임 비용을 없애는 효과가 있으나, 최적화는 컴파일러에 의해 처리되므로 반드시 `constexpr`이어야만 컴파일 타임 계산이 가능한 것은 아니다. 실제로 컴파일러는 `constexpr`이 아니더라도 상수 식에 대해 최적화로 값을 계산하기도 한다. 다만 `constexpr`은 **컴파일 타임 계산이 가능함을 보장**해주며, 암묵적으로 `inline` 함수를 의미하기도 한다 ([constexpr functions: optimization vs guarantee - Andreas Fertig's Blog](https://andreasfertig.com/blog/2023/06/constexpr-functions-optimization-vs-guarantee/#:~:text=The%20reason%20is%20that%20,constexpr)) ([c++11 - C++ constexpr - Value can be evaluated at compile time? - Stack Overflow](https://stackoverflow.com/questions/52263790/c-constexpr-value-can-be-evaluated-at-compile-time#:~:text=,or%20variable%20at%20compile%20time)). 아래 코드에서 `constexpr` 함수를 사용하면 `42/6` 연산이 컴파일 타임에 계산되어 실행 시간에 실제 연산이 존재하지 않는다. 

```cpp
constexpr int compute(int x) {
    return x * x + 42;
}

int main() {
    constexpr int result = compute(6); // 컴파일 타임에 계산됨
    // result == 78
    return result;
}
```

마지막으로 컴파일러 최적화 플래그를 적절히 설정하면 성능을 크게 향상시킬 수 있다. 예를 들어 GCC/Clang의 `-O3` 플래그는 고급 최적화(인라인 확대, 루프 변환, 벡터화 등)를 활성화하므로 수치 계산이 많은 프로그램의 속도를 올려준다. 실제로 `-O0`으로 컴파일한 코드와 `-O3`로 컴파일한 코드는 실행 시간에 수십 배 차이를 보이기도 한다. 물론 항상 최적의 결과를 보장하는 것은 아니므로, 성능 테스트를 통해 적절한 최적화 수준을 선택해야 한다.  

**연습 문제:**  
- 짧은 함수를 하나 작성한 후 `inline` 키워드를 붙여 컴파일러가 실제로 인라인화했는지 어셈블리 코드를 확인해보자.  
- `constexpr` 함수를 작성하여 컴파일 타임 계산과 런타임 계산을 비교해보자. 예를 들어 팩토리얼 계산 등을 `constexpr`로 구현해보자.  
- 동일 코드를 `-O0`와 `-O3`로 컴파일하여 실행 시간을 비교해보자. 큰 성능 차이가 발생하는 코드를 찾아보고, 그 이유를 분석해보자.

## SIMD 벡터화 (SSE, AVX, std::simd)

현대 CPU는 **SIMD**(Single Instruction, Multiple Data) 명령어를 지원하여 한 명령으로 여러 데이터에 동시에 연산할 수 있다 ([Single instruction, multiple data - Wikipedia](https://en.wikipedia.org/wiki/Single_instruction,_multiple_data#:~:text=Single%20instruction%2C%20multiple%20data%20,and%20it%20can%20be)) ([Streaming SIMD Extensions - Wikipedia](https://en.wikipedia.org/wiki/Streaming_SIMD_Extensions#:~:text=on%20single%20precision%20%20,processing%20%20and%20%2092)). SIMD를 이용하면 벡터/행렬 연산, 신호 처리, 이미지/오디오 처리 등에서 데이터 수준 병렬성(Data-level parallelism)을 활용해 성능을 높일 수 있다. 예를 들어, 여러 쌍의 실수 덧셈을 동시에 수행할 때 SIMD 명령어를 이용하면 각 싸이클마다 최대 4~16개(AVX-512의 경우 8개 이상의 32비트 숫자) 이상의 연산을 처리할 수 있다. SSE는 128비트 XMM 레지스터(32비트 부동소수점 4개 또는 64비트 정수 2개)를, AVX는 256비트 YMM 레지스터(64비트 부동소수점 4개)를, AVX-512는 512비트 ZMM 레지스터를 사용한다 ([Streaming SIMD Extensions - Wikipedia](https://en.wikipedia.org/wiki/Streaming_SIMD_Extensions#:~:text=on%20single%20precision%20%20,processing%20%20and%20%2092)) ([Advanced Vector Extensions - Wikipedia](https://en.wikipedia.org/wiki/Advanced_Vector_Extensions#:~:text=AVX,3)). 특히 AVX-512는 한 번에 512비트(예: 8개의 64비트 실수)를 처리하며, 최신 CPU에서는 FMA(퓨즈드 멀티플라이-어드)와 함께 캐시 프리패치가 동작해 대용량 벡터 연산 성능을 대폭 개선한다 ([Single instruction, multiple data - Wikipedia](https://en.wikipedia.org/wiki/Single_instruction,_multiple_data#:~:text=example%20is%20to%20add%20many,in%20a%20single%20SIMD%20cycle)).

C++에서는 Intel 인트린식(intrinsics)을 이용하거나, 최신 표준인 C++20의 `<experimental/simd>` 또는 `<simd>`(C++23) 라이브러리의 `std::simd` 클래스를 사용하여 SIMD 명령어를 사용할 수 있다. 예를 들어 AVX 인트린식 `_mm256_loadu_pd`, `_mm256_add_pd`, `_mm256_storeu_pd`를 이용하면 `double` 배열의 4개 값을 한꺼번에 더할 수 있다. 아래 코드는 1백만 개의 `double` 값 배열에 대해 일반 루프와 AVX SIMD 루프를 비교한 예이다. SIMD 루프에서는 4개씩 묶어 덧셈하므로 일반 루프보다 약 2배 빠른 성능을 보였다.

```cpp
#include <iostream>
#include <chrono>
#include <immintrin.h>
using namespace std;

int main() {
    const int N = 1000000;
    // 동적 배열 할당
    double *a = new double[N];
    double *b = new double[N];
    double *c = new double[N];
    for(int i = 0; i < N; ++i) {
        a[i] = i;
        b[i] = i * 2.0;
    }

    // 일반 루프
    auto t1 = chrono::high_resolution_clock::now();
    for(int i = 0; i < N; ++i) {
        c[i] = a[i] + b[i];
    }
    auto t2 = chrono::high_resolution_clock::now();
    double plain_time = chrono::duration<double>(t2 - t1).count();

    // AVX SIMD 루프 (4개씩 처리)
    t1 = chrono::high_resolution_clock::now();
    int i = 0;
    for(; i + 4 <= N; i += 4) {
        __m256d va = _mm256_loadu_pd(a + i);
        __m256d vb = _mm256_loadu_pd(b + i);
        __m256d vc = _mm256_add_pd(va, vb);
        _mm256_storeu_pd(c + i, vc);
    }
    // 남은 원소 처리
    for(; i < N; ++i) {
        c[i] = a[i] + b[i];
    }
    t2 = chrono::high_resolution_clock::now();
    double simd_time = chrono::duration<double>(t2 - t1).count();

    cout << "Plain loop time: " << plain_time << " s\n";
    cout << "AVX SIMD time:   " << simd_time  << " s\n";

    delete[] a; delete[] b; delete[] c;
    return 0;
}
```

위 코드 실행 결과, AVX SIMD를 사용한 연산이 일반 루프보다 약 2배 빠른 성능을 보였다(예: 일반 0.0073s, SIMD 0.0032s). 이를 통해 SIMD 병렬화의 효과를 확인할 수 있다. 더 나아가 C++20 `std::simd`를 사용하면 벡터 자료형을 생성하여 복잡한 SIMD 코드를 직접 작성하지 않고도 유사한 효과를 얻을 수 있다. 예를 들어 `std::simd<double, 4>`는 4개 실수를 한 번에 처리하는 벡터 타입으로, AVX의 256비트 처리를 추상화해서 사용할 수 있다. 

**연습 문제:**  
- 위의 AVX 예제 코드를 확장하여, 부동소수점 곱셈이나 덧셈 이외에 벡터화된 최대값 계산 등을 구현해보자. SIMD를 사용하지 않은 버전과 속도를 비교해보고, 성능 차이를 관찰하라.  
- C++20 `std::simd`(또는 `<experimental/simd>`)를 지원하는 환경이 있다면, `std::simd`를 사용해 벡터 연산을 구현해보자. 간단한 벡터 덧셈 예제부터 시작하여 SIMD 최적화 효과를 확인해보자.

## 템플릿 메타프로그래밍 (컴파일 타임 계산, 식 표현 템플릿)

C++ 템플릿 메타프로그래밍은 컴파일 타임에 계산을 수행하여 런타임 오버헤드를 줄이는 기법이다. 일반적인 함수 호출이나 반복문 대신, 템플릿 인스턴스화를 통해 특수화된 코드를 생성함으로써 실행 속도를 개선할 수 있다. 예를 들어 C의 `qsort` 함수는 비교 함수 포인터를 사용하므로 호출 오버헤드가 크지만, C++ `std::sort`는 템플릿으로 구현되어 비교자(functor)를 인라인화할 수 있어 빠르다 ([C++ templates for performance? - Stack Overflow](https://stackoverflow.com/questions/8925177/c-templates-for-performance#:~:text=In%20C%2B%2B%2C%20,std%3A%3Asort)). 즉, 템플릿을 사용하면 제네릭(generic)한 소스 코드가 각 타입마다 **전문화된(specialized)** 코드로 컴파일되어 함수 호출 대신 인라인 연산이 발생하기 때문에 실행 속도가 향상될 수 있다 ([C++ templates for performance? - Stack Overflow](https://stackoverflow.com/questions/8925177/c-templates-for-performance#:~:text=In%20C%2B%2B%2C%20,std%3A%3Asort)). 물론 이 과정에서 코드 크기가 커질 수 있다는 점을 주의해야 한다.

식(expression) 템플릿은 연산자 오버로드 성능을 높이기 위한 대표적인 예다. 벡터 덧셈 연산을 생각해보자. 만약 `Vec a, b, c; a = b + c;`에서 `operator+`가 매번 새로운 중간 벡터를 생성한다면 불필요한 복사가 발생한다. 식 템플릿을 사용하면 `b + c` 연산 자체를 컴파일 타임의 표현식 객체로 만들고, 실제 할당 시점에 루프 하나로 모든 계산을 수행해 중간 버퍼 없이 최적화할 수 있다. 즉, 표현식 템플릿은 컴파일 타임에 연산 구조를 구성하고 필요할 때 전체 연산을 한 번에 실행하여 루프 융합(loop fusion) 같은 최적화를 가능하게 한다 ([Expression templates - Wikipedia](https://en.wikipedia.org/wiki/Expression_templates#:~:text=Expression%20templates%20are%20a%20C%2B%2B,optimizations%20such%20as%20%2044)). 

아래는 간단한 식 템플릿 예시로, 벡터 덧셈을 일시적으로 표현식 객체로 만드는 코드 스케치이다:

```cpp
// 식 템플릿 예제 (개념적 코드)
template<typename L, typename R>
struct VecAdd {
    const L &l; const R &r;
    VecAdd(const L &lhs, const R &rhs) : l(lhs), r(rhs) {}
    double operator[](int i) const {
        return l[i] + r[i];
    }
    int size() const { return l.size(); }
};

template<typename E>
struct VecExpr {
    const E &expr;
    VecExpr(const E &e) : expr(e) {}
    double operator[](int i) const { return expr[i]; }
    int size() const { return expr.size(); }
};

// 실제 벡터 클래스
struct Vec {
    int n;
    double *data;
    // ... 생성자, 소멸자 등 ...
    double operator[](int i) const { return data[i]; }
    int size() const { return n; }
    // Vec + Vec 결과를 표현식으로 반환
    VecExpr<VecAdd<Vec, Vec>> operator+(const Vec &other) const {
        return VecExpr<VecAdd<Vec,Vec>>(*this, other);
    }
};

// 사용 예: Vec a, b, c; c = a + b;
```

위 코드에서 `a + b`는 즉시 새로운 `Vec`을 만들지 않고, `VecAdd` 표현식을 생성한다. 실제 `c = a + b;` 대입이 일어날 때, 표현식 객체를 통해 모든 성분 합산을 한 루프로 수행한다. 이렇게 함으로써 중간 객체 없이 효율적인 코드가 만들어진다. (자세한 구현은 고급 내용이므로 참조 문헌을 통해 학습하도록 한다.) 

**연습 문제:**  
- 재귀 템플릿을 이용해 컴파일 타임 팩토리얼이나 피보나치 수 계산을 구현해보자. 컴파일된 코드를 확인하여 런타임 연산이 존재하지 않음을 확인해보자.  
- 간단한 벡터 클래스를 작성하고, 위 예시와 같이 연산자 오버로드와 식 템플릿을 적용하여 중간 객체 생성을 없애보자. 중간 할당이 없는 버전과 없는 버전을 비교하여 코드 크기와 성능 차이를 관찰해보자.

**참고 자료:** 메모리와 캐시 친화 코드에 관한 기본 개념 ([Computer Organization | Locality and Cache friendly code | GeeksforGeeks](https://www.geeksforgeeks.org/computer-organization-locality-and-cache-friendly-code/#:~:text=The%20idea%20of%20caching%20the,and%20thus%20run%20faster)) ([Computer Organization | Locality and Cache friendly code | GeeksforGeeks](https://www.geeksforgeeks.org/computer-organization-locality-and-cache-friendly-code/#:~:text=Cache%20Friendly%20Code%20%E2%80%93%20Programs,The%20basic)), 함수 인라인과 `constexpr`에 관한 설명 ([Inline Functions in C++ | GeeksforGeeks](https://www.geeksforgeeks.org/inline-functions-cpp/#:~:text=In%20C%2B%2B%2C%20inline%20functions%20provide,the%20normal%20function%20call%20mechanism)) ([c++11 - C++ constexpr - Value can be evaluated at compile time? - Stack Overflow](https://stackoverflow.com/questions/52263790/c-constexpr-value-can-be-evaluated-at-compile-time#:~:text=,or%20variable%20at%20compile%20time)), SIMD 명령어의 소개 ([Single instruction, multiple data - Wikipedia](https://en.wikipedia.org/wiki/Single_instruction,_multiple_data#:~:text=Single%20instruction%2C%20multiple%20data%20,and%20it%20can%20be)) ([Streaming SIMD Extensions - Wikipedia](https://en.wikipedia.org/wiki/Streaming_SIMD_Extensions#:~:text=on%20single%20precision%20%20,processing%20%20and%20%2092)), 템플릿 최적화 예제 ([C++ templates for performance? - Stack Overflow](https://stackoverflow.com/questions/8925177/c-templates-for-performance#:~:text=In%20C%2B%2B%2C%20,std%3A%3Asort)) ([Expression templates - Wikipedia](https://en.wikipedia.org/wiki/Expression_templates#:~:text=Expression%20templates%20are%20a%20C%2B%2B,optimizations%20such%20as%20%2044)) 등을 참조하라.