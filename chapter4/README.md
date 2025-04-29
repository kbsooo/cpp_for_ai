# Modern C++ 기반 CPU 병렬 프로그래밍

## 1. 서론  
현대의 CPU는 여러 개의 코어를 내장하고 있으며, 하나의 프로세서 내에서 다수의 스레드를 병렬로 실행할 수 있다. 과거에는 병렬 컴퓨팅이 주로 고성능 컴퓨팅 분야에 한정되었으나, 멀티코어 CPU/GPU의 보급으로 일반 PC에서도 병렬 프로그래밍의 이점을 활용할 수 있게 되었다 ([
		
			
			
				Performance Comparison of Parallel Programming Frameworks in Digital Image Transformation
				-International Journal of Internet, Broadcasting and Communication
			
		
	 | Korea Science](http://koreascience.or.kr/article/JAKO201925462477925.page#:~:text=Previously%2C%20parallel%20computing%20was%20mainly,major%20parallel%20programming%20frameworks%2C%20and)). C++11 이후에는 `std::thread`, `std::async` 같은 표준 라이브러리 기능을 통해 손쉽게 스레드를 생성하여 CPU의 병렬 처리를 구현할 수 있다. 이를 통해 계산 집약적인 작업(예: 행렬 연산, 대규모 데이터 처리 등)의 실행 시간을 단축하고, 응용 프로그램의 처리량과 반응성을 향상시킬 수 있다. 또한 CPU에서는 스레드들이 하나의 메모리 공간을 공유하므로, 메모리 접근이 비교적 자유롭다는 장점이 있다. 

CPU 병렬 프로그래밍의 핵심 동기는 **동시성(concurrency)**과 **병렬성(parallelism)**의 활용이다. 다수의 작업을 동시에 진행(concurrency)함으로써 실제로 각 코어에서 병렬로 연산(parallelism)하게 되어 전체 처리 시간이 줄어든다. 그러나 스레드 간에 자원을 공유할 경우 데이터 경합(race condition)이 발생하거나 교착 상태(deadlock) 같은 문제도 함께 발생할 수 있으므로, 올바른 동기화 기법이 필요하다. 이 장에서는 Modern C++(C++11 이상)에서 제공하는 다양한 병렬 프로그래밍 도구(`std::thread`, `std::async/future`, OpenMP, Intel oneTBB 등)와 그 사용법을 살펴보고, 병렬 프로그래밍 시 주의해야 할 문제를 함께 다룬다. 

## 2. `std::thread`  
`std::thread`는 C++11부터 도입된 스레드 객체로, 함수를 새로운 스레드에서 실행할 수 있게 해준다. 스레드를 생성하려면 함수 또는 함수 객체를 인자로 전달하여 `std::thread` 객체를 생성하면 된다. 생성된 스레드는 **join** 또는 **detach**를 호출해야 한다. `join()`을 호출하면 해당 스레드가 끝날 때까지 현재 스레드가 대기하고, `detach()`를 호출하면 스레드를 분리(detach)하여 독립 실행하게 된다. 분리된 스레드는 별도로 관리하지 않으므로, 메인 스레드가 끝나더라도 독립적으로 실행되며, 메인 스레드 종료 시점에 프로세스가 종료되어 버그가 될 수 있으니 주의해야 한다.  

- **스레드 생성 예시:** 다음 코드는 별도 스레드를 생성하여 함수 `hello()`를 실행하고, `join()`을 호출하여 종료를 기다린다.  
  ```cpp
  #include <iostream>
  #include <thread>
  
  void hello() {
      std::cout << "Hello from thread!\n";
  }
  
  int main() {
      std::thread t(hello);  // 새 스레드 생성
      t.join();             // 스레드 종료 대기
      std::cout << "Thread has finished.\n";
      return 0;
  }
  ```  
  위 코드는 컴파일 시 `-pthread` 옵션을 지정해야 한다. 예를 들어 `g++ -std=c++11 -pthread thread_example.cpp -o thread_example`로 빌드하고, `./thread_example`로 실행할 수 있다.  

- **`join` vs `detach`:** `join()`을 호출하지 않고 스레드 객체가 범위를 벗어나면 프로그램이 예외를 발생시킨다. 반면 `detach()`를 호출하면 스레드가 분리되어 백그라운드에서 실행된다. 예를 들어:  
  ```cpp
  std::thread t(func);
  t.detach(); // 스레드를 분리
  // 이 시점부터 t 스레드는 메인 스레드와 독립적으로 실행된다.
  ```  
  스레드를 분리한 후에는 `join()`할 수 없으며, 스레드 관리가 어려워지므로 일반적으로는 `join()`을 사용하여 명시적으로 종료를 기다리는 것이 안전하다.

### 공유 자원 보호(`std::mutex`)  
스레드가 동일한 공유 자원(예: 전역 변수, 자료구조 등)에 동시에 접근하면 데이터 경합(race condition)이 발생하여 잘못된 결과나 예기치 못한 동작이 일어날 수 있다. C++에서는 `std::mutex`와 같은 상호배제(Mutex) 객체를 사용하여 한 번에 하나의 스레드만 자원에 접근하도록 보장할 수 있다 ([Mutex in C++ | GeeksforGeeks](https://www.geeksforgeeks.org/std-mutex-in-cpp/#:~:text=Mutex%20stands%20for%20Mutual%20Exclusion,of%20variables%2C%20data%20structures%2C%20etc)). 

- **뮤텍스 사용 예시:** 두 개의 스레드가 전역 변수 `counter`를 증가시키는 예제이다. 뮤텍스 없이 실행하면 잘못된 결과가 나올 수 있으나, 뮤텍스로 보호하면 올바르게 작동한다.  
  ```cpp
  #include <iostream>
  #include <thread>
  #include <mutex>
  
  std::mutex mtx;
  int counter = 0;
  
  void increment(int times) {
      for(int i = 0; i < times; ++i) {
          std::lock_guard<std::mutex> lock(mtx); // 뮤텍스 잠금
          ++counter;
      }
  }
  
  int main() {
      std::thread t1(increment, 1000000);
      std::thread t2(increment, 1000000);
      t1.join();
      t2.join();
      std::cout << "Counter: " << counter << "\n"; // 기대값: 2000000
      return 0;
  }
  ```  
  위 코드에서 `std::lock_guard<std::mutex>`를 사용하여 임계 영역을 자동으로 잠그고 해제한다. `std::mutex`는 공유 데이터를 여러 스레드가 동시에 접근하지 못하도록 보호하는 동기화 도구다 ([Mutex in C++ | GeeksforGeeks](https://www.geeksforgeeks.org/std-mutex-in-cpp/#:~:text=Mutex%20stands%20for%20Mutual%20Exclusion,of%20variables%2C%20data%20structures%2C%20etc)).  

### 교착 상태(Deadlock)와 방지  
여러 개의 뮤텍스를 순서대로 잠그는 경우, 서로 다른 순서로 잠금을 시도하면 **데드락(deadlock)**이 발생할 수 있다. 예를 들어, 스레드 A는 뮤텍스 `m1`를 얻은 뒤 `m2`를 얻으려고 하고, 동시에 스레드 B는 `m2`를 얻은 뒤 `m1`를 얻으려고 하면 두 스레드가 서로 상대 스레드가 해제할 때까지 기다려 프로그램이 교착된다 ([Avoiding deadlocks the C++ way](https://vorbrodt.blog/2019/10/12/avoiding-deadlocks-the-c-way/#:~:text=Often%20times%20a%20deadlock%20occurs,Nasty%20business%21%20A%20partial)).  

- **데드락 방지:** 하나의 방법은 항상 뮤텍스를 획득하는 순서를 고정하는 것이다(예: 먼저 `m1`를, 그 다음 `m2`를 잠근다). C++ 표준 라이브러리는 `std::lock()` 함수나 C++17부터 지원되는 `std::scoped_lock`을 제공한다. `std::lock(m1, m2)`는 두 뮤텍스를 동시에 잠그되 데드락이 발생하지 않도록 조정하며, `std::scoped_lock`는 내부적으로 `std::lock`을 사용하여 예외 안전하게 다수의 뮤텍스를 잠근다 ([Avoiding deadlocks the C++ way](https://vorbrodt.blog/2019/10/12/avoiding-deadlocks-the-c-way/#:~:text=unlock,implementation%20looks%20something%20like%20this)). 예를 들어:  
  ```cpp
  std::mutex m1, m2;
  void safeFunc() {
      std::scoped_lock lock(m1, m2); // m1과 m2를 안전하게 잠금
      // 임계 영역 작업
  }
  ```  
  위 코드에서 `std::scoped_lock`은 두 뮤텍스를 내부적으로 올바른 순서로 잠근 후, 함수 종료 시 자동으로 해제하므로 데드락 위험이 없다 ([Avoiding deadlocks the C++ way](https://vorbrodt.blog/2019/10/12/avoiding-deadlocks-the-c-way/#:~:text=unlock,implementation%20looks%20something%20like%20this)). 이 외에도 `std::lock`을 사용하여 `std::lock_guard`나 `std::unique_lock`과 조합하는 방법도 있다.

### 연습 문제  
1. **스레드 생성 및 join:** 두 개의 스레드를 생성하여 각각 서로 다른 메시지를 출력하고, 메인 스레드는 두 스레드가 종료될 때까지 기다리도록 프로그램을 작성하라.  
2. **경쟁 상태 해결:** 전역 변수 `counter`를 여러 스레드가 100만 번 증가시키는 프로그램을 작성하라. 우선 **뮤텍스 없이** 실행하여 결과가 기대값과 다른 경우를 확인하고, 그 후 **뮤텍스(`std::mutex`)를 사용**하여 올바른 결과를 얻도록 수정하라.  

## 3. `std::async`와 futures  
`std::async`는 비동기적으로 함수를 실행하고, 그 결과를 `std::future` 객체로 반환하는 편리한 기능이다. 예를 들어 `std::async(std::launch::async, func, args...)` 형태로 호출하면 함수 `func`가 별도 스레드에서 즉시 실행되고, 반환값은 `std::future`를 통해 나중에 가져올 수 있다. `std::future`는 `get()` 메서드로 결과를 얻거나, `wait()`로 완료를 기다릴 수 있다. `std::async` 호출 시 정책(std::launch)을 명시하지 않으면 구현체가 스레드를 새로 생성하거나 지연 실행(deferred) 모드 중 하나를 선택할 수 있다. 일반적으로 동기 실행을 원하지 않으면 `std::launch::async`를 사용한다.  

- **`std::async` 예시:** 다음 코드는 배열의 요소를 합산하는 작업을 비동기적으로 실행한다.  
  ```cpp
  #include <iostream>
  #include <future>
  #include <vector>
  
  int sum(const std::vector<int>& v) {
      int s = 0;
      for(int x : v) s += x;
      return s;
  }
  
  int main() {
      std::vector<int> data(1000);
      for(int i = 0; i < 1000; ++i) data[i] = i;
  
      // std::async를 사용하여 비동기 합계 계산
      std::future<int> fut = std::async(std::launch::async, sum, std::cref(data));
      // 다른 작업을 수행할 수 있다.
  
      int result = fut.get();  // 합계 결과를 가져옴
      std::cout << "Sum = " << result << "\n";
      return 0;
  }
  ```  
  위 코드를 `g++ -std=c++11 async_example.cpp -o async_example`로 빌드하고 실행하면, 별도 스레드에서 `sum` 함수가 실행되어 결과를 반환한다.

- **`std::promise`와 `std::future` 사용:** `std::promise`는 생산자(설정자) 스레드가 값을 설정하고, `std::future`는 소비자(수신자) 스레드가 그 값을 가져오도록 해준다. 이를 통해 스레드 간 안전하게 값을 전달할 수 있다 ([std::promise in C++ | GeeksforGeeks](https://www.geeksforgeeks.org/std-promise-in-cpp/#:~:text=std%3A%3Apromise%20is%20a%20class%20template,thereby%20defining%20the%20future%20references)). 예를 들어:  
  ```cpp
  #include <iostream>
  #include <thread>
  #include <future>
  
  void produce(std::promise<int> prom) {
      // 어떤 작업 수행 후 값 설정
      prom.set_value(42);
  }
  
  int main() {
      std::promise<int> prom;
      std::future<int> fut = prom.get_future();
  
      std::thread t(produce, std::move(prom));
      int value = fut.get();  // 프로미스에서 설정한 값 42를 가져옴
      t.join();
  
      std::cout << "Received value: " << value << "\n";
      return 0;
  }
  ```  
  이 예제에서 `produce` 함수는 `promise`를 전달받아 `set_value`로 값을 설정하고, 메인 스레드는 `future.get()`으로 그 값을 받아올 수 있다. `std::promise`는 한 스레드가 값을 약속(promise)하고 다른 스레드가 나중에 얻어오는 메커니즘을 제공한다 ([std::promise in C++ | GeeksforGeeks](https://www.geeksforgeeks.org/std-promise-in-cpp/#:~:text=std%3A%3Apromise%20is%20a%20class%20template,thereby%20defining%20the%20future%20references)).  

### 연습 문제  
1. **std::async 사용:** 주어진 벡터에 있는 정수들의 제곱 합을 계산하는 함수를 작성하고, `std::async`를 사용하여 이를 비동기로 수행하는 프로그램을 작성하라.  
2. **std::promise 사용:** `std::promise`와 `std::future`를 사용하여 두 개의 스레드 간에 메시지를 전달하는 프로그램을 작성하라. 한 스레드는 문자열을 생성하여 `std::promise<std::string>`에 설정하고, 메인 스레드는 `std::future`를 통해 해당 문자열을 출력한다.  

## 4. OpenMP  
OpenMP(Open Multi-Processing)는 C/C++/Fortran 코드에서 병렬 처리를 쉽게 구현할 수 있게 해주는 지시문(Direc­tive) 기반 라이브러리다. 반복문 등의 코드 앞에 `#pragma omp parallel for` 등의 지시문을 추가하면 컴파일러가 자동으로 병렬화한다. OpenMP를 사용하려면 `<omp.h>` 헤더를 포함하고, 컴파일 시 `-fopenmp` 플래그를 지정해야 한다. 예를 들어 `g++ -std=c++11 -fopenmp openmp_example.cpp -o openmp_example`로 빌드한다.  

- **`parallel for` 사용 예:** 배열 요소를 합산하는 코드를 OpenMP로 병렬화한다.  
  ```cpp
  #include <iostream>
  #include <omp.h>
  #include <vector>
  
  int main() {
      const int N = 100;
      std::vector<int> data(N);
      for(int i = 0; i < N; ++i) data[i] = i+1;
  
      int sum = 0;
      #pragma omp parallel for reduction(+:sum)
      for(int i = 0; i < N; ++i) {
          sum += data[i];
      }
      std::cout << "Sum = " << sum << "\n";
      return 0;
  }
  ```  
  위 코드에서 `reduction(+:sum)`은 `sum` 변수를 스레드마다 지역적으로 합산한 후 최종 결과를 결합해 준다. 이 코드는 멀티코어에서 병렬로 실행되어 `sum` 값을 계산한다.  

- **`reduction`과 `critical`:** `reduction`은 위 예시처럼 덧셈, 곱셈 등 단일 연산에 유용하다. 임계 영역을 직접 제어하려면 `#pragma omp critical`를 사용할 수 있다. 예를 들어 여러 스레드가 `count`를 증가시키는 코드는 다음과 같다.  
  ```cpp
  #include <iostream>
  #include <omp.h>
  
  int main() {
      int count = 0;
      #pragma omp parallel
      {
          #pragma omp critical
          {
              ++count;
              std::cout << "Thread " << omp_get_thread_num() << "\n";
          }
      }
      std::cout << "Count = " << count << "\n";
      return 0;
  }
  ```  
  여기서 `#pragma omp critical` 블록 내 코드는 한 번에 하나의 스레드만 실행하므로, `count`를 안전하게 갱신할 수 있다.  

### 연습 문제  
1. **OpenMP로 병렬화:** 1부터 1000까지의 합을 계산하는 프로그램을 작성하되, 반복문을 OpenMP `parallel for`로 병렬화하라.  
2. **OpenMP reduction 사용:** 크기가 1000인 벡터에 대해 원소들의 제곱 합을 계산하되, OpenMP의 `reduction`을 사용하여 병렬화하라.  

## 5. Intel oneTBB (최신 기준)  
Intel oneAPI Threading Building Blocks(oneTBB)는 Intel TBB의 최신 버전으로, 범용 C++ 병렬 알고리즘과 병렬 데이터 구조를 제공하는 라이브러리다. oneTBB는 이전 TBB와 소스 코드 수준에서는 거의 호환되지만, 바이너리 호환성은 보장하지 않으며 일부 인터페이스가 변경되었다 ([c++ - What is oneAPI and how does it compare to TBB? - Stack Overflow](https://stackoverflow.com/questions/69381876/what-is-oneapi-and-how-does-it-compare-to-tbb#:~:text=oneTBB%20is%20the%20next%20version,TBB%29%20and%20TBB%20Revamp)). 예를 들어 헤더 파일 경로가 `oneapi/tbb.h`로 바뀌었으며, `tbb::` 대신 `oneapi::tbb::` 네임스페이스를 사용한다. (기존 TBB 코드를 그대로 사용하려면 `tbb/tbb.h`도 제공되지만, 권장 방식은 `oneapi/tbb.h`를 사용하는 것이다 ([c++ - What is oneAPI and how does it compare to TBB? - Stack Overflow](https://stackoverflow.com/questions/69381876/what-is-oneapi-and-how-does-it-compare-to-tbb#:~:text=oneTBB%20is%20the%20next%20version,TBB%29%20and%20TBB%20Revamp)).) oneTBB는 내부적으로 태스크 기반 스케줄러를 사용하여 작업을 분할하고 코어에 할당한다.  

- **`parallel_for` 예시:** TBB의 `parallel_for`를 사용하여 벡터의 각 요소를 제곱하는 예제이다.  
  ```cpp
  #include <iostream>
  #include <vector>
  #include <oneapi/tbb/tbb.h>
  
  int main() {
      const int N = 1000;
      std::vector<int> data(N);
      for(int i = 0; i < N; ++i) data[i] = i;
  
      // oneTBB parallel_for
      oneapi::tbb::parallel_for(oneapi::tbb::blocked_range<int>(0, N),
          [&](const oneapi::tbb::blocked_range<int>& r) {
              for(int i = r.begin(); i < r.end(); ++i) {
                  data[i] = data[i] * data[i];
              }
          });
  
      // 결과 확인
      long long sum = 0;
      for(int x : data) sum += x;
      std::cout << "Sum of squares = " << sum << "\n";
      return 0;
  }
  ```  
  위 예제에서 `parallel_for`는 범위(range)를 적절히 분할하여 다수의 스레드에 작업을 분배한다. 빌드시에는 `-ltbb` 옵션이 필요하다(`g++ -std=c++11 tbb_for.cpp -ltbb -o tbb_for`).  

- **`parallel_reduce` 예시:** 병렬 누적 합을 구하는 예제이다. 아래 코드는 1부터 100까지의 합을 계산한다.  
  ```cpp
  #include <iostream>
  #include <oneapi/tbb/tbb.h>
  
  int main() {
      // oneTBB parallel_reduce를 사용하여 1부터 100까지 합 계산
      int sum = oneapi::tbb::parallel_reduce(
          oneapi::tbb::blocked_range<int>(1, 101), 0,
          [](const oneapi::tbb::blocked_range<int>& r, int init) {
              for(int i = r.begin(); i < r.end(); ++i)
                  init += i;
              return init;
          },
          [](int a, int b) {
              return a + b;
          });
      std::cout << "Sum = " << sum << "\n"; // 5050 출력
      return 0;
  }
  ```  
  이 코드는 내부적으로 범위를 나누어 각 부분합을 계산한 후 `Combiner` 함수(`[](int a,int b){return a+b;}`)로 합친다. 위 예제 역시 `-ltbb`로 컴파일한다.  

- **태스크 기반 모델:** oneTBB는 태스크 기반 프로그래밍을 지원한다. `oneapi::tbb::task_group`을 사용하면 임의의 작업을 병렬로 실행할 수 있다. 예를 들어 두 개의 태스크를 생성하여 실행하고 완료를 기다리는 코드는 다음과 같다.  
  ```cpp
  #include <iostream>
  #include <oneapi/tbb/task_group.h>
  
  int main() {
      oneapi::tbb::task_group tg;
      int result1 = 0, result2 = 0;
  
      tg.run([&](){ result1 = 10; });
      tg.run([&](){ result2 = 20; });
      tg.wait(); // 두 태스크 완료를 기다림
  
      std::cout << "Results: " << result1 << ", " << result2 << "\n";
      return 0;
  }
  ```  
  위 예제는 두 람다 함수를 태스크로 등록하고, `wait()`로 둘 다 완료될 때까지 대기한다. `task_group`을 사용하면 동적으로 태스크를 추가하고 관리할 수 있으며, 메인 스레드도 다른 태스크를 도울 수 있다.  

### 연습 문제  
1. **tbb::parallel_for 사용:** 길이 1000인 벡터를 생성한 뒤, 각 요소에 인덱스의 제곱 값을 대입하고 벡터의 합을 계산하는 프로그램을 `oneapi::tbb::parallel_for`를 사용하여 작성하라.  
2. **task_group 사용:** `oneapi::tbb::task_group`을 이용해 두 개의 태스크를 실행하라. 첫 번째 태스크는 0부터 49까지의 합을 계산하고, 두 번째 태스크는 50부터 99까지의 합을 계산하여, 각 결과를 출력하라.  

## 6. 병렬 프로그래밍의 어려움  
병렬 프로그래밍에서는 여러 스레드가 동시에 동작하기 때문에 특정 문제와 위험이 존재한다. 대표적으로 **경쟁 상태(race condition)**, **거짓 공유(false sharing)**, **캐시 일관성(cache coherence)** 문제가 있다.  

- **경쟁 상태(race condition):** 여러 스레드가 공유 변수에 접근하여 읽거나 쓸 때, 실행 순서에 따라 결과가 달라지는 현상이다. 위 `counter` 예제처럼 뮤텍스 없이 동시 접근하면 예상치 못한 값이 나올 수 있다. 간단한 해결책으로는 앞서 본 뮤텍스나 `std::atomic`를 사용하는 것이다. 예를 들어 `std::atomic<int> counter;`를 사용하면 별도의 락 없이도 동시 접근이 안전해진다. `std::atomic` 타입은 한 스레드가 값을 쓰고 다른 스레드가 읽어도 동작이 정의되며, 내부적으로 메모리 장벽 등을 사용해 원자성을 보장한다 ([std::atomic - cppreference.com](https://en.cppreference.com/w/cpp/atomic/atomic#:~:text=Each%20instantiation%20and%20full%20specialization,for%20details%20on%20data%20races)).  

- **거짓 공유(false sharing):** 서로 다른 스레드가 접근하는 변수가 물리적으로 같은 캐시 라인(cache line)에 위치할 때 발생한다. 예를 들어 두 스레드가 서로 다른 배열 요소(같은 캐시 라인에 있음)를 빈번히 수정하면, 캐시 일관성 프로토콜로 인해 불필요한 캐시 무효화가 발생하여 성능이 크게 저하될 수 있다. 이를 방지하려면 구조체나 배열 사이에 여유 공간(padding)을 넣거나, 스레드 로컬 변수로 분리하는 등의 기법을 사용한다.  

- **캐시 일관성:** 멀티코어 시스템에서 각 코어의 캐시에 같은 데이터의 복사본이 있을 수 있으며, 어떤 코어가 해당 데이터를 수정하면 다른 코어의 캐시는 불일치 상태가 된다. 이를 자동으로 해결하기 위해 캐시 일관성 프로토콜(MESI 등)이 동작하지만, 이 과정에서 메모리 지연이 발생할 수 있다. 따라서 스레드가 빈번히 공유 자원을 변경하면 캐시 동기화 비용으로 성능 병목이 생길 수 있다.  

이 외에도 디버깅의 어려움, 비결정적 행동(non-determinism) 등 병렬 프로그래밍 고유의 어려움이 많다. 따라서 처음부터 무분별하게 스레드를 생성하기보다, 필요한 부분을 신중하게 병렬화하고 동기화 기법을 적절히 사용하는 것이 중요하다.  

### 연습 문제  
1. 다음 코드에서 발생하는 경쟁 상태를 설명하고, `std::mutex`나 `std::atomic`를 사용하여 해결하라.  
   ```cpp
   #include <iostream>
   #include <thread>
   
   int counter = 0;
   void worker() {
       for(int i = 0; i < 1000000; ++i) {
           counter++;
       }
   }
   int main() {
       std::thread t1(worker);
       std::thread t2(worker);
       t1.join(); t2.join();
       std::cout << "Counter = " << counter << "\n";
       return 0;
   }
   ```  
2. 거짓 공유(false sharing)의 예를 들어보고, 이를 방지할 수 있는 방법을 설명하라.  

## 7. 정리 및 권장 학습 경로  
이 장에서는 Modern C++를 사용한 CPU 병렬 프로그래밍의 기초를 다루었다. `std::thread`와 동기화 도구(뮤텍스, 락 등)를 통해 스레드를 생성하고 안전하게 동작하도록 하는 방법을 살펴보았고, `std::async`와 `std::future/ std::promise`를 이용한 고수준 비동기 프로그래밍 기법을 익혔다. 또한 OpenMP 지시문을 통해 반복문의 자동 병렬화 방법과, Intel oneTBB를 이용한 병렬 알고리즘(예: `parallel_for`, `parallel_reduce`) 및 태스크 기반 모델을 소개하였다. 마지막으로 병렬 프로그래밍에서 흔히 마주치는 경쟁 상태, 거짓 공유, 캐시 일관성 문제 등을 설명하였다.

병렬 프로그래밍은 디버깅이 어렵고 복잡도가 높지만, 제대로 구현하면 어플리케이션의 성능을 크게 향상시킬 수 있다. 더 심도 있게 공부하려면 C++ 표준 라이브러리의 병렬 알고리즘(예: C++17의 병렬 STL), Boost Asio, Intel TBB 공식 문서, OpenMP 공식 사양 등을 참고하라. 또한 Threa­ding Library나 Concurrency 프로그래밍 서적(예: Anthony Williams의 *“C++ Concurrency in Action”* 등)을 통해 메모리 모델과 고급 동기화 기법을 학습하는 것이 도움이 된다. 본 장에서 소개한 예제와 연습 문제를 풀어보면서 개념을 체득한 뒤, 실제 성능 병렬화가 필요한 프로젝트에 적용해 보는 것이 권장 학습 경로다.