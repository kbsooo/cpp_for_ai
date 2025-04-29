# 현대적 C++ 문법

C++11부터 C++20까지의 현대적 C++는 다양한 새로운 기능을 도입하여 코드의 간결성과 안전성을 높였습니다. 그중 `auto` 키워드는 변수 선언 시 타입을 초기화식으로부터 **자동으로 추론**하도록 합니다 ([auto (C++) | Microsoft Learn](https://learn.microsoft.com/en-us/cpp/cpp/auto-cpp?view=msvc-170#:~:text=The%20,parameter%2C%20to%20deduce%20its%20type)). 이로 인해 복잡한 타입 이름을 일일이 쓰지 않아도 되어 코드가 간결해집니다. 예를 들어, 다음 코드에서 `auto`는 초기화 값 `3.14`를 보고 `double`로 추론합니다. 

```cpp
#include <iostream>
using namespace std;

int main() {
    auto x = 3.14;   // double x = 3.14; 와 동일
    cout << typeid(x).name() << endl;  // 예: "d" (double을 나타냄)
    return 0;
}
```

`auto`는 함수의 반환 타입 추론이나 복잡한 템플릿 타입 선언에도 유용합니다. 한편, **람다 표현식(lambda)** 는 C++11에서 도입된 익명 함수 객체로, 함수를 마치 로컬에서 정의하듯 작성할 수 있게 해줍니다 ([Lambda expressions in C++ | Microsoft Learn](https://learn.microsoft.com/en-us/cpp/cpp/lambda-expressions-in-cpp?view=msvc-170#:~:text=In%20C%2B%2B11%20and%20later%2C%20a,lambdas%20are%2C%20and%20compares%20them)). 람다는 `[캡처](매개변수)` 구문과 함수 본체 `{}`로 이루어지며, 필요한 경우 외부 변수도 캡처하여 사용할 수 있습니다. 예를 들어, 아래 코드는 벡터 `v`의 각 요소를 2배로 만드는 람다를 사용하고 있습니다. 

```cpp
#include <iostream>
#include <vector>
#include <algorithm>
using namespace std;

int main() {
    vector<int> v = {1, 2, 3, 4, 5};
    int factor = 2;
    // 캡처[&]는 함수 외부 변수 factor를 참조로 캡처
    for_each(v.begin(), v.end(), [&](int &elem) {
        elem *= factor;  // 외부 변수 factor를 사용
    });
    for (int elem : v) {
        cout << elem << " ";  // 출력: 2 4 6 8 10
    }
    return 0;
}
```

또한, **범위 기반 for문**(range-based for loop)은 C++11에 도입된 새로운 반복문으로, 배열이나 벡터 같은 반복 가능한 객체의 모든 요소를 자동으로 순회할 수 있습니다 ([Range-Based for Loop in C++ | GeeksforGeeks](https://www.geeksforgeeks.org/range-based-loop-c/#:~:text=In%20C%2B%2B%2C%20the%20range,compared%20to%20traditional%20for%20loops)). 전통적인 반복문보다 구문이 간결해지고 코드의 가독성이 높아집니다. 예를 들면: 

```cpp
#include <iostream>
#include <vector>
using namespace std;

int main() {
    vector<string> names = {"Alice", "Bob", "Charlie"};
    // 범위 기반 for문으로 모든 요소 출력
    for (const auto &name : names) {
        cout << name << " ";
    }
    // 출력: Alice Bob Charlie
    return 0;
}
```

마지막으로, C++11에서는 `constexpr` 키워드를 도입하여 함수나 변수를 **컴파일 시점에 평가 가능한 상수식(constant expression)** 으로 선언할 수 있게 했습니다 ([constexpr specifier (since C++11) - cppreference.com](https://en.cppreference.com/w/cpp/language/constexpr#:~:text=The%20constexpr%20specifier%20declares%20that,provided%20that%20appropriate%20function)). `constexpr` 함수는 컴파일러가 호출 시 인자가 모두 상수이면 컴파일 타임에 미리 계산할 수 있어 성능 향상에 도움을 줍니다. 예를 들어:

```cpp
#include <iostream>
using namespace std;

constexpr int square(int x) {
    return x * x;
}

int main() {
    // 컴파일 시 square(5)가 계산되어 25로 대체됨
    constexpr int size = square(5);
    int arr[size];  // 크기가 25인 배열
    cout << square(3) << endl;  // 출력: 9
    return 0;
}
```

**연습 문제:**  
- `auto` 키워드의 장점과 한계점을 설명하고, `auto`를 사용하지 말아야 할 경우를 예를 들어 보세요.  
- 람다 표현식에서 `[&]`와 `[=]` 캡처가 각각 어떤 의미인지 설명하고, 간단한 예제를 작성해 보세요.  
- 범위 기반 for문을 사용하여 배열이나 STL 컨테이너를 순회하는 코드를 작성해 보세요.  
- `constexpr` 함수와 일반 함수의 차이점을 설명하고, `constexpr`을 사용할 때 주의할 점을 서술해 보세요.  

# 메모리 관리

C++에서는 자원의 안전한 관리를 위해 **RAII**(Resource Acquisition Is Initialization) 기법을 활용합니다. RAII는 객체의 생성자에서 리소스(예: 동적 메모리, 파일 핸들, 뮤텍스 락 등)를 획득하고, 소멸자에서 해제하도록 함으로써 자원 누수를 방지합니다 ([RAII - cppreference.com](https://en.cppreference.com/w/cpp/language/raii#:~:text=Resource%20Acquisition%20Is%20Initialization%20or,the%20lifetime%20of%20an%20object)) ([RAII - cppreference.com](https://en.cppreference.com/w/cpp/language/raii#:~:text=RAII%20guarantees%20that%20the%20resource,to%20eliminate%20resource%20leaks%20and)). 즉, 객체의 **수명 주기와 함께 자원의 획득과 해제가 이루어지도록 설계**하는 것입니다. 예를 들어, 클래스의 소멸자에서 `delete`를 호출하면, 해당 객체가 함수 범위를 벗어날 때 자동으로 동적 메모리가 해제됩니다. 이런 방식을 통해 프로그래머는 수동으로 `delete`를 호출하는 실수를 줄일 수 있습니다.

스마트 포인터는 RAII를 구현한 대표적인 도구입니다. `std::unique_ptr`는 단일 소유권 모델을 제공하는 스마트 포인터로, **오직 하나의 포인터만이 자원을 소유**합니다. 즉, `unique_ptr` 객체가 범위를 벗어나면 자동으로 소멸되면서 할당된 메모리를 해제합니다 ([When to use std::unique_ptr vs. other smart pointers - Mastering std::unique_ptr for Memory Management in C++ | StudyRaid](https://app.studyraid.com/en/read/11857/377001/when-to-use-stduniqueptr-vs-other-smart-pointers#:~:text=Use%20,unless%20shared%20ownership%20is%20necessary)). 반면 `std::shared_ptr`는 참조 카운팅을 사용하는 공유 소유 모델로, 여러 포인터가 같은 자원을 공유할 수 있습니다. 내부 카운터를 통해 소유자가 더 이상 없을 때 자원이 해제되도록 보장합니다 ([When to use std::unique_ptr vs. other smart pointers - Mastering std::unique_ptr for Memory Management in C++ | StudyRaid](https://app.studyraid.com/en/read/11857/377001/when-to-use-stduniqueptr-vs-other-smart-pointers#:~:text=Use%20,unless%20shared%20ownership%20is%20necessary)). 순환 참조를 방지하기 위해 `std::weak_ptr`를 사용하기도 합니다. 예제 코드를 통해 살펴보겠습니다:

```cpp
#include <iostream>
#include <memory>
using namespace std;

class Resource {
public:
    Resource() { cout << "Resource acquired\n"; }
    ~Resource() { cout << "Resource destroyed\n"; }
};

int main() {
    {
        // unique_ptr 예제
        unique_ptr<Resource> ptr = make_unique<Resource>();
        // ptr이 범위를 벗어나면 소멸자 호출과 함께 자원 해제
    } // 출력: "Resource acquired" "Resource destroyed"
    
    {
        // shared_ptr 예제
        shared_ptr<Resource> sp1 = make_shared<Resource>();
        {
            shared_ptr<Resource> sp2 = sp1; // 참조 카운트가 2가 됨
            cout << "sp2 goes out of scope\n";
        } // sp2 소멸, 참조 카운트 1
        cout << "sp1 still exists\n";
    } // 마지막 shared_ptr 소멸, 리소스 파괴
    // 출력 순서:
    // "Resource acquired"
    // "sp2 goes out of scope"
    // "sp1 still exists"
    // "Resource destroyed"
    
    return 0;
}
```

위 예제에서 `unique_ptr`와 `shared_ptr`는 각각 소유권 모델에 맞게 자원을 자동 관리합니다. **스마트 포인터 사용의 이점**은 수동 `new`/`delete`를 할 필요 없이 RAII에 따라 예외가 발생하더라도 소멸자가 호출되어 메모리 누수가 방지된다는 점입니다 ([RAII - cppreference.com](https://en.cppreference.com/w/cpp/language/raii#:~:text=RAII%20guarantees%20that%20the%20resource,to%20eliminate%20resource%20leaks%20and)) ([When to use std::unique_ptr vs. other smart pointers - Mastering std::unique_ptr for Memory Management in C++ | StudyRaid](https://app.studyraid.com/en/read/11857/377001/when-to-use-stduniqueptr-vs-other-smart-pointers#:~:text=Use%20,unless%20shared%20ownership%20is%20necessary)). 

**연습 문제:**  
- RAII와 스마트 포인터가 메모리 안전성에 어떻게 기여하는지 설명해 보세요.  
- `std::unique_ptr`와 `std::shared_ptr`의 차이점을 요약하고, 각자가 적합한 상황을 예를 들어 보세요.  
- 다음 코드에서 메모리 누수가 발생하는 문제를 찾아 수정해 보세요. (_힌트: 스마트 포인터를 사용하세요._)  
  ```cpp
  void leak() {
      int* p = new int(42);
      // ... p를 사용
      // 누락된 delete로 메모리 누수 발생
  }
  ```

# 템플릿과 STL

C++의 **템플릿**은 제네릭 프로그래밍을 가능하게 하는 강력한 도구로, 다양한 데이터 타입에 대해 **동일한 알고리즘**을 적용할 수 있게 해줍니다 ([Templates in C++ | GeeksforGeeks](https://www.geeksforgeeks.org/templates-cpp/#:~:text=C%2B%2B%20template%20is%20a%20powerful,code%20for%20different%20data%20types)). 템플릿 키워드와 타입 매개변수를 사용하여 함수나 클래스를 정의하면, 필요한 타입을 인스턴스화할 때마다 별도의 구현 코드를 생성하지 않고 하나의 정의로 여러 타입을 처리할 수 있습니다. 예를 들어, 두 값을 비교해 큰 값을 반환하는 함수 템플릿을 작성할 수 있습니다:

```cpp
#include <iostream>
using namespace std;

// 함수 템플릿 정의
template <typename T>
T myMax(T a, T b) {
    return (a > b) ? a : b;
}

int main() {
    cout << myMax<int>(3, 7) << endl;    // 출력: 7
    cout << myMax<double>(3.5, 2.1) << endl;  // 출력: 3.5
    cout << myMax<char>('g', 'e') << endl;    // 출력: g
    return 0;
}
```

이처럼 템플릿을 사용하면 `int`, `double`, `char` 등 여러 타입에 대해 동일한 `myMax` 함수를 재사용할 수 있습니다 ([Templates in C++ | GeeksforGeeks](https://www.geeksforgeeks.org/templates-cpp/#:~:text=C%2B%2B%20template%20is%20a%20powerful,code%20for%20different%20data%20types)). 클래스 템플릿도 마찬가지로 벡터, 연결 리스트 등 다양한 자료구조를 **타입 독립적으로** 구현할 수 있게 해줍니다.

STL(Standard Template Library)은 이러한 템플릿 기반의 자료구조와 알고리즘을 제공합니다. **컨테이너(container)** 는 STL의 자료 구조 구현체로, 내부에서 요소를 저장하고 관리합니다. 벡터(vector), 리스트(list), 맵(map) 등 여러 가지 컨테이너가 있으며, 모두 클래스 템플릿으로 구현되어 있어 다양한 타입의 데이터를 저장할 수 있습니다 ([Containers in C++ STL | GeeksforGeeks](https://www.geeksforgeeks.org/containers-cpp-stl/#:~:text=Standard%20Template%20Library%20,in%20the%20data%20types%20supported)). 예를 들어 `std::vector<int>`는 `int`를 저장하는 동적 배열 컨테이너입니다. 컨테이너를 사용할 때는 반복자(iterator)를 통해 요소에 접근할 수 있으며, 앞서 본 범위 기반 for문도 유용하게 활용됩니다.

또한 STL은 **알고리즘** 라이브러리도 제공합니다. `std::sort`, `std::find`, `std::for_each` 등 정렬·탐색·반복 등에 사용되는 알고리즘을 미리 구현해 놓았고, 이는 성능 최적화가 잘 되어 있습니다 ([C++ STL Algorithm Library | GeeksforGeeks](https://www.geeksforgeeks.org/c-magicians-stl-algorithms/#:~:text=Standard%20Template%20Library%20,faster%2C%20and%20more%20readable%20code)). 예를 들어, `std::sort`를 사용하여 벡터를 오름차순으로 정렬해 보겠습니다:

```cpp
#include <iostream>
#include <vector>
#include <algorithm>
using namespace std;

int main() {
    vector<int> v = {3, 1, 4, 1, 5};
    sort(v.begin(), v.end());  // <algorithm> 헤더의 std::sort
    for (auto x : v) {
        cout << x << " ";  // 출력: 1 1 3 4 5
    }
    return 0;
}
```

이처럼 STL 컨테이너와 알고리즘을 적절히 활용하면 복잡한 자료구조나 반복 작업을 간단하게 처리할 수 있습니다 ([Containers in C++ STL | GeeksforGeeks](https://www.geeksforgeeks.org/containers-cpp-stl/#:~:text=Standard%20Template%20Library%20,in%20the%20data%20types%20supported)) ([C++ STL Algorithm Library | GeeksforGeeks](https://www.geeksforgeeks.org/c-magicians-stl-algorithms/#:~:text=Standard%20Template%20Library%20,faster%2C%20and%20more%20readable%20code)).

**연습 문제:**  
- 템플릿을 사용하여 두 값 중 큰 값을 반환하는 함수를 작성하고, 다양한 타입으로 테스트해 보세요.  
- STL 컨테이너 중 `std::vector`와 `std::map`의 용도와 특징을 비교해 보세요.  
- `std::sort`와 `std::find`를 사용한 예제를 작성하여 벡터에서 특정 값을 찾거나 정렬해 보세요.

# 스레드 프로그래밍 기초

멀티스레딩은 프로그램을 여러 **실행 단위(thread)** 로 분할하여 병렬로 실행하는 기법으로, 다중 CPU 코어를 효율적으로 활용해 성능을 향상시킵니다 ([Multithreading in C++ | GeeksforGeeks](https://www.geeksforgeeks.org/multithreading-in-cpp/#:~:text=Multithreading%20is%20a%20technique%20where,thread%3E%20header%20file)). C++11부터 `<thread>` 헤더를 통해 멀티스레드 프로그래밍을 지원하며, `std::thread` 클래스를 사용해 새로운 스레드를 생성할 수 있습니다 ([Multithreading in C++ | GeeksforGeeks](https://www.geeksforgeeks.org/multithreading-in-cpp/#:~:text=Multithreading%20is%20a%20technique%20where,thread%3E%20header%20file)). 예를 들어, 아래 코드는 별도의 스레드를 생성하여 `printMessage` 함수를 실행하고, 메인 스레드에서는 `join()`을 호출하여 해당 스레드가 완료될 때까지 기다립니다.

```cpp
#include <iostream>
#include <thread>
using namespace std;

void printMessage(const string &msg) {
    cout << "Thread says: " << msg << endl;
}

int main() {
    // 새로운 스레드를 생성하여 함수 실행
    thread t(printMessage, "Hello");
    // 메인 스레드는 스레드 t가 끝날 때까지 기다림
    t.join();
    cout << "Main thread finished." << endl;
    return 0;
}
// 출력 예:
// Thread says: Hello
// Main thread finished.
```

스레드 간에 **공유 자원**(예: 전역 변수, 동적 메모리 등)에 동시에 접근할 때는 **데이터 경합(data race)** 이 발생할 수 있습니다. 이를 방지하기 위해 C++은 뮤텍스(mutex)와 같은 동기화 도구를 제공합니다. `std::mutex`는 상호 배제(Mutual Exclusion)를 위한 동기화 프리미티브로, 한 스레드가 `lock()`을 호출해 자원에 접근하는 동안 다른 스레드는 대기하도록 합니다 ([Mutex in C++ | GeeksforGeeks](https://www.geeksforgeeks.org/std-mutex-in-cpp/#:~:text=Mutex%20stands%20for%20Mutual%20Exclusion,of%20variables%2C%20data%20structures%2C%20etc)) ([Mutex in C++ | GeeksforGeeks](https://www.geeksforgeeks.org/std-mutex-in-cpp/#:~:text=In%20C%2B%2B%2C%20when%20multiple%20threads,the%20current%20thread%20is%20done)). 예를 들어, 다음 코드는 뮤텍스를 사용하여 전역 카운터를 안전하게 증가시킵니다.

```cpp
#include <iostream>
#include <thread>
#include <mutex>
using namespace std;

int counter = 0;
mutex mtx;

void incrementCounter(int loops) {
    for (int i = 0; i < loops; ++i) {
        lock_guard<mutex> lock(mtx);  // 스코프 내에서 뮤텍스를 자동 잠금/해제
        ++counter;
    }
}

int main() {
    thread t1(incrementCounter, 100000);
    thread t2(incrementCounter, 100000);
    t1.join();
    t2.join();
    cout << "Counter: " << counter << endl;
    return 0;
}
// 뮤텍스를 사용하지 않으면 counter는 예상값(200000)보다 작게 나올 수 있음.
```

또한 `std::atomic`은 명시적 잠금 없이도 변수에 대한 원자적(atomic) 연산을 제공하여 데이터 경합을 방지합니다 ([C++ 11 – <atomic> Header | GeeksforGeeks](https://www.geeksforgeeks.org/cpp-11-atomic-header/#:~:text=In%20C%2B%2B11%2C%20the%20,In%20this)). 예를 들어, `std::atomic<int>` 타입을 사용하면 다음과 같이 간단히 스레드 간 안전한 카운터를 구현할 수 있습니다.

```cpp
#include <iostream>
#include <thread>
#include <atomic>
using namespace std;

atomic<int> atomicCounter{0};

void incrementAtomic(int loops) {
    for (int i = 0; i < loops; ++i) {
        atomicCounter.fetch_add(1, memory_order_relaxed);
    }
}

int main() {
    thread t1(incrementAtomic, 100000);
    thread t2(incrementAtomic, 100000);
    t1.join();
    t2.join();
    cout << "Atomic Counter: " << atomicCounter.load() << endl;
    return 0;
}
// 뮤텍스 없이도 atomicCounter는 항상 200000을 출력함.
```

위 예제에서 `atomicCounter`는 모든 증감 연산을 **원자적으로** 처리하므로 추가적인 잠금 없이도 스레드 안전성을 보장합니다 ([C++ 11 – <atomic> Header | GeeksforGeeks](https://www.geeksforgeeks.org/cpp-11-atomic-header/#:~:text=In%20C%2B%2B11%2C%20the%20,In%20this)).

**연습 문제:**  
- `std::thread`, `std::mutex`, `std::atomic`의 역할과 차이점을 각각 설명해 보세요.  
- 다음 코드는 race condition이 발생합니다. `std::mutex` 또는 `std::atomic`을 사용하여 이를 해결해 보세요.  
  ```cpp
  #include <thread>
  int counter = 0;
  void foo() { for(int i=0;i<1000;i++) ++counter; }
  int main() {
      std::thread t1(foo);
      std::thread t2(foo);
      t1.join();
      t2.join();
      // 원하는 값: 2000
      return 0;
  }
  ```  
- `std::thread`의 `join()`과 `detach()`의 차이점을 설명하고, 각각의 사용 예를 들어 보세요.  

**참고 자료:** 본 장에서 다룬 내용은 C++11 이후 도입된 언어 기능과 기법으로 구성되었습니다. 각주에 인용된 자료들은 이 개념들을 상세히 설명하고 있으니 참고하시기 바랍니다 ([auto (C++) | Microsoft Learn](https://learn.microsoft.com/en-us/cpp/cpp/auto-cpp?view=msvc-170#:~:text=The%20,parameter%2C%20to%20deduce%20its%20type)) ([Lambda expressions in C++ | Microsoft Learn](https://learn.microsoft.com/en-us/cpp/cpp/lambda-expressions-in-cpp?view=msvc-170#:~:text=In%20C%2B%2B11%20and%20later%2C%20a,lambdas%20are%2C%20and%20compares%20them)) ([Range-Based for Loop in C++ | GeeksforGeeks](https://www.geeksforgeeks.org/range-based-loop-c/#:~:text=In%20C%2B%2B%2C%20the%20range,compared%20to%20traditional%20for%20loops)) ([constexpr specifier (since C++11) - cppreference.com](https://en.cppreference.com/w/cpp/language/constexpr#:~:text=The%20constexpr%20specifier%20declares%20that,provided%20that%20appropriate%20function)) ([RAII - cppreference.com](https://en.cppreference.com/w/cpp/language/raii#:~:text=Resource%20Acquisition%20Is%20Initialization%20or,the%20lifetime%20of%20an%20object)) ([When to use std::unique_ptr vs. other smart pointers - Mastering std::unique_ptr for Memory Management in C++ | StudyRaid](https://app.studyraid.com/en/read/11857/377001/when-to-use-stduniqueptr-vs-other-smart-pointers#:~:text=Use%20,unless%20shared%20ownership%20is%20necessary)) ([Templates in C++ | GeeksforGeeks](https://www.geeksforgeeks.org/templates-cpp/#:~:text=C%2B%2B%20template%20is%20a%20powerful,code%20for%20different%20data%20types)) ([Containers in C++ STL | GeeksforGeeks](https://www.geeksforgeeks.org/containers-cpp-stl/#:~:text=Standard%20Template%20Library%20,in%20the%20data%20types%20supported)) ([C++ STL Algorithm Library | GeeksforGeeks](https://www.geeksforgeeks.org/c-magicians-stl-algorithms/#:~:text=Standard%20Template%20Library%20,faster%2C%20and%20more%20readable%20code)) ([Multithreading in C++ | GeeksforGeeks](https://www.geeksforgeeks.org/multithreading-in-cpp/#:~:text=Multithreading%20is%20a%20technique%20where,thread%3E%20header%20file)) ([Mutex in C++ | GeeksforGeeks](https://www.geeksforgeeks.org/std-mutex-in-cpp/#:~:text=Mutex%20stands%20for%20Mutual%20Exclusion,of%20variables%2C%20data%20structures%2C%20etc)) ([C++ 11 – <atomic> Header | GeeksforGeeks](https://www.geeksforgeeks.org/cpp-11-atomic-header/#:~:text=In%20C%2B%2B11%2C%20the%20,In%20this)).