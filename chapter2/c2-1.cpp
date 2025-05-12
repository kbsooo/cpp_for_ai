#include <iostream>
#include <chrono>
#include <memory>
using namespace std;

// 1. 스택 메모리와 힙 메모리 할당의 오버헤드를 측정하는 작은 프로그램을 작성해보자. 예를 들어 new/delete 반복문과 로컬 배열 사용 반복문을 각각 측정하고 비교해보자



// 2. std::shared_ptr를 이용해 데이터 구조를 구현한 뒤, 동일 기능을 std::unique_ptr로 바꾸어 실행 속도를 비교해보자. 스마트 포인터 생성 및 소멸 횟수에 따른 시간 차리를 관찰해보자.

int main() {
    // Q1
    cout << "Q1\n";
    const int N = 2000000;
    const int M = 1000;

    auto t1 = chrono::high_resolution_clock::now();
    for (int i = 0; i < N; ++i) {
        int arr[M];
        arr[0] = i;
    }

    auto t2 = chrono::high_resolution_clock::now();
    auto t3 = chrono::high_resolution_clock::now();
    for (int i = 0; i < N; ++i) {
        int *p = new int[M];
        p[0] = i;
        delete [] p;
    }
    auto t4 = chrono::high_resolution_clock::now();

    double stack_time = chrono::duration<double>(t2 - t1).count();
    double heap_time = chrono::duration<double>(t4 - t3).count();
    cout << "stack time: " << stack_time << "s\n";
    cout << "heap time: " << heap_time << "s\n";

    // Q2
    cout << "Q2\n";
    auto t5 = chrono::high_resolution_clock::now();
    for (int i = 0; i < N; ++i) {
        unique_ptr<int> p(new int(i));
    }

    auto t6 = chrono::high_resolution_clock::now();
    for (int i = 0; i < N; ++i) {
        int *p = new int(i);
        delete p;
    }

    auto t7 = chrono::high_resolution_clock::now();
    for (int i = 0; i < N; ++i) {
        shared_ptr<int> p(new int(i));
    }
    
    auto t8 = chrono::high_resolution_clock::now();

    double unique_time = chrono::duration<double>(t6 - t5).count();
    double raw_time = chrono::duration<double>(t7 - t6).count();
    double shared_time = chrono::duration<double>(t8 - t7).count();

    cout << "unique_ptr time: " << unique_time << "s\n";
    cout << "raw new\\delete: " << raw_time << "s\n";
    cout << "shared_ptr time: " << shared_time << "s\n";
    return 0;
}
