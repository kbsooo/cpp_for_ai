// 1. 템플릿을 사용하여 두 값 중 큰 값을 반환하는 함수를 작성하고, 다양한 타입으로 테스트해 보세요.
#include <iostream>
using namespace std;

template <typename T>
T myMax(T a, T b) { return (a > b) ? a : b; }

int main() {
    cout << myMax<int>(3, 7) << endl;
    cout << myMax<double>(3.5, 2.1) << endl;
    cout << myMax<char>('g', 'e') << endl;
    return 0;
}

// 2. STL 컨테이너 중 std::vector와 std::map의 용도와 특징을 비교해 보세요.

// std::vector: 순차적인 데이터 집합을 효율적으로 관리하고 싶을 때, 인덱스를 통한 빠른 접근이 필요할 때 사용합니다
// std::map: 고유한 키를 기준으로 데이터를 저장하고 빠르게 검색하고 싶을 때 사용합니다. 데이터가 키에 의해 정렬된 상태로 유지됩니다

// 3. std::sort와 std::find를 사용한 예제를 작성하여 벡터에서 특정 값을 찾거나 정렬해 보세요.
#include <iostream>
#include <vector>
#include <algorithm>
#include <numeric>
#include <random>

void printVector(const std::vector<int>& vec, const std::string& title) {
    std::cout << title << ": ";
    for (int val: vec) {
        std::cout << val << " ";
    }
    std::cout << std::endl;
}

int main() {
    std::vector<int> numbers = {50, 10, 90, 30, 70, 20, 80, 40, 60, 100};
    printVector(numbers, "org vector");

    std::sort(numbers.begin(), numbers.end());
    printVector(numbers, "Sorted vector ");

    int value_to_find = 70;
    auto it_found = std::find(numbers.begin(), numbers.end(), value_to_find);

    if (it_found != numbers.end()) {
        int index = std::distance(numbers.begin(), it_found);
        std::cout << "Value " << value_to_find << " found at index " << index << "." << std::endl;
    } else {
        std::cout << "Value " << value_to_find << " not found in the vector." << std::endl;
    }

    int value_not_in_vector = 55;
    it_found = std::find(numbers.begin(), numbers.end(), value_not_in_vector);
    if (it_found != numbers.end()) {
        int index = std::distance(numbers.begin(), it_found);
        std::cout << "Value " << value_not_in_vector << " found at index " << index << "." << std::endl;
    } else {
        std::cout << "Value " << value_not_in_vector << " not found in the vector." << std::endl;
    }

    return 0;
}