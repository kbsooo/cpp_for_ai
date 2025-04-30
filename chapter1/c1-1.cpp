#include <iostream>
using namespace std;

int main()
{
  auto x = 3.14;                    // double x = 3.14 와 동일
  cout << typeid(x).name() << endl; // -> "d" (double)
  return 0;
}
