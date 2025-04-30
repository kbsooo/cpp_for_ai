#include <iostream>
#include <vector>
#include <algorithm>
using namespace std;

int main()
{
  vector<int> v = {1, 2, 3, 4, 5};
  int factor = 2;

  for_each(v.begin(), v.end(), [&](int &elem)
           { elem *= factor; });

  for (int elem : v)
  {
    cout << elem << " ";
  }

  return 0;
}