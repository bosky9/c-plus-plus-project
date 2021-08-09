#include "families/family.hpp"

#include <iostream>

using namespace std;

int main() {
    Family f{"exp"};

    cout << f._transform(5);
    cout << '\n' << f._itransform(f._transform(5));
}