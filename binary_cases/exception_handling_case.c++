#include <iostream>
using namespace std;

int calculateGCD(int a, int b) {
    if (b == 0) {
        return a;
    }
    return calculateGCD(b, a % b);
}

int main() {
    int num1, num2;
    cout << "input 2 number" << endl;
    cin >> num1 >> num2;

    try {
        if (num1 % num2 != 0 && num2 % num1 != 0) {
            throw "both are not divisible";
        }
    } catch (const char* errorMsg) {
        cout << "catch exception" << errorMsg << endl;
        int gcd = calculateGCD(num1, num2);
        cout << "caculate gcd" << gcd << endl;
    }

    return 0;
}