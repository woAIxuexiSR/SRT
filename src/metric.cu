#include <iostream>

#include "metric.h"
#include "argparse.h"

int main(int argc, char* argv[])
{
    auto args = argparser("Example program")
        .add_option("-v", "--verbose", "increase verbosity")
        .add_option("-n", "--name", "name", "test")
        .add_argument<int>("input1", "input number 1")
        .add_argument<int>("input2", "input number 2")
        .parse(argc, argv);

    int n1 = args.get<int>("input1");
    int n2 = args.get<int>("input2");
    int ans = n1 + n2;

    if(args.get<bool>("-v"))
    {
        std::cout << "Verbose mode" << std::endl;
        std::cout << "Name: " << args.get<std::string>("-n") << std::endl;
        std::cout << "Input 1: " << n1 << std::endl;
        std::cout << "Input 2: " << n2 << std::endl;
        std::cout << "Answer: " << ans << std::endl;
    }
    else
        std::cout << ans << std::endl;


    return 0;
}