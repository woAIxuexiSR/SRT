#include <iostream>
#include "renderer.h"
#include "profiler.h"

class A
{
public:
    int a;

public:
    friend void to_json(json& j, const A& a)
    {
        j = json{ {"a", a.a} };
    }

    friend void from_json(const json& j, A& a)
    {
        cout << "a" << endl;
        j.at("a").get_to(a.a);
    }
};

class B : public A
{
public:
    int b { 0};

public:
    friend void to_json(json& j, const B& b)
    {
        j = json{ {"a", b.a}, {"b", b.b} };
    }

    friend void from_json(const json& j, B& b)
    {
        cout << "b" << endl;
        j.at("a").get_to(b.a);
        j.at("b").get_to(b.b);
    }
};

int main()
{
    json config = { { "a" , 2 }, {"b", 3} };

    shared_ptr<A> a = make_shared<B>(config);

    json t = config["c"];
    int c = !t.is_null() ? t.value("c", 1) : 1;
    cout << t << endl;
    cout << c << endl;


    return 0;
}