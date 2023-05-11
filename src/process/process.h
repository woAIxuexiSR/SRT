#pragma once

#include "definition.h"
#include "my_math.h"
#include "scene.h"
#include "film.h"
#include "profiler.h"
#include <functional>

class RenderProcess
{
protected:
    bool enable;
    int width, height;
    shared_ptr<Scene> scene;

public:
    RenderProcess() {}
    RenderProcess(int _w, int _h, shared_ptr<Scene> _s = nullptr)
        : enable(true), width(_w), height(_h), scene(_s) {}

    virtual string get_name() const { return "RenderProcess"; }
    virtual void render(shared_ptr<Film> film) = 0;
    virtual void render_ui() {};
};


class RenderProcessFactory
{
private:
    using map_type = unordered_map<string, std::function<shared_ptr<RenderProcess>(void)> >;

public:
    static map_type& get_map()
    {
        static map_type map;
        return map;
    }

    template<class T>
    struct Register
    {
        Register(const string& name)
        {
            auto& map = get_map();
            map.emplace(name, [] { return make_shared<T>();});
        }
    };

    static void print_registered_processes()
    {
        auto& map = get_map();
        cout << "Registered Render Processes:" << endl;
        for (auto& it : map)
            cout << "    " << it.first << endl;
    }
};

#define REGISTER_RENDER_PROCESS(T) \
    static RenderProcessFactory::Register<T> reg

#define REGISTER_RENDER_PROCESS_CPP(T) \
    RenderProcessFactory::Register<T> T::reg(#T)
