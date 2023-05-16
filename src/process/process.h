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
    shared_ptr<Scene> scene = nullptr;

public:
    RenderProcess() {}

    virtual void set_enable(bool _enable) { enable = _enable; }
    virtual void resize(int _w, int _h) { width = _w; height = _h; }
    virtual void set_scene(shared_ptr<Scene> _scene) { scene = _scene; }

    virtual void render(shared_ptr<Film> film) = 0;
    virtual void render_ui() {};
};


class RenderProcessFactory
{
private:
    using map_type = unordered_map<string, std::function<shared_ptr<RenderProcess>(const json&)> >;

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
            map.emplace(name, [](const json& params) { return make_shared<T>(params);});
        }
    };

    static shared_ptr<RenderProcess> create_process(const string& name, const json& params)
    {
        auto& map = get_map();
        auto it = map.find(name);
        if (it == map.end())
        {
            cout << "ERROR::Render Process " << name << " not found!" << endl;
            return nullptr;
        }
        return it->second(params);
    }

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
