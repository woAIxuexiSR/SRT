#pragma once

#include "definition.h"
#include "helper_cuda.h"


class CpuTimer
{
private:
    std::chrono::time_point<std::chrono::high_resolution_clock> start, end;
public:
    void start_timer() { start = std::chrono::high_resolution_clock::now(); }
    void end_timer() { end = std::chrono::high_resolution_clock::now(); }

    // time in ms
    float get_time()
    {
        std::chrono::duration<float, std::milli> duration = end - start;
        return duration.count();
    }
};


class GpuTimer
{
private:
    cudaEvent_t start, end;
public:
    GpuTimer() { checkCudaErrors(cudaEventCreate(&start)); checkCudaErrors(cudaEventCreate(&end)); }
    ~GpuTimer() { checkCudaErrors(cudaEventDestroy(start)); checkCudaErrors(cudaEventDestroy(end)); }
    void start_timer() { checkCudaErrors(cudaEventRecord(start, 0)); }
    void end_timer() { checkCudaErrors(cudaEventRecord(end, 0)); }

    // time in ms
    float get_time()
    {
        checkCudaErrors(cudaEventSynchronize(end));
        float time;
        checkCudaErrors(cudaEventElapsedTime(&time, start, end));
        return time;
    }
};


class ProfilerRecorder
{
public:
    string name;
    int count;
    float cpu_time, gpu_time;
    shared_ptr<CpuTimer> cpu_timer;
    shared_ptr<GpuTimer> gpu_timer;

public:
    ProfilerRecorder(const string& _n)
        : name(_n), count(0), cpu_time(0), gpu_time(0)
    {
        cpu_timer = make_shared<CpuTimer>();
        gpu_timer = make_shared<GpuTimer>();
    }

    void start_event()
    {
        cpu_timer->start_timer();
        gpu_timer->start_timer();
    }

    void end_event()
    {
        cpu_timer->end_timer();
        gpu_timer->end_timer();
        cpu_time += cpu_timer->get_time();
        gpu_time += gpu_timer->get_time();
        count++;
    }
};


class Profiler
{
public:
    vector<ProfilerRecorder> recorder;
    unordered_map<string, int> m;
    string current_event;

private:

    void start_event(const string& name)
    {
        string e_name = current_event + "/" + name;

        int idx = -1;
        if (m.find(e_name) == m.end())
        {
            m[e_name] = idx = recorder.size();
            recorder.push_back(ProfilerRecorder(e_name));
        }
        else idx = m[e_name];

        recorder[idx].start_event();
        current_event = e_name;
    }

    void end_event()
    {
        string e_name = current_event;
        if(m.find(e_name) == m.end()) return;
        int idx = m[e_name];
        recorder[idx].end_event();
        current_event = current_event.substr(0, current_event.find_last_of("/"));
    }

public:
    static Profiler& get_profiler()
    {
        static Profiler profiler;
        return profiler;
    }

    struct Recorder
    {
        Recorder(const string& name)
        {
            auto& profiler = get_profiler();
            profiler.start_event(name);
        }
        ~Recorder()
        {
            auto& profiler = get_profiler();
            profiler.end_event();
        }
    };

    static void reset()
    {
        auto& profiler = get_profiler();
        profiler.recorder.clear();
        profiler.m.clear();
        profiler.current_event = "";
    }

    static void stop()
    {
        auto& profiler = get_profiler();
        while (profiler.current_event != "")
            profiler.end_event();
    }

    static void print()
    {
        auto& profiler = get_profiler();
        cout << "Profiler result:" << endl;
        for (auto& r : profiler.recorder)
        {
            string name = r.name.substr(r.name.find_last_of("/") + 1);
            int level = std::count(r.name.begin(), r.name.end(), '/') - 1;
            string indent = "";
            for (int i = 0; i < level; i++) indent += "  ";

            cout << indent << name << ": " << r.count << " times, ";
            cout << "cpu time: " << r.cpu_time << " ms, ";
            cout << "gpu time: " << r.gpu_time << " ms" << endl;
        }
    }

    static void render_ui()
    {

    }
};

#define PROFILE(name) Profiler::Recorder recorder##__LINE__(name)
