#pragma once

#include "definition.h"
#include <sstream>
#include <algorithm>
#include <functional>


class short_circuit_option
{
public:
    string short_name;
    string long_name;
    string help;
    std::function<void(void)> callback;

    short_circuit_option(const string& _sn, const string& _ln, const string& _h, const std::function<void(void)>& _cb)
        : short_name(_sn), long_name(_ln), help(_h), callback(_cb)
    { }
};

class option
{
public:
    string short_name;
    string long_name;
    string help;
    std::size_t type;
    string value;

    option(const string& _sn, const string& _ln, const string& _h, std::size_t _hc, const string& _v)
        : short_name(_sn), long_name(_ln), help(_h), type(_hc), value(_v)
    { }
};

class argument
{
public:
    string name;
    string help;
    std::size_t type;
    string value;

    argument(const string& _n, const string& _h, std::size_t _hc)
        : name(_n), help(_h), type(_hc)
    { }
};

class argparser
{
private:
    string description;
    string program_name;
    vector<short_circuit_option> sc_options;
    vector<option> options;
    vector<argument> arguments;


public:
    argparser(const string& _des) : description(_des) {}

    argparser& set_program_name(const string& _name)
    {
        program_name = _name;
        return *this;
    }

    argparser& add_sc_option(const string& sname, const string& lname, const string& help, const std::function<void(void)>& callback)
    {
        if (check_option_exists(sname, lname))
        {
            cout << "Shortcut option " << sname << " or " << lname << " already exists" << endl;
            exit(-1);
        }
        sc_options.emplace_back(short_circuit_option(sname, lname, help, callback));
        return *this;
    }

    argparser& add_help_option()
    {
        return add_sc_option("-h", "--help", "Show help message", [this]() { print_help(); });
    }

    template<class T>
    argparser& add_option(const string& sname, const string& lname, const string& help, T&& value)
    {
        if (check_option_exists(sname, lname))
        {
            cout << "Option " << sname << " or " << lname << " already exists" << endl;
            exit(-1);
        }
        options.emplace_back(option(sname, lname, help, typeid(T).hash_code(), to_str(value)));
        return *this;
    }

    argparser& add_option(const string& sname, const string& lname, const string& help)
    {
        add_option<bool>(sname, lname, help, false);
        return *this;
    }

    template<class T>
    argparser& add_argument(const string& name, const string& help)
    {
        auto it = std::find_if(arguments.begin(), arguments.end(), [&](const argument& a) {
            return a.name == name;
            });
        if (it != arguments.end())
        {
            cout << "Argument " << name << " already exists" << endl;
            exit(-1);
        }
        arguments.emplace_back(argument(name, help, typeid(T).hash_code()));
        return *this;
    }

    template<class T>
    T get(const string& name) const
    {
        auto oit = std::find_if(options.begin(), options.end(), [&name](const option& o) {
            return o.short_name.substr(1) == name || o.long_name.substr(2) == name
                || o.short_name == name || o.long_name == name;
            });
        if (oit != options.end())
            return parse_value<T>(oit->value);

        auto ait = std::find_if(arguments.begin(), arguments.end(), [&name](const argument& a) {
            return a.name == name;
            });
        if (ait != arguments.end())
            return parse_value<T>(ait->value);

        cout << "Error: option or argument " << name << " not found!" << endl;
        exit(-1);
    }

    argparser& parse(int argc, char* argv[])
    {
        if (program_name == "")
            program_name = argv[0];

        if (argc == 1)
        {
            if (arguments.size() != 0)
            {
                print_usage();
                exit(0);
            }
            return *this;
        }

        vector<string> tokens;
        for (int i = 1; i < argc; i++)
            tokens.emplace_back(argv[i]);

        // parse short circuit options
        for (auto& sc : sc_options)
        {
            auto it = std::find_if(tokens.begin(), tokens.end(), [&sc](const string& t) {
                return t == sc.short_name || t == sc.long_name;
                });
            if (it == tokens.end())
                continue;
            sc.callback();
            exit(0);
        }

        // parse options
        for (auto&& opt : options)
        {
            auto it = std::find_if(tokens.begin(), tokens.end(), [&](const string& s) {
                return s == opt.short_name || s == opt.long_name;
                });

            if (it == tokens.end())
                continue;

            it = tokens.erase(it);
            if (opt.type == typeid(bool).hash_code())
            {
                opt.value = "1";
                continue;
            }

            if (it == tokens.end() || it->front() == '-')
            {
                cout << "Error parse option : " << opt.short_name << " " << opt.long_name << " should have value" << endl;
                exit(-1);
            }

            opt.value = *it;
            tokens.erase(it);
        }

        // parse arguments
        if (tokens.size() != arguments.size())
        {
            cout << "Error parse arguments : " << tokens.size() << " arguments provided, but " << arguments.size() << " expected" << endl;
            exit(-1);
        }
        for (int i = 0; i < arguments.size(); i++)
            arguments[i].value = tokens[i];

        return *this;
    }

    void print_usage()
    {
        cout << "Usage: " << program_name << " [options] ";
        for (auto&& arg : arguments)
            cout << arg.name << " ";
        cout << endl;
    }

    void print_help()
    {
        print_usage();
        cout << endl;
        cout << description << endl;
        cout << endl;
        cout << "Options:" << endl;
        for (auto&& opt : options)
        {
            cout << "  " << opt.short_name << ", " << opt.long_name << " : " << opt.help << endl;
        }
        for (auto&& sc : sc_options)
        {
            cout << "  " << sc.short_name << ", " << sc.long_name << " : " << sc.help << endl;
        }
        cout << endl;
        cout << "Arguments:" << endl;
        for (auto&& arg : arguments)
        {
            cout << "  " << arg.name << " : " << arg.help << endl;
        }
    }

    bool check_option_exists(const string& sname, const string& lname) const
    {
        auto it = std::find_if(options.begin(), options.end(), [&](const option& o) {
            return o.short_name == sname || o.long_name == lname;
            });
        if (it != options.end())
            return true;
        auto sc_it = std::find_if(sc_options.begin(), sc_options.end(), [&](const short_circuit_option& o) {
            return o.short_name == sname || o.long_name == lname;
            });
        return sc_it != sc_options.end();
    }
};