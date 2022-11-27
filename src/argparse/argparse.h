#pragma once

#include <string>
#include <vector>
#include <iostream>
#include <sstream>
#include <algorithm>
#include <functional>


template<class T>
T parse_value(const std::string& value)
{
    std::stringstream ss(value);
    T result;
    ss >> result;
    return result;
}

template<class T>
std::string to_str(const T& value)
{
    std::stringstream ss;
    ss << value;
    return ss.str();
}

class short_circuit_option
{
public:
    std::string short_name;
    std::string long_name;
    std::string help;
    std::function<void(void)> callback;

    short_circuit_option(const std::string& _sn, const std::string& _ln, const std::string& _h, const std::function<void(void)>& _cb)
        : short_name(_sn), long_name(_ln), help(_h), callback(_cb)
    { }
};

class option
{
public:
    std::string short_name;
    std::string long_name;
    std::string help;
    std::size_t type;
    std::string value;

    option(const std::string& _sn, const std::string& _ln, const std::string& _h, std::size_t _hc, const std::string& _v)
        : short_name(_sn), long_name(_ln), help(_h), type(_hc), value(_v)
    { }
};

class argument
{
public:
    std::string name;
    std::string help;
    std::size_t type;
    std::string value;

    argument(const std::string& _n, const std::string& _h, std::size_t _hc)
        : name(_n), help(_h), type(_hc)
    { }
};

class argparser
{
private:
    std::string description;
    std::string program_name;
    std::vector<short_circuit_option> sc_options;
    std::vector<option> options;
    std::vector<argument> arguments;


public:
    argparser(const std::string& _des) : description(_des) {}

    argparser& set_program_name(const std::string& _name)
    {
        program_name = _name;
        return *this;
    }

    argparser& add_sc_option(const std::string& sname, const std::string& lname, const std::string& help, const std::function<void(void)>& callback)
    {
        if(check_option_exists(sname, lname))
        {
            std::cout << "Shortcut option " << sname << " or " << lname << " already exists" << std::endl;
            exit(-1);
        }
        sc_options.emplace_back(short_circuit_option(sname, lname, help, callback));
        return *this;
    }

    argparser& add_help_option()
    {
        return add_sc_option("-h", "--help", "Show help message", [this](){ print_help(); });
    }

    template<class T>
    argparser& add_option(const std::string& sname, const std::string& lname, const std::string& help, T&& value)
    {
        if(check_option_exists(sname, lname))
        {
            std::cout << "Option " << sname << " or " << lname << " already exists" << std::endl;
            exit(-1);
        }
        options.emplace_back(option(sname, lname, help, typeid(T).hash_code(), to_str(value)));
        return *this;
    }

    argparser& add_option(const std::string& sname, const std::string& lname, const std::string& help)
    {
        add_option<bool>(sname, lname, help, false);
        return *this;
    }

    template<class T>
    argparser& add_argument(const std::string& name, const std::string& help)
    {
        auto it = std::find_if(arguments.begin(), arguments.end(), [&](const argument& a) {
            return a.name == name;
        });
        if(it != arguments.end())
        {
            std::cout << "Argument " << name << " already exists" << std::endl;
            exit(-1);
        }
        arguments.emplace_back(argument(name, help, typeid(T).hash_code()));
        return *this;
    }

    template<class T>
    T get(const std::string& name) const
    {
        auto oit = std::find_if(options.begin(), options.end(), [&name](const option& o) {
            return o.short_name.substr(1) == name || o.long_name.substr(2) == name
                || o.short_name == name || o.long_name == name;
        });
        if(oit != options.end())
            return parse_value<T>(oit->value);

        auto ait = std::find_if(arguments.begin(), arguments.end(), [&name](const argument& a) {
            return a.name == name;
        });
        if(ait != arguments.end())
            return parse_value<T>(ait->value);

        std::cout << "Error: option or argument " << name << " not found!" << std::endl;
        exit(-1);
    }

    argparser& parse(int argc, char* argv[])
    {
        if(program_name == "")
            program_name = argv[0];

        if(argc == 1)
        {
            print_usage();
            exit(0);
        }

        std::vector<std::string> tokens;
        for(int i = 1; i < argc; i++)
            tokens.emplace_back(argv[i]);

        // parse short circuit options
        for(auto& sc : sc_options)
        {
            auto it = std::find_if(tokens.begin(), tokens.end(), [&sc](const std::string& t) {
                return t == sc.short_name || t == sc.long_name;
            });
            if(it == tokens.end())
                continue;
            sc.callback();
            exit(0);
        }

        // parse options
        for(auto &&opt : options)
        {
            auto it = std::find_if(tokens.begin(), tokens.end(), [&](const std::string& s) {
                return s == opt.short_name || s == opt.long_name;
            });
            
            if(it == tokens.end())
                continue;

            it = tokens.erase(it);
            if(opt.type == typeid(bool).hash_code())
            {
                opt.value = "1";
                continue;
            }

            if(it == tokens.end() || it->front() == '-')
            {
                std::cout << "Error parse option : " << opt.short_name << " " << opt.long_name << " should have value" << std::endl;
                exit(-1);
            }

            opt.value = *it;
            tokens.erase(it);
        }

        // parse arguments
        if(tokens.size() != arguments.size())
        {
            std::cout << "Error parse arguments : " << tokens.size() << " arguments provided, but " << arguments.size() << " expected" << std::endl;
            exit(-1);
        }
        for(int i = 0; i < arguments.size(); i++)
            arguments[i].value = tokens[i];

        return *this;
    }

    void print_usage()
    {
        std::cout << "Usage: " << program_name << " [options] ";
        for(auto &&arg : arguments)
            std::cout << arg.name << " ";
        std::cout << std::endl;
    }

    void print_help()
    {
        print_usage();
        std::cout << std::endl;
        std::cout << description << std::endl;
        std::cout << std::endl;
        std::cout << "Options:" << std::endl;
        for(auto &&opt : options)
        {
            std::cout << "  " << opt.short_name << ", " << opt.long_name << " : " << opt.help << std::endl;
        }
        for(auto &&sc : sc_options)
        {
            std::cout << "  " << sc.short_name << ", " << sc.long_name << " : " << sc.help << std::endl;
        }
        std::cout << std::endl;
        std::cout << "Arguments:" << std::endl;
        for(auto &&arg : arguments)
        {
            std::cout << "  " << arg.name << " : " << arg.help << std::endl;
        }
    }

    bool check_option_exists(const std::string& sname, const std::string& lname) const
    {
        auto it = std::find_if(options.begin(), options.end(), [&](const option& o) {
            return o.short_name == sname || o.long_name == lname;
        });
        if(it != options.end())
            return true;
        auto sc_it = std::find_if(sc_options.begin(), sc_options.end(), [&](const short_circuit_option& o) {
            return o.short_name == sname || o.long_name == lname;
        });
        return sc_it != sc_options.end();
    }
};