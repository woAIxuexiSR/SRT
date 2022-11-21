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
    std::vector<option> options;
    std::vector<argument> arguments;


public:
    argparser(const std::string& _des) : description(_des) {}

    argparser& set_program_name(const std::string& _name)
    {
        program_name = _name;
        return *this;
    }

    template<class T>
    argparser& add_option(const std::string& sname, const std::string& lname, const std::string& help, T&& value)
    {
        options.emplace_back(option(sname, lname, help, typeid(T).hash_code(), to_str(value)));
        return *this;
    }

    argparser& add_option(const std::string& sname, const std::string& lname, const std::string& help)
    {
        options.emplace_back(option(sname, lname, help, typeid(bool).hash_code(), "0"));
        return *this;
    }

    template<class T>
    argparser& add_argument(const std::string& name, const std::string& help)
    {
        arguments.emplace_back(argument(name, help, typeid(T).hash_code()));
        return *this;
    }

    template<class T>
    T get(const std::string& name) const
    {
        auto oit = std::find_if(options.begin(), options.end(), [&name](const option& o) {
            return o.short_name == name || o.long_name == name;
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
};