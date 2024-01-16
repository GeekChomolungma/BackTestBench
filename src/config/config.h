#include <iostream>
#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/ini_parser.hpp>

class Config {
public:
    Config(const std::string& filename) {
        boost::property_tree::ini_parser::read_ini(filename, pt);
    }

    std::string getDatabaseHost() const {
        return pt.get<std::string>("database.host");
    }

    int getDatabasePort() const {
        return pt.get<int>("database.port");
    }

    std::string getDatabaseUsername() const {
        return pt.get<std::string>("database.username");
    }

    std::string getDatabasePassword() const {
        return pt.get<std::string>("database.password");
    }

    std::string getUri() const{
        return pt.get<std::string>("database.uri");
    }

private:
    boost::property_tree::ptree pt;
};