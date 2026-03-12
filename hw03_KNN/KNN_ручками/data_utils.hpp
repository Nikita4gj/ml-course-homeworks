#include<iostream>
#include<stdexcept>
#include<string_view>
#include<string>
#include<fstream>
#include<sstream>
#include<random>
#include<unordered_set>
#include<algorithm>
#include<variant>

#include"types.hpp"

#pragma once

using datatypes::DataFrame;
using datatypes::Predictions;


namespace utils{
    std::tuple<DataFrame, Predictions> read_csv(const std::string& path)
    {
        DataFrame data;
        Predictions targets;

        std::ifstream in(path);

        if(!in.is_open())
            throw std::runtime_error("The file could not be opened");

        std::string temp;

        std::getline(in, temp);

        std::istringstream sin(std::move(temp));

        int target_index = -1;
        do
        {
            target_index++;
            std::getline(sin, temp, ',');
        } while(!sin.eof() && temp != "target");

        if(temp!="target")
            throw std::runtime_error("There is no target column in the csv file");
        
        int row = 0;

        while(std::getline(in ,temp))
        {
            sin.str(temp);
            sin.clear();
            int col = 0;
            targets.emplace_back();
            data.emplace_back();

            while (std::getline(sin, temp, ','))
            {
                if(temp.empty())
                    throw std::invalid_argument(std::string("The empty value in ") + std::to_string(row) + " column");
                
                if(col == target_index)
                    targets[row] = std::stoi(temp);
                else
                    data[row].emplace_back(std::stod(temp));
                col++;
            }
            row++;
        }
        
        return {data, targets};
    }

    std::tuple<DataFrame, DataFrame, Predictions, Predictions> train_test_split(const DataFrame& data,
        const Predictions& targets,
        int random_seed = 52,
        double test_split = 0.3
    )
    {
        if(data.empty())
            throw std::invalid_argument("The data is empty");

        if(test_split>=1 || test_split<=0)
            throw std::invalid_argument("Test_split must be in (0, 1)");
        
        std::mt19937 gen(random_seed);

        std::vector<size_t> rows(data.size());

        std::iota(rows.begin(), rows.end(), 0);
        
        std::shuffle(rows.begin(), rows.end(), gen);

        size_t tests = data.size()*test_split;

        DataFrame X_test(tests), X_train(data.size()-tests);

        Predictions Y_test(tests), Y_train(data.size()-tests);

        for(int i = 0, j = 0; i<rows.size(); ++i, ++j)
        {
            if(i<tests)
            {
                X_test[i] = data[rows[i]];
                Y_test[i] = targets[rows[i]];
            }
            else
            {
                X_train[i-tests] = data[rows[i]];
                Y_train[i-tests] = targets[rows[i]];   
            }
        }
        return {X_train, X_test, Y_train, Y_test};
    }  
    
    void print(const DataFrame& data,
                const char* sep = " ",
                const char* end = "\n")
    {
        for(const auto& row : data)
        {
            for(auto elem : row)
                std::cout << elem << sep;
            std::cout << end;
        }
    }

    void print(const std::variant<Predictions, datatypes::Series>& data,
                const char* sep = " ",
                const char* end = "\n")
    {
        std::visit([sep, end](const auto& data)
        {
            for(auto elem : data)
                std::cout << elem << sep;
            std::cout << end;
        }, data);
    }
}


