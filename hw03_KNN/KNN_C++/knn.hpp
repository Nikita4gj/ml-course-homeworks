#include<stdexcept>
#include<vector>
#include<cmath>
#include<algorithm>
#include<numeric>
#include<unordered_map>
#include"types.hpp"
#include"data_utils.hpp"

#pragma once

using namespace datatypes;

namespace KNN{

    class Metric //* Euclidian
    {
        public:
            virtual double operator()(const Series& first, const Series& second) const
            {
                if(first.size() != second.size())
                    throw std::length_error("The size of the Series is not the same");
    
                size_t cols = first.size();
             
                double distance;
    
                distance = std::accumulate(first.begin(), first.end(), 0. ,[&second, i = 0](double dist, double val) mutable
                {
                    return dist + std::pow(val - second[i++], 2);
                });
    
                return std::sqrt(distance);
            }
    
            virtual ~Metric() = default;
    };
    
    class KNNClassifier
    {
        private:
            DataFrame X_train;
            Predictions Y_train;
    
            bool fitted = false;

            std::vector<int> classes;
    
            int k;
    
        Distances calc_distances(const Series& row)
        {
            size_t rows = X_train.size();
    
            Metric metric;
    
            Distances distances;
    
            distances.reserve(rows);
    
            for(int i = 0; i<rows; ++i)
                distances.emplace_back(Y_train[i], metric(row, X_train[i]));  
    
            return distances;
        }
        public:
            KNNClassifier(int n_neighbors) : k{n_neighbors}
            {}
    
            void fit(const DataFrame& X_train, const Predictions& Y_train)
            {
                if(X_train.size() != Y_train.size())
                    throw std::length_error("The number of rows is different");
    
                if(X_train.empty())
                    throw std::invalid_argument("The data is empty");
    
                this->X_train = X_train;
                this->Y_train = Y_train;
    
                auto Y_copy = Y_train;

                fitted = true;
    
                //classes = {Y_copy.begin(), std::unique(Y_copy.begin(), Y_copy.end())};
            }
    
            Predictions predict(const DataFrame& X_test)
            {
                if(!fit)
                    throw std::logic_error("Fit was not called");
    
                size_t rows = X_test.size();
    
                if(!rows)
                    throw std::invalid_argument("Df size = 0");
    
                size_t cols = X_test.begin()->size();
    
                if(cols != X_train.begin()->size())
                    throw std::length_error("The number of columns does not match");
    
                Predictions Y_pred;
    
                Y_pred.reserve(rows);
    
                for(int i = 0; i<rows;++i)
                {
                    auto distances = calc_distances(X_test[i]);
    
                    std::nth_element(distances.begin(), distances.begin() + k, distances.end(), 
                    [](const Distances::value_type& first, const Distances::value_type& second)
                    {
                        return first.second < second.second;
                    });
    
                    std::unordered_map<int, int> counter;
    
                    for(int j = 0; j<k; ++j)
                        counter[distances[j].first]++;

                    Y_pred.emplace_back(std::max_element(counter.begin(), counter.end(), 
                    [](const auto& a, const auto& b)
                    {
                        return a.second < b.second;
                    })->first);
                }
                return Y_pred;
            }
    
    };
}
