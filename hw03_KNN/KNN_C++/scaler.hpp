#include<cmath>
#include<stdexcept>
#include"types.hpp"

#pragma once

using namespace datatypes;

class StandardScaler
    {   
        private:
            Series means;
            Series stds;
    
            bool fitted = false;
        public:
            void fit(const DataFrame& X_train)
            {
                size_t rows = X_train.size();
    
                if(!rows)
                    throw std::invalid_argument("Df size = 0");
    
                size_t cols = X_train.begin()->size();
    
                means.resize(cols, 0);
                stds.resize(cols, 0);
    
                for(int j = 0; j<cols; ++j)
                {
                    for(int i = 0; i<rows; ++i)
                        means[j]+=X_train[i][j];                                        
                    means[j]/=rows;
                }
    
                for(int j = 0; j<cols; ++j)
                {
                    for(int i = 0; i<rows; ++i)
                        stds[j]+=std::pow(X_train[i][j]-means[j], 2);                                        
                    
                    stds[j]/=rows;
                    stds[j] = std::sqrt(stds[j]);
                }
    
                fitted = true;
            } 
            
            DataFrame transform(const DataFrame& X) const
            {
                if(!fitted)
                    throw std::logic_error("Fit was not called");
                
                size_t rows = X.size();
    
                size_t cols = X.begin()->size();
    
                if(cols != means.size())
                    throw std::invalid_argument("The number of columns does not match");
        
                DataFrame transformed(rows, Series(cols));
    
    
                for(int j = 0; j<cols; ++j)
                {
                    for(int i = 0; i<rows; ++i)
                        if(stds[j])
                            transformed[i][j] = (X[i][j]-means[j])/stds[j];                
                }
    
                return transformed;
            }
    };