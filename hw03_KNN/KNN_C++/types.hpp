#include<vector>

#pragma once

namespace datatypes{

    using DataFrame = std::vector<std::vector<double>>;
        
    using Series = std::vector<double>;
    
    using Predictions = std::vector<int>;
    
    using Distances = std::vector<std::pair<int, double>>;

}
    