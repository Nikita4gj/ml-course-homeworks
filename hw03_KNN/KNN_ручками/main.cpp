#include<iostream>

#include "knn.hpp"
#include"types.hpp"
#include"data_utils.hpp"
#include"scaler.hpp"

int main()
{
    auto [data, targets] = utils::read_csv("data/wine.csv");
    
    auto [X_train, X_test, y_train, y_test] = utils::train_test_split(data, targets);
        
    StandardScaler scaler;
    
    scaler.fit(X_train);
    
    auto X_train_scaled = scaler.transform(X_train);
    
    auto X_test_scaled = scaler.transform(X_test);
    
    int optimized_k = -1;

    double max_accuracy = 0;

    for(int k = 1; k<=X_train.size()/10; ++k)
    {
        auto knn = KNN::KNNClassifier(k);
    
        knn.fit(X_train_scaled, y_train);
    
        auto y_pred = knn.predict(X_test_scaled);
    
        int count_right = 0;
    
        for(int i = 0; i<y_pred.size(); ++i)
            count_right+=(y_pred[i]==y_test[i]);
    
        double accuracy = (double)count_right/y_pred.size();
        std::cout << "Accuracy for " << k << " neighbors: " << accuracy << "\n";

        max_accuracy = accuracy>max_accuracy?optimized_k = k, accuracy: max_accuracy; 
    }

    std::cout << "Max accuracy: " << max_accuracy << " whith optimized_k = " << optimized_k << "\n";
    return 0;

}