#pragma once

#include "latent_variables.hpp"
#include "output/tableprinter.hpp"
#include "tests/nhst.hpp"

#include <algorithm>

class Results {
protected:
    std::vector<std::string> _X_names;
    std::string _model_name;
    std::string _model_type;
    LatentVariables _z;
    Eigen::VectorXd _z_values;
    Eigen::VectorXd _results; // FIXME: OptimizeResult type in Python (da scipy) ma viene utilizzato solo l'array
                                  // non gli altri oggetti al suo interno
    Eigen::MatrixXd _data;        ///< Predicted values for the time series and length-adjusted time series
    int _index;                   // FIXME: Può essere anche una lista di indici
    bool _multivariate_model;
    std::function<double(Eigen::VectorXd)> _objective_object; ///< Likelihood or posterior
    std::string _method;
    uint8_t _z_hide;
    int _max_lag;
    Eigen::VectorXd _ihessian;
    Eigen::VectorXd _signal;
    // FIXME: _scores, _states e _states_var non sono mai usate con i modelli ARMA
    Eigen::VectorXd _scores;
    Eigen::VectorXd _states;
    Eigen::VectorXd _states_var;
    size_t _data_length;
    std::string _data_name;
    uint8_t _rounding_points;
    double _aic;
    double _bic;

    Results(std::vector<std::string> data_name, std::vector<std::string> X_names, std::string model_name,
            const std::string& model_type, const LatentVariables& latent_variables, Eigen::VectorXd results,
            Eigen::MatrixXd data, int index, bool multivariate_model,
            std::function<double(Eigen::VectorXd)> objective_object, std::string method, bool z_hide, int max_lag,
            Eigen::VectorXd ihessian = Eigen::VectorXd::Zero(0), Eigen::VectorXd signal = Eigen::VectorXd::Zero(0),
            Eigen::VectorXd scores = Eigen::VectorXd::Zero(0),
            Eigen::VectorXd states = Eigen::VectorXd::Zero(0), Eigen::VectorXd states_var = Eigen::VectorXd::Zero(0));
};

class MLEResults : Results {
private:
    double _loglik;

public:
    MLEResults(std::vector<std::string> data_name, std::vector<std::string> X_names, std::string model_name,
               const std::string& model_type, LatentVariables latent_variables, Eigen::VectorXd results,
               Eigen::MatrixXd data, int index, bool multivariate_model,
               std::function<double(Eigen::VectorXd)> objective_object, std::string method, bool z_hide, int max_lag,
               Eigen::VectorXd ihessian = Eigen::VectorXd::Zero(0), Eigen::VectorXd scores = Eigen::VectorXd::Zero(0),
               Eigen::VectorXd states     = Eigen::VectorXd::Zero(0),
               Eigen::VectorXd states_var = Eigen::VectorXd::Zero(0));

    /**
     * @brief Overload of the stream operation
     * @param stream The output stream object
     * @param mle_results The MLEResults object to stream
     * @return The output stream
     */
    friend std::ostream& operator<<(std::ostream& stream, const MLEResults& mle_results);

    void summary_without_hessian() const;
};