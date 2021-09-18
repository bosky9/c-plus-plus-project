#pragma once

#include "latent_variables.hpp"
#include "output/tableprinter.hpp"
#include "tests/nhst.hpp"

#include <algorithm>
#include <cmath>
#include <utility>

class Results {
protected:
    std::vector<std::string> _X_names;
    std::string _model_name;
    std::string _model_type;
    LatentVariables _z;
    Eigen::VectorXd _z_values;
    Eigen::VectorXd _results; // FIXME: OptimizeResult type in Python (da scipy) ma viene utilizzato solo l'array
                              // non gli altri oggetti al suo interno
    Eigen::MatrixXd _data;    ///< Predicted values for the time series and length-adjusted time series
    std::vector<size_t> _index;
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

    /**
     * @brief Constructor for Results
     * @param data_name
     * @param X_names
     * @param model_name
     * @param model_type
     * @param latent_variables
     * @param results
     * @param data
     * @param index
     * @param multivariate_model
     * @param objective_object
     * @param method
     * @param z_hide
     * @param max_lag
     * @param ihessian
     * @param scores
     * @param states
     * @param states_var
     */
    Results(std::vector<std::string> data_name, std::vector<std::string> X_names, std::string model_name,
            const std::string& model_type, const LatentVariables& latent_variables, Eigen::VectorXd results,
            Eigen::MatrixXd data, std::vector<size_t> index, bool multivariate_model,
            std::function<double(Eigen::VectorXd)> objective_object, std::string method, bool z_hide, int max_lag,
            Eigen::VectorXd ihessian = Eigen::VectorXd::Zero(0), Eigen::VectorXd signal = Eigen::VectorXd::Zero(0),
            Eigen::VectorXd scores = Eigen::VectorXd::Zero(0), Eigen::VectorXd states = Eigen::VectorXd::Zero(0),
            Eigen::VectorXd states_var = Eigen::VectorXd::Zero(0));

    static [[nodiscard]] double round_to(double x, uint8_t rounding_points);

public:
    virtual void summary(bool transformed) = 0;
};

class MLEResults : Results {
private:
    double _loglik;

    /**
     * @brief Prints results with hessian
     * @param transformed
     */
    void summary_with_hessian(bool transformed = true) const;

    /**
     * @brief Prints results without hessian
     */
    void summary_without_hessian() const;

public:
    /**
     * @brief Constructor for MLEResults
     * @param data_name
     * @param X_names
     * @param model_name
     * @param model_type
     * @param latent_variables
     * @param results
     * @param data
     * @param index
     * @param multivariate_model
     * @param objective_object
     * @param method
     * @param z_hide
     * @param max_lag
     * @param ihessian
     * @param scores
     * @param states
     * @param states_var
     */
    MLEResults(std::vector<std::string> data_name, std::vector<std::string> X_names, std::string model_name,
               const std::string& model_type, const LatentVariables& latent_variables, Eigen::VectorXd results,
               Eigen::MatrixXd data, std::vector<size_t> index, bool multivariate_model,
               std::function<double(Eigen::VectorXd)> objective_object, std::string method, bool z_hide, int max_lag,
               Eigen::VectorXd ihessian = Eigen::VectorXd::Zero(0), Eigen::VectorXd signal = Eigen::VectorXd::Zero(0),
               Eigen::VectorXd scores = Eigen::VectorXd::Zero(0), Eigen::VectorXd states = Eigen::VectorXd::Zero(0),
               Eigen::VectorXd states_var = Eigen::VectorXd::Zero(0));

    /**
     * @brief Stream operator for MLEResults
     * @param stream Output stream
     * @param mleresults MLEResults object
     * @return Output stream
     */
    friend std::ostream& operator<<(std::ostream& stream, const MLEResults& mleresults);

    /**
     * @brief Prints results
     * @param transformed
     */
    void summary(bool transformed) override;
};