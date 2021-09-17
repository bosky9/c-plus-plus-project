#include "results.hpp"

Results::Results(std::vector<std::string> data_name, std::vector<std::string> X_names, std::string model_name,
                 const std::string& model_type, const LatentVariables& latent_variables, std::vector<double> results,
                 Eigen::MatrixXd data, int index, bool multivariate_model,
                 std::function<double(Eigen::VectorXd)> objective_object, std::string method, bool z_hide, int max_lag,
                 Eigen::VectorXd ihessian, Eigen::VectorXd scores, Eigen::VectorXd states, Eigen::VectorXd states_var)
    : _X_names{std::move(std::move(std::move(std::move(std::move(X_names)))))}, _model_name{std::move(
                                                                                        std::move(model_name))},
      _model_type{model_type}, _z{latent_variables}, _z_values{latent_variables.get_z_values()}, _results{std::move(
                                                                                                         results)},
      _data{std::move(std::move(data))}, _index{index}, _multivariate_model{multivariate_model},
      _objective_object{std::move(objective_object)}, _method{std::move(method)}, _z_hide{static_cast<uint8_t>(z_hide)},
      _max_lag{max_lag}, _ihessian{std::move(ihessian)}, _scores{std::move(scores)}, _states{std::move(states)},
      _states_var{std::move(states_var)} {
    if (_multivariate_model) {
        _data_length = _data.row(0).size();
    } else {
        _data_length = _data.rows();
    }

    std::ostringstream oss;
    std::copy(data_name.begin(), data_name.end(), std::ostream_iterator<std::string>(oss, ","));
    _data_name = oss.str();

    if (_model_type == "LLT" || model_type == "LLEV")
        _rounding_points = 10;
    else
        _rounding_points = 4;
}

MLEResults::MLEResults(std::vector<std::string> data_name, std::vector<std::string> X_names, std::string model_name,
                       const std::string& model_type, LatentVariables latent_variables, std::vector<double> results,
                       Eigen::MatrixXd data, int index, bool multivariate_model,
                       std::function<double(Eigen::VectorXd)> objective_object, std::string method, bool z_hide,
                       int max_lag, Eigen::VectorXd ihessian, Eigen::VectorXd scores, Eigen::VectorXd states,
                       Eigen::VectorXd states_var)
    : Results{std::move(data_name),
              X_names,
              model_name,
              model_type,
              latent_variables,
              results,
              data,
              index,
              multivariate_model,
              std::move(objective_object),
              method,
              z_hide,
              max_lag,
              ihessian,
              scores,
              states,
              std::move(states_var)} {
    if (_method == "MLE" || _method == "OLS") {
        _loglik = -_objective_object(_z_values);
        _aic    = 2 * _z_values.size() + 2 * _objective_object(_z_values);
        _bic    = 2 * _objective_object(_z_values) + _z_values.size() + log(_data_length);
    } else if (_method == "PML") {
        _aic = 2 * _z_values.size() + 2 * _objective_object(_z_values);
        _bic = 2 * _objective_object(_z_values) + _z_values.size() + log(_data_length);
    }
}
