#include "results.hpp"

Results::Results(std::vector<std::string> data_name, std::vector<std::string> X_names, std::string model_name,
                 const std::string& model_type, const LatentVariables& latent_variables, Eigen::VectorXd results,
                 Eigen::MatrixXd data, int index, bool multivariate_model,
                 std::function<double(Eigen::VectorXd)> objective_object, std::string method, bool z_hide, int max_lag,
                 Eigen::VectorXd ihessian, Eigen::VectorXd signal, Eigen::VectorXd scores, Eigen::VectorXd states,
                 Eigen::VectorXd states_var)
    : _X_names{std::move(std::move(std::move(std::move(std::move(X_names)))))}, _model_name{std::move(
                                                                                        std::move(model_name))},
      _model_type{model_type}, _z{latent_variables}, _z_values{latent_variables.get_z_values()}, _results{std::move(
                                                                                                         results)},
      _data{std::move(std::move(data))}, _index{index}, _multivariate_model{multivariate_model},
      _objective_object{std::move(objective_object)}, _method{std::move(method)}, _z_hide{static_cast<uint8_t>(z_hide)},
      _max_lag{max_lag}, _ihessian{std::move(ihessian)}, _signal{std::move(signal)}, _scores{std::move(scores)},
      _states{std::move(states)}, _states_var{std::move(states_var)} {
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
                       const std::string& model_type, LatentVariables latent_variables, Eigen::VectorXd results,
                       Eigen::MatrixXd data, int index, bool multivariate_model,
                       std::function<double(Eigen::VectorXd)> objective_object, std::string method, bool z_hide,
                       int max_lag, Eigen::VectorXd ihessian = nullptr, Eigen::VectorXd signal = nullptr,
                       Eigen::VectorXd scores = nullptr, Eigen::VectorXd states = nullptr,
                       Eigen::VectorXd states_var = nullptr)
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
              signal,
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

inline std::ostream& operator<<(std::ostream& stream, const MLEResults& mle_results) {
    if (mle_results._method == "MLE")
        stream << "MLE Results Object";
    else if (mle_results._method == "OLS")
        stream << "OLS Results Object";
    else
        stream << "PML Results Object";
    stream << "\n=========================="
              "\nDependent variable: " << mle_results._data_name << "\nRegressors: ";
    for (const std::string& s : mle_results._X_names)
        stream << s << " ";
    stream << "\n=========================="
              "\nLatent Variable Attributes: ";
    if (mle_results._ihessian != nullptr)
        stream << "\n.ihessian: Inverse Hessian";
    stream << "\n.z : LatentVariables() object";
    if (mle_results._results != nullptr)
        stream << "\n.results : optimizer results";
    stream << "\n\nImplied Model Attributes: "
              "\n.aic: Akaike Information Criterion"
              "\n.bic: Bayesian Information Criterion"
              "\n.data: Model Data"
              "\n.index: Model Index";
    if (mle_results._method == "MLE" || mle_results._method == "OLS")
        stream << "\n.loglik: Loglikelihood";
    if (mle_results._scores != nullptr)
        stream << "\n.scores: Model Scores";
    if (mle_results._signal != nullptr)
        stream << "\n.signal: Model Signal";
    if (mle_results._states != nullptr)
        stream << "\n.states: Model States";
    if (mle_results._states_var != nullptr)
        stream << "\n.states_var: Model State Variances";
    stream << "\n.results : optimizer results"
              "\n\nMethods: "
              "\n.summary() : printed results";
    return stream;
}

void MLEResults::summary_without_hessian() const {
    Eigen::VectorXd t_z = _z.get_z_values(true);
    std::cout << "\nHessian not invertible! Consider a different model specification.\n";
    // initialize data
    std::list<std::map<std::string, std::string>> data;
    std::vector<std::string> z_names = _z.get_z_names();
    std::vector<std::function<double(double)>> z_transforms = _z.get_z_transforms();
    for (size_t i{0}; i < z_names.size(); i++)
        data.push_back({{"z_name", z_names.at(i)},
                        {"z_value", std::to_string(round(z_transforms.at(i)(_results(i))*10000)/10000)}});
    // create fmts
    std::list<std::tuple<std::string, std::string, int>> fmt = {
            std::make_tuple("Latent Variable", "z_name", 40),
            std::make_tuple("Estimate", "z_value", 10)
    };
    std::list<std::tuple<std::string, std::string, int>> model_fmt = {
            std::make_tuple(_model_name, "model_details", 55),
            std::make_tuple("", "model_results", 50)
    };
    // initialize model_details
    std::list<std::map<std::string, std::string>> model_details;
    std::string obj_desc = (_method == "MLE") ? "Log Likelihood: " : "Unnormalized Log Posterior: ";
    obj_desc += std::to_string(round(-_objective_object(_results)*10000)/10000);
    model_details.push_back({{"model_details", "Dependent Variable: " + _data_name},
                             {"model_results", "Method: " + _method}});
    model_details.push_back({{"model_details", "Start Date: " + std::to_string(_index.at(_max_lag))},
                             {"model_results", obj_desc}});
    model_details.push_back({{"model_details", "End Date: " + std::to_string(_index.at(_index.size()-1))},
                             {"model_results", "AIC: " + std::to_string(_aic)}});
    model_details.push_back({{"model_details", "Number of observations: " + std::to_string(_data_length)},
                             {"model_results", "BIC: " + std::to_string(_bic)}});
    std::cout << "\n" << TablePrinter(model_fmt, " ", "=")._call_(model_details) << "\n"
              << std::string(106, '=') << "\n" << TablePrinter(fmt, " ", "=")._call_(data) << "\n"
              << std::string(106, '=');
    if (_model_name.find("Skewt") != std::string::npos)
        std::cout << "\nWARNING: Skew t distribution is not well-suited for MLE or MAP inference"
                     "\nWorkaround 1: Use a t-distribution instead for MLE/MAP"
                     "\nWorkaround 2: Use M-H or BBVI inference for Skew t distribution";
}