#include "arima/arima.hpp"

#include "data_check.hpp"
#include "families/family.hpp"

ARIMA::ARIMA(const std::vector<double>& data, const std::vector<double>& index, size_t ar, size_t ma, size_t integ,
             const Family& family)
    : TSM{"ARIMA"}, _ar{ar}, _ma{ma}, _integ{integ} {
    // Latent Variable information
    _z_no               = _ar + _ma + 2;
    _max_lag            = std::max(_ar, _ma);
    _z_hide             = 0;
    _supported_methods  = {"MLE", "PML", "Laplace", "M-H", "BBVI"};
    _default_method     = "MLE";
    _multivariate_model = false;

    // Format the data
    CheckedData c_data = data_check(data, index);
    _data              = c_data.transformed_data;
    _data_name         = c_data.data_name;
    _index             = c_data.data_index;

    // Difference data
    for (size_t order{0}; order < _integ; order++)
        _data = diff(_data);
    _data_name.at(0) = "Differenced " + _data_name.at(0);

    _x = ar_matrix();
    create_latent_variables();

    _family             = family;
    FamilyAttributes fa = family.setup(); // TODO: Creare il metodo virtual in family che ritorna delle info nulle
    _model_name_short   = fa.name;
    _link               = fa.link;
    _scale              = fa.scale;
    _shape              = fa.shape;
    _skewness           = fa.skewness;
    _mean_transform     = fa.mean_transform;
    _cythonized         = fa.cythonized;
    _model_name         = _model_name_short + " ARIMA(" + std::to_string(_ar) + "," + std::to_string(_integ) + "," +
                  std::to_string(_ma) + ")";

    // Build any remaining latent variables that are specific to the family chosen
    // TODO: continue
}

ARIMA::ARIMA(const std::map<std::string, std::vector<double>>& data, const std::vector<double>& index,
             const std::string& target, size_t ar, size_t ma, size_t integ, const Family& family)
    : TSM{"ARIMA"}, _ar{ar}, _ma{ma}, _integ{integ} {
    // Latent Variable information
    _z_no               = _ar + _ma + 2;
    _max_lag            = std::max(_ar, _ma);
    _z_hide             = 0;
    _supported_methods  = {"MLE", "PML", "Laplace", "M-H", "BBVI"};
    _default_method     = "MLE";
    _multivariate_model = false;

    // Format the data
    CheckedData c_data = data_check(data, index, target);
    _data              = c_data.transformed_data;
    _data_name         = c_data.data_name;
    _index             = c_data.data_index;

    // TODO: continue
}

Eigen::MatrixXd ARIMA::ar_matrix() {
    Eigen::MatrixXd X{Eigen::VectorXd::Zero(_data_length - _max_lag)};
    std::vector<double> data{};

    if (_ar != 0) {
        for (Eigen::Index i{0}; i < _ar; i++) {
            std::copy(_data.begin() + _max_lag - i - 1, _data.begin() + i - 1, std::back_inserter(data));
            X.row(i + 1) = Eigen::VectorXd::Map(data.data(), data.size());
        }
    }

    return X;
}

void ARIMA::create_latent_variables() {
    _latent_variables.add_z("Constant", Normal(0, 3), Normal(0, 3));

    for (size_t ar_terms{0}; ar_terms < _ar; ar_terms++)
        _latent_variables.add_z("AR(" + std::to_string(ar_terms + 1) + ")", Normal(0, 0.5), Normal(0.3));

    for (size_t ma_terms{0}; ma_terms < _ma; ma_terms++)
        _latent_variables.add_z("MA(" + std::to_string(ma_terms + 1) + ")", Normal(0, 0.5), Normal(0.3));
}

std::tuple<double, double, double> ARIMA::get_scale_and_shape(Eigen::VectorXd transformed_lvs) {
    double model_shape    = 0;
    double model_scale    = 0;
    double model_skewness = 0;

    if (_scale) {
        if (_shape) {
            model_shape = transformed_lvs(Eigen::last);
            model_scale = transformed_lvs(Eigen::last - 1);
        } else
            model_scale = transformed_lvs(Eigen::last);
    }

    if (_skewness)
        model_skewness = transformed_lvs(Eigen::last - 2);

    return std::make_tuple(model_scale, model_shape, model_skewness);
}

std::tuple<Eigen::VectorXd, Eigen::VectorXd, Eigen::VectorXd>
ARIMA::get_scale_and_shape_sim(Eigen::MatrixXd transformed_lvs) {
    Eigen::VectorXd model_shape    = Eigen::VectorXd::Zero(transformed_lvs.cols());
    Eigen::VectorXd model_scale    = Eigen::VectorXd::Zero(transformed_lvs.cols());
    Eigen::VectorXd model_skewness = Eigen::VectorXd::Zero(transformed_lvs.cols());

    if (_scale) {
        if (_shape) {
            // Apply trasform() to every element inside the matrix last row
            model_shape = transformed_lvs(Eigen::last, Eigen::all);
            std::transform(model_shape.begin(), model_shape.end(), model_shape,
                           _latent_variables.get_z_list().back().get_prior()->get_transform());

            model_scale = transformed_lvs(Eigen::last - 1, Eigen::all);
            std::transform(model_scale.begin(), model_scale.end(), model_scale,
                           _latent_variables.get_z_list().back().get_prior()->get_transform());
        } else {
            model_scale = transformed_lvs(Eigen::last, Eigen::all);
            std::transform(model_scale.begin(), model_scale.end(), model_scale,
                           _latent_variables.get_z_list().back().get_prior()->get_transform());
        }
    }

    if (_skewness) {
        model_skewness = transformed_lvs(Eigen::last - 2, Eigen::all);
        std::transform(model_skewness.begin(), model_skewness.end(), model_skewness,
                       _latent_variables.get_z_list().back().get_prior()->get_transform());
    }

    return std::move(std::make_tuple(model_scale, model_shape, model_skewness));
}

std::tuple<Eigen::VectorXd, Eigen::VectorXd> ARIMA::normal_model(Eigen::VectorXd beta) {
    std::vector<double> data;
    std::copy(_data.begin() + _max_lag, _data.end(), std::back_inserter(data));
    Eigen::VectorXd Y{Eigen::VectorXd::Map(data.data(), data.size())};

    // Transform latent variables
    Eigen::VectorXd z(beta.size());
    for (Eigen::Index i{0}; i < beta.size(); i++) {
        z[i] = _latent_variables.get_z_list()[i].get_prior()->get_transform()(beta[i]);
    }

    // Constant and AR terms
    Eigen::VectorXd mu;
    if (_ar != 0)
        mu = _x.transpose().array() * z(Eigen::seq(0, -_family_z_no - _ma)).array();
    else
        mu = Eigen::VectorXd::Zero(Y.size()) * z[0];

    // MA terms
    if (_ma != 0)
        mu = arima_recursion_normal(z, mu, Y, _max_lag, Y.size(), _ar, _ma);

    return {mu, Y};
}

std::tuple<Eigen::VectorXd, Eigen::VectorXd> ARIMA::non_normal_model(Eigen::VectorXd beta) {
    std::vector<double> data;
    std::copy(_data.begin() + _max_lag, _data.end(), std::back_inserter(data));
    Eigen::VectorXd Y{Eigen::VectorXd::Map(data.data(), data.size())};

    // Transform latent variables
    Eigen::VectorXd z(beta.size());
    for (Eigen::Index i{0}; i < beta.size(); i++) {
        z[i] = _latent_variables.get_z_list()[i].get_prior()->get_transform()(beta[i]);
    }

    // Constant and AR terms
    Eigen::VectorXd mu;
    if (_ar != 0)
        mu = _x.transpose().array() * z(Eigen::seq(0, -_family_z_no - _ma)).array();
    else
        mu = Eigen::VectorXd::Zero(Y.size()) * z[0];

    // MA terms
    if (_ma != 0) { // TODO: Rivedere la definizione di _link (in questo caso usata per un vettore!)
        Eigen::VectorXd link_mu(mu.size());
        std::transform(mu.begin(), mu.end(), link_mu.begin(), _link);
        mu = arima_recursion(z, mu, link_mu, Y, _max_lag, Y.size(), _ar, _ma);
    }

    return {mu, Y};
}

std::tuple<Eigen::VectorXd, Eigen::VectorXd> ARIMA::mb_normal_model(Eigen::VectorXd beta, size_t mini_batch) {
    int rand_int = rand() % (_data.size() - mini_batch - _max_lag + 1);
    std::vector<double> sample(mini_batch);
    std::iota(sample.begin(), sample.end(), rand_int);

    std::vector<double> data;
    std::copy(_data.begin() + _max_lag, _data.end(), std::back_inserter(data));
    Eigen::VectorXd Y{Eigen::VectorXd::Map(data.data(), data.size())};
    Y = Y(Eigen::all, sample);
    Eigen::MatrixXd X(_x.rows(), sample.size());
    for (Eigen::Index i{0}; i < _x.rows(); i++)
        X.row(i) = _x.row(i)(Eigen::all, sample);

    // Transform latent variables
    Eigen::VectorXd z(beta.size());
    for (Eigen::Index i{0}; i < beta.size(); i++) {
        z[i] = _latent_variables.get_z_list()[i].get_prior()->get_transform()(beta[i]);
    }

    // Constant and AR terms
    Eigen::VectorXd mu;
    if (_ar != 0)
        mu = _x.transpose().array() * z(Eigen::seq(0, -_family_z_no - _ma)).array();
    else
        mu = Eigen::VectorXd::Zero(Y.size()) * z[0];

    // MA terms
    if (_ma != 0)
        mu = arima_recursion_normal(z, mu, Y, _max_lag, Y.size(), _ar, _ma);

    return {mu, Y};
}

std::tuple<Eigen::VectorXd, Eigen::VectorXd> ARIMA::mb_non_normal_model(Eigen::VectorXd beta, size_t mini_batch) {
    int rand_int = rand() % (_data.size() - mini_batch - _max_lag + 1);
    std::vector<double> sample(mini_batch);
    std::iota(sample.begin(), sample.end(), rand_int);

    std::vector<double> data;
    std::copy(_data.begin() + _max_lag, _data.end(), std::back_inserter(data));
    Eigen::VectorXd Y{Eigen::VectorXd::Map(data.data(), data.size())};
    Y = Y(Eigen::all, sample);
    Eigen::MatrixXd X(_x.rows(), sample.size());
    for (Eigen::Index i{0}; i < _x.rows(); i++)
        X.row(i) = _x.row(i)(Eigen::all, sample);

    // Transform latent variables
    Eigen::VectorXd z(beta.size());
    for (Eigen::Index i{0}; i < beta.size(); i++) {
        z[i] = _latent_variables.get_z_list()[i].get_prior()->get_transform()(beta[i]);
    }

    // Constant and AR terms
    Eigen::VectorXd mu;
    if (_ar != 0)
        mu = _x.transpose().array() * z(Eigen::seq(0, -_family_z_no - _ma)).array();
    else
        mu = Eigen::VectorXd::Zero(Y.size()) * z[0];

    // MA terms
    if (_ma != 0) { // TODO: Rivedere la definizione di _link (in questo caso usata per un vettore!)
        Eigen::VectorXd link_mu(mu.size());
        std::transform(mu.begin(), mu.end(), link_mu.begin(), _link);
        mu = arima_recursion(z, mu, link_mu, Y, _max_lag, Y.size(), _ar, _ma);
    }

    return {mu, Y};
}
