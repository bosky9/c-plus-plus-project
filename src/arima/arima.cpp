#include "arima/arima.hpp"

#include "multivariate_normal.hpp"

ARIMA::ARIMA(const std::vector<double>& data, const std::vector<double>& index, size_t ar, size_t ma, size_t integ,
             const Family& family)
    : TSM{"ARIMA"} {
    // Latent Variable information
    _z_no               = _ar + _ma + 2;
    _max_lag            = std::max(static_cast<int>(_ar), static_cast<int>(_ma));
    _z_hide             = false; // Whether to cutoff latent variables from results table
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

    _family.reset(family.clone());
    FamilyAttributes fa = family.setup();
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
    std::vector<lv_to_build> lvs = _family->build_latent_variables();
    for (size_t no{0}; no < lvs.size(); no++) {
        lv_to_build lv = lvs[no];
        _latent_variables.add_z(std::get<0>(lv), std::get<1>(lv), std::get<2>(lv));
        _latent_variables.set_z_starting_value(1 + no + _ar + _ma, std::get<3>(lv));
    }
    _latent_variables.set_z_starting_value(
            0, _mean_transform(static_cast<double>(std::reduce(_data.begin(), _data.end())) /
                               static_cast<double>(_data.size())));

    _family_z_no = lvs.size();
    _z_no        = _latent_variables.get_z_list().size();
}

ARIMA::ARIMA(const std::map<std::string, std::vector<double>>& data, const std::vector<double>& index,
             const std::string& target, size_t ar, size_t ma, size_t integ, const Family& family)
    : TSM{"ARIMA"}, _ar{ar}, _ma{ma}, _integ{integ} {
    // Latent Variable information
    _z_no               = _ar + _ma + 2;
    _max_lag            = std::max(static_cast<int>(_ar), static_cast<int>(_ma));
    _z_hide             = false;
    _supported_methods  = {"MLE", "PML", "Laplace", "M-H", "BBVI"};
    _default_method     = "MLE";
    _multivariate_model = false;

    // Format the data
    CheckedData c_data = data_check(data, index, target);
    _data              = c_data.transformed_data;
    _data_name         = c_data.data_name;
    _index             = c_data.data_index;

    // Difference data
    for (size_t order{0}; order < _integ; order++)
        _data = diff(_data);
    _data_name.at(0) = "Differenced " + _data_name.at(0);

    _x = ar_matrix();
    create_latent_variables();

    _family.reset(family.clone());
    FamilyAttributes fa = family.setup();
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
    std::vector<lv_to_build> lvs = _family->build_latent_variables();
    for (size_t no{0}; no < lvs.size(); no++) {
        lv_to_build lv = lvs[no];
        _latent_variables.add_z(std::get<0>(lv), std::get<1>(lv), std::get<2>(lv));
        _latent_variables.set_z_starting_value(1 + no + _ar + _ma, std::get<3>(lv));
    }
    _latent_variables.set_z_starting_value(
            0, _mean_transform(static_cast<double>(std::reduce(_data.begin(), _data.end())) /
                               static_cast<double>(_data.size())));

    _family_z_no = lvs.size();
    _z_no        = _latent_variables.get_z_list().size();
}

Eigen::MatrixXd ARIMA::ar_matrix() {
    Eigen::MatrixXd X{Eigen::VectorXd::Zero(static_cast<Eigen::Index>(_data_length - _max_lag))};
    std::vector<double> data{};

    if (_ar != 0) {
        for (Eigen::Index i{0}; i < _ar; i++) {
            std::copy(_data.begin() + _max_lag - i - 1, _data.end() - i - 1, std::back_inserter(data));
            X.row(i + 1) = Eigen::VectorXd::Map(data.data(), static_cast<Eigen::Index>(data.size()));
        }
    }

    return X;
}

void ARIMA::create_latent_variables() {
    Normal n1{Normal(0, 3)};
    _latent_variables.add_z("Constant", reinterpret_cast<Family*>(&n1), reinterpret_cast<Family*>(&n1));

    n1 = Normal(0, 0.5);
    Normal n2{Normal(0.3)};
    for (size_t ar_terms{0}; ar_terms < _ar; ar_terms++)
        _latent_variables.add_z("AR(" + std::to_string(ar_terms + 1) + ")", reinterpret_cast<Family*>(&n1),
                                reinterpret_cast<Family*>(&n2));

    for (size_t ma_terms{0}; ma_terms < _ma; ma_terms++)
        _latent_variables.add_z("MA(" + std::to_string(ma_terms + 1) + ")", reinterpret_cast<Family*>(&n1),
                                reinterpret_cast<Family*>(&n2));
}

std::tuple<double, double, double> ARIMA::get_scale_and_shape(const Eigen::VectorXd& transformed_lvs) const {
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
ARIMA::get_scale_and_shape_sim(const Eigen::MatrixXd& transformed_lvs) const {
    Eigen::VectorXd model_shape    = Eigen::VectorXd::Zero(transformed_lvs.cols());
    Eigen::VectorXd model_scale    = Eigen::VectorXd::Zero(transformed_lvs.cols());
    Eigen::VectorXd model_skewness = Eigen::VectorXd::Zero(transformed_lvs.cols());

    std::vector<double> s;
    if (_scale) {
        if (_shape) {
            // Apply trasform() to every element inside the matrix last row
            model_shape = transformed_lvs(Eigen::last, Eigen::all);
            s           = std::vector<double>(&model_shape[0], model_shape.data());
            std::transform(s.begin(), s.end(), s.begin(), [this](double n) {
                return _latent_variables.get_z_list()[-1].get_prior()->get_transform()(n);
            });
            model_shape = Eigen::VectorXd::Map(s.data(), static_cast<Eigen::Index>(s.size()));
        }

        model_scale = transformed_lvs(Eigen::last - 1, Eigen::all);
        s           = std::vector<double>(&model_scale[0], model_scale.data());
        std::transform(s.begin(), s.end(), s.begin(),
                       [this](double n) { return _latent_variables.get_z_list()[-2].get_prior()->get_transform()(n); });
        model_scale = Eigen::VectorXd::Map(s.data(), static_cast<Eigen::Index>(s.size()));
    }

    if (_skewness) {
        model_skewness = transformed_lvs(Eigen::last - 2, Eigen::all);
        s              = std::vector<double>(&model_skewness[0], model_skewness.data());
        std::transform(s.begin(), s.end(), s.begin(),
                       [this](double n) { return _latent_variables.get_z_list()[-3].get_prior()->get_transform()(n); });
        model_skewness = Eigen::VectorXd::Map(s.data(), static_cast<Eigen::Index>(s.size()));
    }

    return std::move(std::make_tuple(model_scale, model_shape, model_skewness));
}

std::pair<Eigen::VectorXd, Eigen::VectorXd> ARIMA::model(const Eigen::VectorXd& beta) const {
    // If Normal family is selected, we use faster likelihood functions
    if (instanceof <Normal>(_family.get()))
        return normal_model(beta);
    // TODO: else if (...) with missing models
    else
        return non_normal_model(beta);
}

std::pair<Eigen::VectorXd, Eigen::VectorXd> ARIMA::mb_model(const Eigen::VectorXd& beta, size_t mini_batch) const {
    // If Normal family is selected, we use faster likelihood functions
    if (instanceof <Normal>(_family.get()))
        return mb_normal_model(beta, mini_batch);
    // TODO: else if (...) with missing models
    else
        return mb_non_normal_model(beta, mini_batch);
}

std::pair<Eigen::VectorXd, Eigen::VectorXd> ARIMA::normal_model(const Eigen::VectorXd& beta) const {
    std::vector<double> data;
    std::copy(_data.begin() + _max_lag, _data.end(), std::back_inserter(data));
    Eigen::VectorXd Y{Eigen::VectorXd::Map(data.data(), static_cast<Eigen::Index>(data.size()))};

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

std::pair<Eigen::VectorXd, Eigen::VectorXd> ARIMA::non_normal_model(const Eigen::VectorXd& beta) const {
    std::vector<double> data;
    std::copy(_data.begin() + _max_lag, _data.end(), std::back_inserter(data));
    Eigen::VectorXd Y{Eigen::VectorXd::Map(data.data(), static_cast<Eigen::Index>(data.size()))};

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
    if (_ma != 0) {
        Eigen::VectorXd link_mu(mu.size());
        std::transform(mu.begin(), mu.end(), link_mu.begin(), _link);
        mu = arima_recursion(z, mu, link_mu, Y, _max_lag, Y.size(), _ar, _ma);
    }

    return {mu, Y};
}

std::pair<Eigen::VectorXd, Eigen::VectorXd> ARIMA::mb_normal_model(const Eigen::VectorXd& beta,
                                                                   size_t mini_batch) const {
    size_t rand_int = rand() % (_data.size() - mini_batch - _max_lag + 1);
    std::vector<double> sample(mini_batch);
    std::iota(sample.begin(), sample.end(), rand_int);

    std::vector<double> data;
    std::copy(_data.begin() + _max_lag, _data.end(), std::back_inserter(data));
    Eigen::VectorXd Y{Eigen::VectorXd::Map(data.data(), static_cast<Eigen::Index>(data.size()))};
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

std::pair<Eigen::VectorXd, Eigen::VectorXd> ARIMA::mb_non_normal_model(const Eigen::VectorXd& beta,
                                                                       size_t mini_batch) const {
    size_t rand_int = rand() % (_data.size() - mini_batch - _max_lag + 1);
    std::vector<double> sample(mini_batch);
    std::iota(sample.begin(), sample.end(), rand_int);

    std::vector<double> data;
    std::copy(_data.begin() + _max_lag, _data.end(), std::back_inserter(data));
    Eigen::VectorXd Y{Eigen::VectorXd::Map(data.data(), static_cast<Eigen::Index>(data.size()))};
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
    if (_ma != 0) {
        Eigen::VectorXd link_mu(mu.size());
        std::transform(mu.begin(), mu.end(), link_mu.begin(), _link);
        mu = arima_recursion(z, mu, link_mu, Y, _max_lag, Y.size(), _ar, _ma);
    }

    return {mu, Y};
}

double ARIMA::neg_loglik(const Eigen::VectorXd& beta) const {
    // If Normal family is selected, we use faster likelihood functions
    if (instanceof <Normal>(_family.get()))
        return normal_neg_loglik(beta);
    // TODO: else if (...) with missing models
    else
        return non_normal_neg_loglik(beta);
}

double ARIMA::mb_neg_loglik(const Eigen::VectorXd& beta, size_t mini_batch) const {
    // If Normal family is selected, we use faster likelihood functions
    if (instanceof <Normal>(_family.get()))
        return normal_mb_neg_loglik(beta, mini_batch);
    // TODO: else if (...) with missing models
    else
        return non_normal_mb_neg_loglik(beta, mini_batch);
}

double ARIMA::normal_neg_loglik(const Eigen::VectorXd& beta) const {
    std::pair<Eigen::VectorXd, Eigen::VectorXd> mu_y = model(beta);
    Eigen::VectorXd scale{{_latent_variables.get_z_priors().back()->get_transform()(beta(Eigen::last))}};
    return -Mvn::logpdf(mu_y.second, mu_y.first, scale).sum();
}

double ARIMA::normal_mb_neg_loglik(const Eigen::VectorXd& beta, size_t mini_batch) const {
    std::pair<Eigen::VectorXd, Eigen::VectorXd> mu_y = mb_model(beta, mini_batch);
    Eigen::VectorXd scale{{_latent_variables.get_z_priors().back()->get_transform()(beta(Eigen::last))}};
    return -Mvn::logpdf(mu_y.second, mu_y.first, scale).sum();
}

double ARIMA::non_normal_neg_loglik(const Eigen::VectorXd& beta) const {
    std::pair<Eigen::VectorXd, Eigen::VectorXd> mu_y = model(beta);
    Eigen::VectorXd transformed_parameters(beta.size());
    std::vector<Family*> priors = _latent_variables.get_z_priors();
    for (size_t k{0}; k < beta.size(); k++)
        transformed_parameters(k) = priors.at(k)->get_transform()(beta(k));
    std::tuple<double, double, double> sc_sh_sk = get_scale_and_shape(transformed_parameters);
    Eigen::VectorXd link_mu(mu_y.first.size());
    std::transform(mu_y.first.begin(), mu_y.first.end(), link_mu.begin(), _link);
    return _family->neg_loglikelihood(mu_y.second, link_mu, std::get<0>(sc_sh_sk), std::get<1>(sc_sh_sk),
                                      std::get<2>(sc_sh_sk));
}

double ARIMA::non_normal_mb_neg_loglik(const Eigen::VectorXd& beta, size_t mini_batch) const {
    std::pair<Eigen::VectorXd, Eigen::VectorXd> mu_y = mb_model(beta, mini_batch);
    Eigen::VectorXd transformed_parameters(beta.size());
    std::vector<Family*> priors = _latent_variables.get_z_priors();
    for (size_t k{0}; k < beta.size(); k++)
        transformed_parameters(k) = priors.at(k)->get_transform()(beta(k));
    std::tuple<double, double, double> sc_sh_sk = get_scale_and_shape(transformed_parameters);
    Eigen::VectorXd link_mu(mu_y.first.size());
    std::transform(mu_y.first.begin(), mu_y.first.end(), link_mu.begin(), _link);
    return _family->neg_loglikelihood(mu_y.second, link_mu, std::get<0>(sc_sh_sk), std::get<1>(sc_sh_sk),
                                      std::get<2>(sc_sh_sk));
}

Eigen::VectorXd ARIMA::mean_prediction(Eigen::VectorXd mu, Eigen::VectorXd Y, size_t h, Eigen::VectorXd t_z) {
    // Create arrays to iterate over
    Eigen::VectorXd Y_exp{std::move(Y)};
    Eigen::VectorXd mu_exp{std::move(mu)};

    // Loop over h time periods
    for (size_t t{0}; t < h; t++) {
        double new_value = t_z[0];

        if (_ar != 0) {
            for (Eigen::Index i{1}; i <= _ar; i++)
                new_value += t_z[i] * Y_exp[-i];
        }

        if (_ma != 0) {
            for (Eigen::Index i{1}; i <= _ma; i++) {
                if (i - 1 >= t)
                    new_value += t_z[i + static_cast<Eigen::Index>(_ar)] * (Y_exp[-i] - _link(mu_exp[-i]));
            }
        }

        std::vector<double> Y_exp_v(&Y_exp[0], Y_exp.data());
        if (_model_name2 == "Exponential")
            Y_exp_v.push_back(1.0 / _link(new_value));
        else
            Y_exp_v.push_back(_link(new_value));
        Y_exp = Eigen::VectorXd::Map(Y_exp_v.data(), static_cast<Eigen::Index>(Y_exp_v.size()));

        // For indexing consistency
        std::vector<double> mu_exp_v(&mu_exp[0], mu_exp.data());
        mu_exp_v.push_back(0.0);
        mu_exp = Eigen::VectorXd::Map(mu_exp_v.data(), static_cast<Eigen::Index>(mu_exp_v.size()));
    }

    // FIXME: del mu_exp si può tradurre con ->
    mu_exp = Eigen::VectorXd::Zero(mu_exp.size());

    return Y_exp;
}

Eigen::MatrixXd ARIMA::sim_prediction(const Eigen::VectorXd& mu, const Eigen::VectorXd& Y, size_t h,
                                      Eigen::VectorXd t_params, size_t simulations) {
    auto scale_shape_skew{get_scale_and_shape(t_params)};

    Eigen::MatrixXd sim_vector{
            Eigen::MatrixXd::Zero(static_cast<Eigen::Index>(simulations), static_cast<Eigen::Index>(h))};

    for (Eigen::Index n{0}; n < simulations; n++) {
        // Create arrays to iterate over
        Eigen::VectorXd Y_exp{Y};
        Eigen::VectorXd mu_exp{mu};

        // Loop over h time periods
        for (Eigen::Index t{0}; t < h; t++) {
            double new_value = t_params[0];

            if (_ar != 0) {
                for (Eigen::Index i{1}; i <= _ar; i++)
                    new_value += t_params[i] * Y_exp[-i];
            }

            if (_ma != 0) {
                for (Eigen::Index i{1}; i <= _ma; i++) {
                    if (i - 1 >= t)
                        new_value += t_params[i + static_cast<Eigen::Index>(_ar)] * (Y_exp[-i] - mu_exp[-i]);
                }
            }

            std::vector<double> Y_exp_v(&Y_exp[0], Y_exp.data());
            if (_model_name2 == "Exponential")
                Y_exp_v.push_back(_family->draw_variable(1.0 / _link(new_value), std::get<0>(scale_shape_skew),
                                                         std::get<1>(scale_shape_skew), std::get<2>(scale_shape_skew),
                                                         1)[0]);
            else
                Y_exp_v.push_back(_family->draw_variable(_link(new_value), std::get<0>(scale_shape_skew),
                                                         std::get<1>(scale_shape_skew), std::get<2>(scale_shape_skew),
                                                         1)[0]);

            std::vector<double> mu_exp_v(&mu_exp[0], mu_exp.data());
            mu_exp_v.push_back(0.0);
            mu_exp = Eigen::VectorXd::Map(mu_exp_v.data(), static_cast<Eigen::Index>(mu_exp_v.size()));

            // For indexing consistency
            std::vector<double> Y_exp_h(Y_exp_v.size());
            std::copy(Y_exp_v.end() - static_cast<Eigen::Index>(h), Y_exp_v.end(), std::back_inserter(Y_exp_h));
            sim_vector.row(n) = Eigen::VectorXd::Map(Y_exp_h.data(), static_cast<Eigen::Index>(Y_exp_h.size()));
        }

        // FIXME: del Y_exp si può tradurre con ->
        Y_exp = Eigen::VectorXd::Zero(Y_exp.size());
        // FIXME: del mu_exp si può tradurre con ->
        mu_exp = Eigen::VectorXd::Zero(mu_exp.size());
    }

    return sim_vector.transpose();
}

Eigen::MatrixXd ARIMA::sim_prediction_bayes(long h, size_t simulations) {
    Eigen::MatrixXd sim_vector{
            Eigen::MatrixXd::Zero(static_cast<Eigen::Index>(simulations), static_cast<Eigen::Index>(h))};

    for (Eigen::Index n{0}; n < simulations; n++) {
        Eigen::VectorXd t_z{draw_latent_variables(1).transpose().row(0)};
        auto mu_Y{model(t_z)};
        for (Eigen::Index i{0}; i < t_z.size(); i++)
            t_z[i] = _latent_variables.get_z_list()[i].get_prior()->get_transform()(t_z[i]);

        auto scale_shape_skew{get_scale_and_shape(t_z)};

        // Create arrays to iterate over
        Eigen::VectorXd Y_exp{mu_Y.second};
        Eigen::VectorXd mu_exp{mu_Y.first};

        // Loop over h time periods
        for (Eigen::Index t{0}; t < h; t++) {
            double new_value = t_z[0];

            if (_ar != 0) {
                for (Eigen::Index i{1}; i <= _ar; i++)
                    new_value += t_z[i] * Y_exp[-i];
            }

            if (_ma != 0) {
                for (Eigen::Index i{1}; i <= _ma; i++) {
                    if (i - 1 >= t)
                        new_value += t_z[i + static_cast<Eigen::Index>(_ar)] * (Y_exp[-i] - mu_exp[-i]);
                }
            }

            std::vector<double> Y_exp_v(&Y_exp[0], Y_exp.data());
            if (_model_name2 == "Exponential")
                Y_exp_v.push_back(_family->draw_variable(1.0 / _link(new_value), std::get<0>(scale_shape_skew),
                                                         std::get<1>(scale_shape_skew), std::get<2>(scale_shape_skew),
                                                         1)[0]);
            else
                Y_exp_v.push_back(_family->draw_variable(_link(new_value), std::get<0>(scale_shape_skew),
                                                         std::get<1>(scale_shape_skew), std::get<2>(scale_shape_skew),
                                                         1)[0]);

            std::vector<double> mu_exp_v(&mu_exp[0], mu_exp.data());
            mu_exp_v.push_back(0.0);
            mu_exp = Eigen::VectorXd::Map(mu_exp_v.data(), static_cast<Eigen::Index>(mu_exp_v.size()));

            // For indexing consistency
            std::vector<double> Y_exp_h(Y_exp_v.size());
            std::copy(Y_exp_v.end() - static_cast<Eigen::Index>(h), Y_exp_v.end(), std::back_inserter(Y_exp_h));
            sim_vector.row(n) = Eigen::VectorXd::Map(Y_exp_h.data(), static_cast<Eigen::Index>(Y_exp_h.size()));
        }

        // FIXME: del Y_exp si può tradurre con ->
        Y_exp = Eigen::VectorXd::Zero(Y_exp.size());
        // FIXME: del mu_exp si può tradurre con ->
        mu_exp = Eigen::VectorXd::Zero(mu_exp.size());
    }

    return sim_vector.transpose();
}

std::tuple<std::vector<double>, std::vector<double>, std::vector<double>, std::vector<double>>
ARIMA::summarize_simulations(Eigen::VectorXd mean_values, Eigen::MatrixXd sim_vector, std::vector<double> date_index,
                             long h, long past_values) {
    std::vector<double> error_bars;
    for (size_t pre{5}; pre < 100; pre += 5) {
        error_bars.push_back(mean_values[-h - 1]);
        for (Eigen::Index i{0}; i < sim_vector.rows(); i++)
            error_bars.push_back(percentile(sim_vector.row(i), pre));
    }

    std::vector<double> mv(&mean_values[0], mean_values.data());
    std::vector<double> forecasted_values;
    if (_latent_variables.get_estimation_method() == "M-H") {
        forecasted_values.push_back(mean_values[-h - 1]);
        for (Eigen::Index i{0}; i < sim_vector.rows(); i++)
            forecasted_values.push_back(mean(sim_vector.row(i)));
    } else
        std::copy(mv.end() - h - 1, mv.end(), std::back_inserter(forecasted_values));

    std::vector<double> plot_values;
    std::copy(mv.end() - h - past_values, mv.end(), std::back_inserter(plot_values));
    std::vector<double> plot_index;
    std::copy(date_index.end() - h - past_values, date_index.end(), std::back_inserter(plot_index));

    return {error_bars, forecasted_values, plot_values, plot_index};
}

void ARIMA::plot_fit(std::optional<size_t> width, std::optional<size_t> height) {
    plt::figure_size(width.value(), height.value());
    std::vector<double> date_index;
    std::copy(_index.begin() + static_cast<long>(std::max(_ar, _ma)), _index.begin() + static_cast<long>(_data_length),
              std::back_inserter(date_index));
    auto mu_Y = model(_latent_variables.get_z_values());

    // Catch specific family properties (imply different link functions/moments)
    std::vector<double> values_to_plot(mu_Y.first.size());
    if (_model_name2 == "Exponential")
        std::transform(mu_Y.first.begin(), mu_Y.first.end(), values_to_plot.begin(),
                       [this](double x) { return 1.0 / _link(x); });
    else if (_model_name2 == "Skewt") {
        Eigen::VectorXd t_params{transform_z()};
        auto scale_shape_skew{get_scale_and_shape(t_params)};
        double m1{
                (std::sqrt(std::get<1>(scale_shape_skew)) * std::tgamma((std::get<1>(scale_shape_skew) - 1.0) / 2.0)) /
                (std::sqrt(M_PI) * std::tgamma(std::get<1>(scale_shape_skew) / 2.0))};
        double additional_loc{(std::get<2>(scale_shape_skew) - (1.0 / std::get<2>(scale_shape_skew))) *
                              std::get<0>(scale_shape_skew) * m1};
        std::transform(mu_Y.first.begin(), mu_Y.first.end(), values_to_plot.begin(),
                       [additional_loc](double x) { return x + additional_loc; });
    } else
        std::transform(mu_Y.first.begin(), mu_Y.first.end(), values_to_plot.begin(),
                       [this](double x) { return _link(x); });

    std::vector<double> Y(&mu_Y.second[0], mu_Y.second.data());
    plt::named_plot("Data", date_index, Y);
    plt::named_plot("ARIMA model", date_index, values_to_plot, "k");
    plt::title(std::accumulate(_data_name.begin(), _data_name.end(), std::string{}));
    plt::legend({{"loc", "2"}});
    plt::save("../data/arima/plot_fit.png");
    // plt::show();
}

void ARIMA::plot_predict(size_t h, size_t past_values, bool intervals, std::optional<size_t> width,
                         std::optional<size_t> height) {
    assert(_latent_variables.is_estimated() && "No latent variables estimated!");

    auto mu_Y{model(_latent_variables.get_z_values())};
    std::vector<double> date_index{shift_dates(h)};
    std::vector<double> error_bars;
    std::vector<double> forecasted_values;
    std::vector<double> plot_values;
    std::vector<double> plot_index;

    if (_latent_variables.get_estimation_method() == "M-H") {
        Eigen::MatrixXd sim_vector{sim_prediction_bayes(static_cast<Eigen::Index>(h), 1500)};
        std::vector<double> Y(&mu_Y.second[0], mu_Y.second.data());

        for (size_t pre{5}; pre < 100; pre += 5) {
            error_bars.push_back(Y.back());
            for (Eigen::Index i{0}; i < sim_vector.rows(); i++)
                error_bars.push_back(percentile(sim_vector.row(i), pre));
        }

        forecasted_values.push_back(Y.back());
        for (Eigen::Index i{0}; i < sim_vector.rows(); i++)
            forecasted_values.push_back(mean(sim_vector.row(i)));

        std::copy(Y.end() - 1 - static_cast<long>(past_values), Y.end() - 2, std::back_inserter(plot_values));
        plot_values.insert(plot_values.end(), forecasted_values.begin(), forecasted_values.end());
        std::copy(date_index.end() - static_cast<long>(h + past_values), date_index.end(),
                  std::back_inserter(plot_index));
    } else {
        Eigen::VectorXd t_z{transform_z()};
        Eigen::VectorXd mean_values{mean_prediction(mu_Y.first, mu_Y.second, h, t_z)};
        Eigen::VectorXd fv = mean_values(Eigen::seq(Eigen::last - h, Eigen::last));
        forecasted_values  = std::vector<double>(&fv[0], fv.data());

        if (_model_name2 == "Skewt") {
            auto scale_shape_skew{get_scale_and_shape(t_z)};
            double m1{(std::sqrt(std::get<1>(scale_shape_skew)) *
                       std::tgamma((std::get<1>(scale_shape_skew) - 1.0) / 2.0)) /
                      (std::sqrt(M_PI) * std::tgamma(std::get<1>(scale_shape_skew) / 2.0))};
            std::transform(forecasted_values.begin(), forecasted_values.end(), forecasted_values.begin(),
                           [scale_shape_skew, m1](double x) {
                               return (std::get<2>(scale_shape_skew) - 1.0 / std::get<2>(scale_shape_skew)) *
                                      std::get<0>(scale_shape_skew) * m1;
                           });
        }

        Eigen::MatrixXd sim_values;
        if (intervals)
            sim_values = sim_prediction(mu_Y.first, mu_Y.second, h, t_z, 15000);
        else
            sim_values = sim_prediction(mu_Y.first, mu_Y.second, h, t_z, 2);

        auto results{summarize_simulations(mean_values, sim_values, date_index, static_cast<Eigen::Index>(h),
                                           static_cast<Eigen::Index>(past_values))};
        error_bars        = std::get<0>(results);
        forecasted_values = std::get<1>(results);
        plot_values       = std::get<2>(results);
        plot_index        = std::get<3>(results);
    }

    plt::figure_size(width.value(), height.value());
    if (intervals) {
        std::vector<double> alpha;
        for (size_t i{50}; i > 12; i -= 2)
            alpha.push_back(0.15 * static_cast<double>(i) / 100.0);
        for (size_t i{0}; i < error_bars.size(); i++) {
            std::vector<double> date_index_h;
            std::copy(date_index.end() - static_cast<long>(h) - 1, date_index.end(), std::back_inserter(date_index_h));
            plt::fill_between(date_index_h, std::vector<double>{error_bars[i]}, std::vector<double>{error_bars[-i - 1]},
                              {{"alpha", std::to_string(alpha[i])}});
        }

        plt::plot(plot_index, plot_values);
        plt::title("Forecast for " + std::accumulate(_data_name.begin(), _data_name.end(), std::string{}));
        plt::xlabel("Time");
        plt::ylabel(std::accumulate(_data_name.begin(), _data_name.end(), std::string{}));
        plt::save("../data/arima/plot_predict.png");
        // plt::show();
    }
}