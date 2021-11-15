#include "arima/arima.hpp"

ARIMA::ARIMA(const std::vector<double>& data, size_t ar, size_t ma, size_t integ, const Family& family) : TSM{"ARIMA"} {
    // Latent Variable information
    _ar                 = ar;
    _ma                 = ma;
    _integ              = integ;
    _z_no               = _ar + _ma + 2;
    _max_lag            = std::max(static_cast<int>(_ar), static_cast<int>(_ma));
    _z_hide             = false; // Whether to cutoff latent variables from results table
    _supported_methods  = {"MLE", "PML", "Laplace", "M-H", "BBVI"};
    _default_method     = "MLE";
    _multivariate_model = false;

    // Format the data
    SingleDataFrame checked_data = data_check(data);
    _data_frame.data             = checked_data.data;
    _data_frame.data_name        = checked_data.data_name;
    _data_frame.index            = checked_data.index;
    _data_original               = data;

    // Difference data
    for (int64_t order{0}; order < _integ; order++)
        _data_frame.data = diff(_data_frame.data);
    _data_frame.data_name = "Differenced " + _data_frame.data_name;
    _data_length          = _data_frame.data.size();

    _x = ar_matrix();
    create_latent_variables();

    _family = (family.clone());
    FamilyAttributes fa = family.setup();
    _model_name2        = fa.name;
    _link               = fa.link;
    _scale              = fa.scale;
    _shape              = fa.shape;
    _skewness           = fa.skewness;
    _mean_transform     = fa.mean_transform;
    _model_name         = _model_name2 + " ARIMA(" + std::to_string(_ar) + "," + std::to_string(_integ) + "," +
                  std::to_string(_ma) + ")";

    // Build any remaining latent variables that are specific to the family chosen
    std::vector<lv_to_build> lvs = _family->build_latent_variables();
    for (int64_t no{0}; no < lvs.size(); no++) {
        lv_to_build lv = lvs.at(no);
        _latent_variables.add_z(std::get<0>(lv), std::get<1>(lv), std::get<2>(lv));
        _latent_variables.set_z_starting_value(1 + no + _ar + _ma, std::get<3>(lv));
        delete std::get<1>(lv);
        delete std::get<2>(lv);
    }
    _latent_variables.set_z_starting_value(
            0, _mean_transform(static_cast<double>(std::reduce(_data_frame.data.begin(), _data_frame.data.end())) /
                               static_cast<double>(_data_frame.data.size())));

    _family_z_no = lvs.size();
    _z_no        = _latent_variables.get_z_list().size();

    // If Normal family is selected, we use faster likelihood functions
    if (isinstance<Normal>(_family.get())) {
        _model         = {[this](const Eigen::VectorXd& x) { return normal_model(x); }};
        _mb_model      = {[this](const Eigen::VectorXd& x, size_t mb) { return mb_normal_model(x, mb); }};
        _neg_loglik    = {[this](const Eigen::VectorXd& x) { return normal_neg_loglik(x); }};
        _mb_neg_loglik = {[this](const Eigen::VectorXd& x, size_t mb) { return normal_mb_neg_loglik(x, mb); }};
    }
    // TODO: else if (...) with missing models
    else {
        _model         = {[this](const Eigen::VectorXd& x) { return non_normal_model(x); }};
        _mb_model      = {[this](const Eigen::VectorXd& x, size_t mb) { return mb_non_normal_model(x, mb); }};
        _neg_loglik    = {[this](const Eigen::VectorXd& x) { return non_normal_neg_loglik(x); }};
        _mb_neg_loglik = {[this](const Eigen::VectorXd& x, size_t mb) { return non_normal_mb_neg_loglik(x, mb); }};
    }
}

ARIMA::ARIMA(const DataFrame& data_frame, size_t ar, size_t ma, size_t integ,
             const Family& family, const std::string& target)
    : TSM{"ARIMA"} {
    // Latent Variable information
    _ar                 = ar;
    _ma                 = ma;
    _integ              = integ;
    _z_no               = _ar + _ma + 2;
    _max_lag            = std::max(static_cast<int>(_ar), static_cast<int>(_ma));
    _z_hide             = false;
    _supported_methods  = {"MLE", "PML", "Laplace", "M-H", "BBVI"};
    _default_method     = "MLE";
    _multivariate_model = false;

    // Format the data
    SingleDataFrame checked_data = data_check(data_frame, target);
    _data_frame.data             = checked_data.data;
    _data_frame.data_name        = checked_data.data_name;
    _data_frame.index            = checked_data.index;
    _data_original               = _data_frame.data;

    // Difference data
    for (int64_t order{0}; order < _integ; order++)
        _data_frame.data = diff(_data_frame.data);
    _data_frame.data_name = "Differenced " + _data_frame.data_name;
    _data_length          = _data_frame.data.size();

    _x = ar_matrix();
    create_latent_variables();

    _family = (family.clone());
    FamilyAttributes fa = family.setup();
    _model_name2        = fa.name;
    _link               = fa.link;
    _scale              = fa.scale;
    _shape              = fa.shape;
    _skewness           = fa.skewness;
    _mean_transform     = fa.mean_transform;
    _model_name         = _model_name2 + " ARIMA(" + std::to_string(_ar) + "," + std::to_string(_integ) + "," +
                  std::to_string(_ma) + ")";

    // Build any remaining latent variables that are specific to the family chosen
    std::vector<lv_to_build> lvs = _family->build_latent_variables();
    for (int64_t no{0}; no < lvs.size(); no++) {
        lv_to_build lv = lvs[no];
        _latent_variables.add_z(std::get<0>(lv), std::get<1>(lv), std::get<2>(lv));
        _latent_variables.set_z_starting_value(1 + no + _ar + _ma, std::get<3>(lv));
        delete std::get<1>(lv);
        delete std::get<2>(lv);
    }
    _latent_variables.set_z_starting_value(
            0, _mean_transform(static_cast<double>(std::reduce(_data_frame.data.begin(), _data_frame.data.end())) /
                               static_cast<double>(_data_frame.data.size())));

    _family_z_no = lvs.size();
    _z_no        = _latent_variables.get_z_list().size();

    // If Normal family is selected, we use faster likelihood functions
    if (isinstance<Normal>(_family.get())) {
        _model         = {[this](const Eigen::VectorXd& x) { return normal_model(x); }};
        _mb_model      = {[this](const Eigen::VectorXd& x, size_t mb) { return mb_normal_model(x, mb); }};
        _neg_loglik    = {[this](const Eigen::VectorXd& x) { return normal_neg_loglik(x); }};
        _mb_neg_loglik = {[this](const Eigen::VectorXd& x, size_t mb) { return normal_mb_neg_loglik(x, mb); }};
    }
    // TODO: else if (...) with missing models
    else {
        _model         = {[this](const Eigen::VectorXd& x) { return non_normal_model(x); }};
        _mb_model      = {[this](const Eigen::VectorXd& x, size_t mb) { return mb_non_normal_model(x, mb); }};
        _neg_loglik    = {[this](const Eigen::VectorXd& x) { return non_normal_neg_loglik(x); }};
        _mb_neg_loglik = {[this](const Eigen::VectorXd& x, size_t mb) { return non_normal_mb_neg_loglik(x, mb); }};
    }
}

Eigen::MatrixXd ARIMA::ar_matrix() {
    Eigen::MatrixXd X{Eigen::MatrixXd::Ones(static_cast<Eigen::Index>(_ar + 1),
                                            static_cast<Eigen::Index>(_data_length - _max_lag))};

    if (_ar != 0) {
        for (Eigen::Index i{0}; i < _ar; i++)
            std::copy(_data_frame.data.begin() + _max_lag - i - 1, _data_frame.data.end() - i - 1,
                      X.row(i + 1).begin());
    }

    return X;
}

ModelOutput ARIMA::categorize_model_output(const Eigen::VectorXd& z) const {
    auto mu_Y{_model(z)};
    return {mu_Y.first, mu_Y.second};
}

void ARIMA::create_latent_variables() {
    Normal n1{Normal(0, 3)};
    _latent_variables.add_z("Constant", reinterpret_cast<Family*>(&n1), reinterpret_cast<Family*>(&n1));

    n1 = Normal(0, 0.5);
    Normal n2{Normal(0.3)};
    for (int64_t ar_terms{0}; ar_terms < _ar; ar_terms++)
        _latent_variables.add_z("AR(" + std::to_string(ar_terms + 1) + ")", reinterpret_cast<Family*>(&n1),
                                reinterpret_cast<Family*>(&n2));

    for (int64_t ma_terms{0}; ma_terms < _ma; ma_terms++)
        _latent_variables.add_z("MA(" + std::to_string(ma_terms + 1) + ")", reinterpret_cast<Family*>(&n1),
                                reinterpret_cast<Family*>(&n2));
}

std::tuple<double, double, double> ARIMA::get_scale_and_shape(const Eigen::VectorXd& transformed_lvs) const {
    double model_shape, model_scale, model_skewness{0};

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
    Eigen::VectorXd model_shape, model_scale, model_skewness{Eigen::VectorXd::Zero(transformed_lvs.cols())};

    if (_scale) {
        if (_shape) {
            // Apply trasform() to every element inside the matrix last row
            model_shape = transformed_lvs(Eigen::last, Eigen::all);
            std::transform(model_shape.begin(), model_shape.end(), model_shape.begin(), [this](double n) {
                return _latent_variables.get_z_list().back().get_prior()->get_transform()(n);
            });
        }

        model_scale = transformed_lvs(Eigen::last - 1, Eigen::all);
        std::transform(model_scale.begin(), model_scale.end(), model_scale.begin(), [this](double n) {
            return _latent_variables.get_z_list().at(-2).get_prior()->get_transform()(n);
        });
    }

    if (_skewness) {
        model_skewness = transformed_lvs(-2, Eigen::all);
        std::transform(model_skewness.begin(), model_skewness.end(), model_skewness.begin(), [this](double n) {
            return _latent_variables.get_z_list().at(-3).get_prior()->get_transform()(n);
        });
    }

    return std::move(std::make_tuple(model_scale, model_shape, model_skewness));
}

std::pair<Eigen::VectorXd, Eigen::VectorXd> ARIMA::normal_model(const Eigen::VectorXd& beta) const {
    Eigen::VectorXd Y(_data_frame.data.size() - _max_lag);
    std::copy(_data_frame.data.begin() + _max_lag, _data_frame.data.end(), Y.begin());

    // Transform latent variables
    Eigen::VectorXd z(beta.size());
    for (Eigen::Index i{0}; i < beta.size(); i++) {
        z[i] = _latent_variables.get_z_list().at(i).get_prior()->get_transform()(beta[i]);
    }

    // Constant and AR terms
    Eigen::VectorXd mu;
    if (_ar != 0)
        mu = _x.transpose() * z(Eigen::seq(0, Eigen::last - static_cast<Eigen::Index>(_family_z_no + _ma)));
    else
        mu = Eigen::VectorXd::Zero(Y.size()) * z[0];

    // MA terms
    if (_ma != 0)
        mu = arima_recursion_normal(z, mu, Y, _max_lag, Y.size(), _ar, _ma);

    return {mu, Y};
}

std::pair<Eigen::VectorXd, Eigen::VectorXd> ARIMA::non_normal_model(const Eigen::VectorXd& beta) const {
    std::vector<double> data;
    std::copy(_data_frame.data.begin() + _max_lag, _data_frame.data.end(), std::back_inserter(data));
    Eigen::VectorXd Y{Eigen::VectorXd::Map(data.data(), static_cast<Eigen::Index>(data.size()))};

    // Transform latent variables
    Eigen::VectorXd z(beta.size());
    for (Eigen::Index i{0}; i < beta.size(); i++) {
        z[i] = _latent_variables.get_z_list().at(i).get_prior()->get_transform()(beta[i]);
    }

    // Constant and AR terms
    Eigen::VectorXd mu;
    if (_ar != 0)
        mu = _x.transpose() * z(Eigen::seq(0, Eigen::last - static_cast<Eigen::Index>(_family_z_no + _ma)));
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
    std::random_device r; // Seed with a real random value, if available
    std::default_random_engine e(r());
    std::uniform_int_distribution<size_t> uniform_dist(0, _data_frame.data.size() - mini_batch - _max_lag + 1);
    size_t rand_int = uniform_dist(e);
    std::vector<double> sample(mini_batch);
    std::iota(sample.begin(), sample.end(), rand_int);

    std::vector<double> data;
    std::copy(_data_frame.data.begin() + _max_lag, _data_frame.data.end(), std::back_inserter(data));
    Eigen::VectorXd Y{Eigen::VectorXd::Map(data.data(), static_cast<Eigen::Index>(data.size()))};
    Y = Y(Eigen::all, sample);
    Eigen::MatrixXd X(_x.rows(), sample.size());
    for (Eigen::Index i{0}; i < _x.rows(); i++)
        X.row(i) = _x.row(i)(Eigen::all, sample);

    // Transform latent variables
    Eigen::VectorXd z(beta.size());
    for (Eigen::Index i{0}; i < beta.size(); i++) {
        z[i] = _latent_variables.get_z_list().at(i).get_prior()->get_transform()(beta[i]);
    }

    // Constant and AR terms
    Eigen::VectorXd mu;
    if (_ar != 0)
        mu = _x.transpose() * z(Eigen::seq(0, Eigen::last - static_cast<Eigen::Index>(_family_z_no + _ma)));
    else
        mu = Eigen::VectorXd::Zero(Y.size()) * z[0];

    // MA terms
    if (_ma != 0)
        mu = arima_recursion_normal(z, mu, Y, _max_lag, Y.size(), _ar, _ma);

    return {mu, Y};
}

std::pair<Eigen::VectorXd, Eigen::VectorXd> ARIMA::mb_non_normal_model(const Eigen::VectorXd& beta,
                                                                       size_t mini_batch) const {
    std::random_device r; // Seed with a real random value, if available
    std::default_random_engine e(r());
    std::uniform_int_distribution<size_t> uniform_dist(0, _data_frame.data.size() - mini_batch - _max_lag + 1);
    size_t rand_int = uniform_dist(e);
    std::vector<double> sample(mini_batch);
    std::iota(sample.begin(), sample.end(), rand_int);

    std::vector<double> data;
    std::copy(_data_frame.data.begin() + _max_lag, _data_frame.data.end(), std::back_inserter(data));
    Eigen::VectorXd Y{Eigen::VectorXd::Map(data.data(), static_cast<Eigen::Index>(data.size()))};
    Y = Y(Eigen::all, sample);
    Eigen::MatrixXd X(_x.rows(), sample.size());
    for (Eigen::Index i{0}; i < _x.rows(); i++)
        X.row(i) = _x.row(i)(Eigen::all, sample);

    // Transform latent variables
    Eigen::VectorXd z(beta.size());
    for (Eigen::Index i{0}; i < beta.size(); i++) {
        z[i] = _latent_variables.get_z_list().at(i).get_prior()->get_transform()(beta[i]);
    }

    // Constant and AR terms
    Eigen::VectorXd mu;
    if (_ar != 0)
        mu = _x.transpose() * z(Eigen::seq(0, Eigen::last - static_cast<Eigen::Index>(_family_z_no + _ma)));
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

double ARIMA::normal_neg_loglik(const Eigen::VectorXd& beta) const {
    std::pair<Eigen::VectorXd, Eigen::VectorXd> mu_y = _model(beta);
    Eigen::VectorXd scale{{_latent_variables.get_z_priors().back()->get_transform()(beta(Eigen::last))}};
    return -Mvn::logpdf(mu_y.second, mu_y.first, scale).sum();
}

double ARIMA::normal_mb_neg_loglik(const Eigen::VectorXd& beta, size_t mini_batch) const {
    std::pair<Eigen::VectorXd, Eigen::VectorXd> mu_Y = _mb_model(beta, mini_batch);
    Eigen::VectorXd scale{{_latent_variables.get_z_priors().back()->get_transform()(beta(Eigen::last))}};
    return -Mvn::logpdf(mu_Y.second, mu_Y.first, scale).sum();
}

double ARIMA::non_normal_neg_loglik(const Eigen::VectorXd& beta) const {
    std::pair<Eigen::VectorXd, Eigen::VectorXd> mu_Y = _model(beta);
    Eigen::VectorXd transformed_parameters(beta.size());
    for (Eigen::Index k{0}; k < beta.size(); k++)
        transformed_parameters(k) = _latent_variables.get_z_priors().at(k)->get_transform()(beta(k));
    auto scale_shape_skew = get_scale_and_shape(transformed_parameters);
    Eigen::VectorXd link_mu(mu_Y.first.size());
    std::transform(mu_Y.first.begin(), mu_Y.first.end(), link_mu.begin(), _link);
    return _family->neg_loglikelihood(mu_Y.second, link_mu, std::get<0>(scale_shape_skew));
}

double ARIMA::non_normal_mb_neg_loglik(const Eigen::VectorXd& beta, size_t mini_batch) const {
    std::pair<Eigen::VectorXd, Eigen::VectorXd> mu_Y = _mb_model(beta, mini_batch);
    Eigen::VectorXd transformed_parameters(beta.size());
    for (Eigen::Index k{0}; k < beta.size(); k++)
        transformed_parameters(k) = _latent_variables.get_z_priors().at(k)->get_transform()(beta(k));
    auto scale_shape_skew = get_scale_and_shape(transformed_parameters);
    Eigen::VectorXd link_mu(mu_Y.first.size());
    std::transform(mu_Y.first.begin(), mu_Y.first.end(), link_mu.begin(), _link);
    return _family->neg_loglikelihood(mu_Y.second, link_mu, std::get<0>(scale_shape_skew));
}

Eigen::VectorXd ARIMA::mean_prediction(const Eigen::VectorXd& mu, const Eigen::VectorXd& Y, size_t h,
                                       Eigen::VectorXd t_z) const {
    // Create arrays to iterate over
    Eigen::VectorXd Y_exp{Y};
    Eigen::VectorXd mu_exp{mu};

    // Loop over h time periods
    for (int64_t t{0}; t < h; t++) {
        double new_value = t_z[0];

        if (_ar != 0) {
            for (Eigen::Index i{1}; i <= _ar; i++)
                new_value += t_z[i] * Y_exp(Eigen::last - i);
        }

        if (_ma != 0) {
            for (Eigen::Index i{1}; i <= _ma; i++) {
                if (i - 1 >= t)
                    new_value += t_z[i + static_cast<Eigen::Index>(_ar)] *
                                 (Y_exp(Eigen::last - i) - _link(mu_exp(Eigen::last - i)));
            }
        }

        std::vector<double> Y_exp_v(&Y_exp[0], Y_exp.data() + Y_exp.size());
        if (_model_name2 == "Exponential")
            Y_exp_v.push_back(1.0 / _link(new_value));
        else
            Y_exp_v.push_back(_link(new_value));
        Y_exp = Eigen::VectorXd::Map(Y_exp_v.data(), static_cast<Eigen::Index>(Y_exp_v.size()));

        // For indexing consistency
        std::vector<double> mu_exp_v(&mu_exp[0], mu_exp.data() + mu_exp.size());
        mu_exp_v.push_back(0.0);
        mu_exp = Eigen::VectorXd::Map(mu_exp_v.data(), static_cast<Eigen::Index>(mu_exp_v.size()));
    }

    // FIXME: del mu_exp si può tradurre con ->
    mu_exp = Eigen::VectorXd::Zero(mu_exp.size());

    return Y_exp;
}

Eigen::MatrixXd ARIMA::sim_prediction(const Eigen::VectorXd& mu, const Eigen::VectorXd& Y, size_t h,
                                      const Eigen::VectorXd& t_params, size_t simulations) const {
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
                    new_value += t_params[i] * Y_exp(Eigen::last - i);
            }

            if (_ma != 0) {
                for (Eigen::Index i{1}; i <= _ma; i++) {
                    if (i - 1 >= t)
                        new_value += t_params[i + static_cast<Eigen::Index>(_ar)] *
                                     (Y_exp(Eigen::last - i) - mu_exp(Eigen::last - i));
                }
            }

            std::vector<double> Y_exp_v(&Y_exp[0], Y_exp.data() + Y_exp.size());
            if (_model_name2 == "Exponential")
                Y_exp_v.push_back(_family->draw_variable(1.0 / _link(new_value), std::get<0>(scale_shape_skew), 1)[0]);
            else
                Y_exp_v.push_back(_family->draw_variable(_link(new_value), std::get<0>(scale_shape_skew), 1)[0]);

            std::vector<double> mu_exp_v(&mu_exp[0], mu_exp.data() + mu_exp.size());
            mu_exp_v.push_back(0.0);
            mu_exp = Eigen::VectorXd::Map(mu_exp_v.data(), static_cast<Eigen::Index>(mu_exp_v.size()));

            // For indexing consistency
            std::vector<double> Y_exp_h;
            std::copy(Y_exp_v.end() - static_cast<long>(h), Y_exp_v.end(), std::back_inserter(Y_exp_h));
            sim_vector.row(n) = Eigen::VectorXd::Map(Y_exp_h.data(), static_cast<Eigen::Index>(Y_exp_h.size()));
        }

        // FIXME: del Y_exp si può tradurre con ->
        Y_exp = Eigen::VectorXd::Zero(Y_exp.size());
        // FIXME: del mu_exp si può tradurre con ->
        mu_exp = Eigen::VectorXd::Zero(mu_exp.size());
    }

    return sim_vector.transpose();
}

Eigen::MatrixXd ARIMA::sim_prediction_bayes(size_t h, size_t simulations) const {
    Eigen::MatrixXd sim_vector{
            Eigen::MatrixXd::Zero(static_cast<Eigen::Index>(simulations), static_cast<Eigen::Index>(h))};

    for (Eigen::Index n{0}; n < simulations; n++) {
        Eigen::VectorXd t_z{draw_latent_variables(1).transpose().row(0)};
        auto mu_Y{_model(t_z)};
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
                    new_value += t_z[i] * Y_exp(Eigen::last - i);
            }

            if (_ma != 0) {
                for (Eigen::Index i{1}; i <= _ma; i++) {
                    if (i - 1 >= t)
                        new_value += t_z[i + static_cast<Eigen::Index>(_ar)] *
                                     (Y_exp(Eigen::last - i) - mu_exp(Eigen::last - i));
                }
            }

            std::vector<double> Y_exp_v(&Y_exp[0], Y_exp.data() + Y_exp.size());
            if (_model_name2 == "Exponential")
                Y_exp_v.push_back(_family->draw_variable(1.0 / _link(new_value), std::get<0>(scale_shape_skew), 1)[0]);
            else
                Y_exp_v.push_back(_family->draw_variable(_link(new_value), std::get<0>(scale_shape_skew), 1)[0]);

            std::vector<double> mu_exp_v(&mu_exp[0], mu_exp.data() + mu_exp.size());
            mu_exp_v.push_back(0.0);
            mu_exp = Eigen::VectorXd::Map(mu_exp_v.data(), static_cast<Eigen::Index>(mu_exp_v.size()));

            // For indexing consistency
            std::vector<double> Y_exp_h;
            std::copy(Y_exp_v.end() - static_cast<long>(h), Y_exp_v.end(), std::back_inserter(Y_exp_h));
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
ARIMA::summarize_simulations(const Eigen::VectorXd& mean_values, const Eigen::MatrixXd& sim_vector,
                             const std::vector<double>& date_index, size_t h, size_t past_values) const {
    std::vector<double> error_bars;
    for (int64_t pre{5}; pre < 100; pre += 5) {
        error_bars.push_back(mean_values[static_cast<Eigen::Index>(-h - 1)]);
        for (Eigen::Index i{0}; i < sim_vector.rows(); i++)
            error_bars.push_back(percentile(sim_vector.row(i), pre));
    }

    std::vector<double> mv(&mean_values[0], mean_values.data() + mean_values.size());
    std::vector<double> forecasted_values;
    if (_latent_variables.get_estimation_method() == "M-H") {
        forecasted_values.push_back(mean_values[static_cast<Eigen::Index>(-h - 1)]);
        for (Eigen::Index i{0}; i < sim_vector.rows(); i++)
            forecasted_values.push_back(mean(sim_vector.row(i)));
    } else
        std::copy(mv.end() - static_cast<long>(h + 1), mv.end(), std::back_inserter(forecasted_values));

    std::vector<double> plot_values;
    std::copy(mv.end() - static_cast<long>(h + past_values), mv.end(), std::back_inserter(plot_values));
    std::vector<double> plot_index;
    std::copy(date_index.end() - static_cast<long>(h + past_values), date_index.end(), std::back_inserter(plot_index));

    return {std::move(error_bars), std::move(forecasted_values), std::move(plot_values), std::move(plot_index)};
}

void ARIMA::plot_fit(std::optional<size_t> width, std::optional<size_t> height) const {
    plt::figure_size(width.value(), height.value());
    std::vector<double> date_index;
    std::copy(_data_frame.index.begin() + static_cast<long>(std::max(_ar, _ma)),
              _data_frame.index.begin() + static_cast<long>(_data_length), std::back_inserter(date_index));
    auto mu_Y = _model(_latent_variables.get_z_values());

    // Catch specific family properties (imply different link functions/moments)
    std::vector<double> values_to_plot(mu_Y.first.size());
    if (_model_name2 == "Exponential")
        std::transform(mu_Y.first.begin(), mu_Y.first.end(), values_to_plot.begin(),
                       [this](double x) { return 1.0 / _link(x); });
    else if (_model_name2 == "Skewt") {
        Eigen::VectorXd t_params{transform_z()};
        auto scale_shape_skew{get_scale_and_shape(t_params)};
        double m1{
                (std::sqrt(std::get<1>(scale_shape_skew)) * std::tgamma((std::get<1>(scale_shape_skew) - 1.0) * 0.5)) /
                (std::sqrt(M_PI) * std::tgamma(std::get<1>(scale_shape_skew) * 0.5))};
        double additional_loc{(std::get<2>(scale_shape_skew) - (1.0 / std::get<2>(scale_shape_skew))) *
                              std::get<0>(scale_shape_skew) * m1};
        std::transform(mu_Y.first.begin(), mu_Y.first.end(), values_to_plot.begin(),
                       [additional_loc](double x) { return x + additional_loc; });
    } else
        std::transform(mu_Y.first.begin(), mu_Y.first.end(), values_to_plot.begin(),
                       [this](double x) { return _link(x); });

    std::vector<double> Y(&mu_Y.second[0], mu_Y.second.data() + mu_Y.second.size());
    plt::named_plot("Data", date_index, Y);
    plt::named_plot("ARIMA model", date_index, values_to_plot, "k");
    plt::title(std::accumulate(_data_frame.data_name.begin(), _data_frame.data_name.end(), std::string{}));
    plt::legend({{"loc", "2"}});
    plt::save("../data/arima/plot_fit.png");
    // plt::show();
}

void ARIMA::plot_predict(size_t h, size_t past_values, bool intervals, std::optional<size_t> width,
                         std::optional<size_t> height) const {
    assert(_latent_variables.is_estimated() && "No latent variables estimated!");

    auto mu_Y{_model(_latent_variables.get_z_values())};
    std::vector<double> date_index{shift_dates(h)};
    std::vector<double> error_bars;
    std::vector<double> forecasted_values;
    std::vector<double> plot_values;
    std::vector<double> plot_index;

    if (_latent_variables.get_estimation_method() == "M-H") {
        Eigen::MatrixXd sim_vector{sim_prediction_bayes(static_cast<Eigen::Index>(h), 1500)};
        std::vector<double> Y(&mu_Y.second[0], mu_Y.second.data() + mu_Y.second.size());

        for (int64_t pre{5}; pre < 100; pre += 5) {
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
        Eigen::VectorXd fv = mean_values(Eigen::seq(Eigen::last - static_cast<Eigen::Index>(h), Eigen::last));
        std::copy(mean_values.end() - static_cast<long>(h), mean_values.end(), std::back_inserter(forecasted_values));

        if (_model_name2 == "Skewt") {
            auto scale_shape_skew{get_scale_and_shape(t_z)};
            double m1{(std::sqrt(std::get<1>(scale_shape_skew)) *
                       std::tgamma((std::get<1>(scale_shape_skew) - 1.0) * 0.5)) /
                      (std::sqrt(M_PI) * std::tgamma(std::get<1>(scale_shape_skew) * 0.5))};
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
        for (int64_t i{50}; i > 12; i -= 2)
            alpha.push_back(0.15 * static_cast<double>(i) * 0.01);
        for (int64_t i{0}; i < error_bars.size(); i++) {
            std::vector<double> date_index_h;
            std::copy(date_index.end() - static_cast<long>(h) - 1, date_index.end(), std::back_inserter(date_index_h));
            plt::fill_between(date_index_h, std::vector<double>{error_bars[i]}, std::vector<double>{error_bars[-i - 1]},
                              {{"alpha", std::to_string(alpha[i])}});
        }

        plt::plot(plot_index, plot_values);
        plt::title("Forecast for " +
                   std::accumulate(_data_frame.data_name.begin(), _data_frame.data_name.end(), std::string{}));
        plt::xlabel("Time");
        plt::ylabel(std::accumulate(_data_frame.data_name.begin(), _data_frame.data_name.end(), std::string{}));
        plt::save("../data/arima/plot_predict.png");
        // plt::show();
    }
}


DataFrame ARIMA::predict_is(size_t h, bool fit_once, const std::string& fit_method, bool intervals) const {
    DataFrame predictions;
    LatentVariables saved_lvs{""};

    std::vector<std::string> names{std::accumulate(_data_frame.data_name.begin(), _data_frame.data_name.end(), std::string{}),
                        "1% Prediction Interval", "5% Prediction Interval", "95% Prediction Interval",
                        "99% Prediction Interval"};

    std::vector<double> data_original_t, index;
    DataFrame new_prediction;
    for (Eigen::Index t{0}; t < h; t++) {
        std::copy(_data_original.begin(), _data_original.end() - static_cast<long>(h - t),
                  std::back_inserter(data_original_t));
        std::iota(index.begin(), index.end(), 0);
        ARIMA x{data_original_t, _ar, _ma, _integ, *_family->clone()};
        if (!fit_once) {
            Results* temp_r = x.fit(fit_method);
            delete temp_r;
        }
        if (t == 0) {
            if (fit_once) {
                Results* temp_r = x.fit(fit_method);
                saved_lvs = x._latent_variables;
                delete temp_r;
            }
        } else {
            if (fit_once)
                x._latent_variables = saved_lvs;
        }
        new_prediction = x.predict(1, intervals);
        for (int64_t i{0}; i < new_prediction.data_name.size(); i++) {
            if (predictions.data.size() == i)
                predictions.data.push_back(new_prediction.data[i]);
            else
                predictions.data[i].insert(predictions.data[i].end(), new_prediction.data[i].begin(),
                                           new_prediction.data[i].end());
        }
    }

    std::copy(_data_frame.index.end() - static_cast<long>(h), _data_frame.index.end(),
              std::back_inserter(predictions.index));

    return predictions;
}

void ARIMA::plot_predict_is(size_t h, bool fit_once, const std::string& fit_method, std::optional<size_t> width,
                            std::optional<size_t> height) const {
    plt::figure_size(width.value(), height.value());
    auto predictions{predict_is(h, fit_once, fit_method)};
    std::vector<double> data;
    std::copy(_data_frame.data.end() - static_cast<long>(h), _data_frame.data.end(), std::back_inserter(data));
    plt::named_plot("Data", predictions.index, data);
    for (int64_t i{0}; i < predictions.data_name.size(); i++)
        plt::named_plot("Predictions", predictions.index, predictions.data[i], "k");
    plt::title(std::accumulate(_data_frame.data_name.begin(), _data_frame.data_name.end(), std::string{}));
    plt::legend({{"loc", "2"}});
    plt::save("../data/arima/plot_predict_is.png");
    // plt::show();
}

DataFrame ARIMA::predict(size_t h, bool intervals) const {
    assert(_latent_variables.is_estimated() && "No latent variables estimated!");

    auto mu_Y{_model(_latent_variables.get_z_values())};
    std::vector<double> date_index{shift_dates(h)};

    Eigen::MatrixXd sim_values;
    DataFrame result;
    std::vector<double> forecasted_values, prediction_01, prediction_05, prediction_95, prediction_99;
    if (_latent_variables.get_estimation_method() == "M-H") {
        sim_values = sim_prediction_bayes(h, 15000);
        for (Eigen::Index i{0}; i < sim_values.rows(); i++) {
            forecasted_values.push_back(mean(sim_values.row(i)));
            prediction_01.push_back(percentile(sim_values.row(i), 1));
            prediction_05.push_back(percentile(sim_values.row(i), 5));
            prediction_95.push_back(percentile(sim_values.row(i), 95));
            prediction_99.push_back(percentile(sim_values.row(i), 99));
        }
    } else {
        Eigen::VectorXd t_z{transform_z()};
        t_z = t_z.unaryExpr([](double v) { return std::isnan(v) ? 0.0 : v; });
        Eigen::VectorXd mean_values{mean_prediction(mu_Y.first, mu_Y.second, h, t_z)};
        if (intervals)
            sim_values = sim_prediction(mu_Y.first, mu_Y.second, h, t_z, 15000);
        else
            sim_values = sim_prediction(mu_Y.first, mu_Y.second, h, t_z, 2);

        forecasted_values = std::vector<double>(h);
        if (_model_name2 == "Skewt") {
            auto scale_shape_skew{get_scale_and_shape(t_z)};
            double m1{(std::sqrt(std::get<1>(scale_shape_skew)) *
                       std::tgamma((std::get<1>(scale_shape_skew) - 1.0) * 0.5)) /
                      (std::sqrt(M_PI) * std::tgamma(std::get<1>(scale_shape_skew) * 0.5))};
            double additional_loc{(std::get<2>(scale_shape_skew) - (1.0 / std::get<2>(scale_shape_skew))) *
                                  std::get<0>(scale_shape_skew) * m1};
            std::transform(mean_values.end() - static_cast<long>(h), mean_values.end(), forecasted_values.begin(),
                           [additional_loc](double x) { return x + additional_loc; });
        } else
            std::copy(mean_values.end() - static_cast<long>(h), mean_values.end(), forecasted_values.begin());
    }
    if (!intervals) {
        result.data.emplace_back(forecasted_values);
        result.data_name.push_back(
                std::accumulate(_data_frame.data_name.begin(), _data_frame.data_name.end(), std::string{}));
    } else {
        if (_latent_variables.get_estimation_method() != "M-H") {
            // sim_values = sim_prediction(mu_Y.first, mu_Y.second, 5, t_z, 15000);
            for (Eigen::Index i{0}; i < sim_values.rows(); i++) {
                prediction_01.push_back(percentile(sim_values.row(i), 1));
                prediction_05.push_back(percentile(sim_values.row(i), 5));
                prediction_95.push_back(percentile(sim_values.row(i), 95));
                prediction_99.push_back(percentile(sim_values.row(i), 99));
            }
        }
        result.data_name.push_back(
                std::accumulate(_data_frame.data_name.begin(), _data_frame.data_name.end(), std::string{}));
        result.data.push_back(forecasted_values);
        result.data_name.emplace_back("1% Prediction Interval");
        result.data.push_back(prediction_01);
        result.data_name.emplace_back("5% Prediction Interval");
        result.data.push_back(prediction_05);
        result.data_name.emplace_back("95% Prediction Interval");
        result.data.push_back(prediction_95);
        result.data_name.emplace_back("99% Prediction Interval");
        result.data.push_back(prediction_99);
    }
    std::copy(date_index.end() - static_cast<long>(h), date_index.end(), std::back_inserter(result.index));

    return result;
}

Eigen::MatrixXd ARIMA::sample(size_t nsims) const {
    assert((_latent_variables.get_estimation_method() == "BBVI" ||
            _latent_variables.get_estimation_method() == "M-H") &&
           "No latent variables estimated!");
    Eigen::MatrixXd lv_draws{draw_latent_variables(nsims)};
    std::vector<Eigen::VectorXd> mus;
    for (Eigen::Index i{0}; i < nsims; i++)
        mus.push_back(_model(lv_draws.col(i)).first);

    Eigen::VectorXd temp_mus(mus.at(0).size());
    Eigen::MatrixXd data_draws(nsims, mus.at(0).size());
    for (Eigen::Index i{0}; i < nsims; i++) {
        auto scale_shape_skew{get_scale_and_shape(lv_draws.col(i))};
        std::transform(mus[i].begin(), mus[i].end(), temp_mus.begin(), _link);
        data_draws.row(i) =
                _family->draw_variable(temp_mus, std::get<0>(scale_shape_skew), static_cast<int>(mus.at(i).size()));
    }

    return std::move(data_draws);
}

void ARIMA::plot_sample(size_t nsims, bool plot_data, std::optional<size_t> width, std::optional<size_t> height) const {
    assert((_latent_variables.get_estimation_method() == "BBVI" ||
            _latent_variables.get_estimation_method() == "M-H") &&
           "No latent variables estimated!");

    plt::figure_size(width.value(), height.value());
    std::vector<double> date_index;
    std::copy(_data_frame.index.begin() + static_cast<long>(std::max(_ar, _ma)),
              _data_frame.index.begin() + static_cast<long>(_data_length), std::back_inserter(date_index));
    auto mu_Y = _model(_latent_variables.get_z_values());
    Eigen::MatrixXd draws{sample(nsims).transpose()};
    for (Eigen::Index i{0}; i < draws.rows(); i++)
        plt::named_plot(
                "Posterior Draws", date_index,
                std::vector<double>(&draws.row(i)[0],
                                    draws.row(i).data() +
                                            draws.row(i).size())); // FIXME: alpha = 1.0 parameter only in hist method
    if (plot_data)
        plt::named_plot("Data", date_index,
                        std::vector<double>(&mu_Y.second[0], mu_Y.second.data() + mu_Y.second.size()),
                        "sk"); // FIXME: alpha = 0.5 parameter only in hist method
    plt::title(std::accumulate(_data_frame.data_name.begin(), _data_frame.data_name.end(), std::string{}));
    plt::save("../data/arima/plot_sample.png");
    // plt::show();
}

double ARIMA::ppc(size_t nsims, const std::function<double(Eigen::VectorXd)>& T) const {
    assert((_latent_variables.get_estimation_method() == "BBVI" ||
            _latent_variables.get_estimation_method() == "M-H") &&
           "No latent variables estimated!");

    Eigen::MatrixXd lv_draws{draw_latent_variables(nsims)};
    std::vector<Eigen::VectorXd> mus;
    for (Eigen::Index i{0}; i < nsims; i++)
        mus.push_back(_model(lv_draws.col(i)).first);

    Eigen::VectorXd temp_mus(mus.at(0).size());
    Eigen::MatrixXd data_draws(nsims, mus.at(0).size());
    for (Eigen::Index i{0}; i < nsims; i++) {
        auto scale_shape_skew{get_scale_and_shape(lv_draws.col(i))};
        std::transform(mus[i].begin(), mus[i].end(), temp_mus.begin(), _link);
        data_draws.row(i) =
                _family->draw_variable(temp_mus, std::get<0>(scale_shape_skew), static_cast<int>(mus.at(i).size()));
    }

    Eigen::Matrix sample_data{sample(nsims)};
    std::vector<double> T_sims;
    for (Eigen::Index i{0}; i < sample_data.cols(); i++)
        T_sims.push_back(T(sample_data.col(i)));
    double T_actual{T(Eigen::VectorXd::Map(_data_frame.data.data(), _data_frame.data.size()))};

    std::vector<double> T_sims_greater;
    for (int64_t i{0}; i < T_sims.size(); i++) {
        if (T_sims.at(i) > T_actual)
            T_sims_greater.push_back(T_sims.at(i));
    }

    return static_cast<double>(T_sims_greater.size()) / static_cast<double>(nsims);
}

void ARIMA::plot_ppc(size_t nsims, const std::function<double(Eigen::VectorXd)>& T, const std::string& T_name,
                     std::optional<size_t> width, std::optional<size_t> height) const {
    assert((_latent_variables.get_estimation_method() == "BBVI" ||
            _latent_variables.get_estimation_method() == "M-H") &&
           "No latent variables estimated!");

    Eigen::MatrixXd lv_draws{draw_latent_variables(nsims)};
    std::vector<Eigen::VectorXd> mus;
    for (Eigen::Index i{0}; i < nsims; i++)
        mus.push_back(_model(lv_draws.col(i)).first);

    Eigen::VectorXd temp_mus(mus.at(0).size());
    Eigen::MatrixXd data_draws(nsims, mus.at(0).size());
    for (Eigen::Index i{0}; i < nsims; i++) {
        auto scale_shape_skew{get_scale_and_shape(lv_draws.row(i))};
        std::transform(mus[i].begin(), mus[i].end(), temp_mus.begin(), _link);
        data_draws.row(i) =
                _family->draw_variable(temp_mus, std::get<0>(scale_shape_skew), static_cast<int>(mus.at(i).size()));
    }

    Eigen::Matrix sample_data{sample(nsims)};
    std::vector<double> T_sims;
    for (Eigen::Index i{0}; i < sample_data.cols(); i++)
        T_sims.push_back(T(sample_data.col(i)));
    double T_actual{T(Eigen::VectorXd(_data_frame.data.size(), _data_frame.data.size()))};

    std::string description;
    if (T_name == "mean")
        description = " of the mean";
    else if (T_name == "max")
        description = " of the maximum";
    else if (T_name == "min")
        description = " of the minimum";
    else if (T_name == "median")
        description = " of the median";

    plt::figure_size(width.value(), height.value());
    plt::subplot(1, 1, 1);
    plt::axvline(T_actual);
    plt::plot(T_sims);
    plt::title("Posterior predictive" + description);
    plt::xlabel("T(x)");
    plt::ylabel("Frequency");
    plt::save("../data/arima/plot_ppc.png");
    // plt::show();
}
