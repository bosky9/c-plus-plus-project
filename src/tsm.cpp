#include "tsm.hpp"

Posterior::Posterior(const std::function<double(Eigen::VectorXd)>& posterior) : _posterior{_posterior} {}

Posterior::scalar_t Posterior::operator()(const vector_t& x) const {
    return _posterior(x);
}

TSM::TSM(const std::string& model_type) : _model_type{model_type}, _latent_variables{model_type} {}

BBVIResults* TSM::_bbvi_fit(const std::function<double(Eigen::VectorXd, std::optional<size_t>)>& posterior,
                            const std::string& optimizer, size_t iterations, bool map_start, size_t batch_size,
                            std::optional<size_t> mini_batch, double learning_rate, bool record_elbo,
                            bool quiet_progress, const Eigen::VectorXd& start) {
    Eigen::VectorXd phi{(start.size() > 0) ? start : _latent_variables.get_z_starting_values()}; // If user supplied

    Eigen::VectorXd start_loc;
    if ((_model_type != "GPNARX" || _model_type != "GPR" || _model_type != "GP" || _model_type != "GASRank") &&
        !mini_batch.has_value()) {
        Posterior function{[posterior](Eigen::VectorXd x) { return posterior(std::move(x), std::nullopt); }};
        cppoptlib::solver::Lbfgsb<Posterior> solver(
                Eigen::VectorXd::Constant(phi.size(), std::numeric_limits<double>::min()),
                Eigen::VectorXd::Constant(phi.size(), std::numeric_limits<double>::max()));
        auto [solution, solver_state] = solver.Minimize(function, phi); // PML starting values
        start_loc                     = 0.8 * solution.x + 0.2 * phi;
    } else
        start_loc = phi;
    Eigen::VectorXd start_ses{};

    for (size_t i{0}; i < _latent_variables.get_z_list().size(); i++) {
        std::unique_ptr<Family> approx_dist{_latent_variables.get_z_list()[i].get_q()};
        if (static_cast<std::string>(typeid(approx_dist).name()) == "Normal") {
            _latent_variables.get_z_list()[i].get_q()->vi_change_param(0, start_loc[static_cast<Eigen::Index>(i)]);
            if (start_ses.size() == 0)
                _latent_variables.get_z_list()[i].get_q()->vi_change_param(1, std::exp(-3.0));
            else
                _latent_variables.get_z_list()[i].get_q()->vi_change_param(1, start_ses[static_cast<Eigen::Index>(i)]);
        }
    }

    std::vector<Normal*> q_list;
    for (size_t i{0}; i < _latent_variables.get_z_list().size(); i++)
        q_list.push_back(dynamic_cast<Normal*>(_latent_variables.get_z_list()[i].get_q()));

    std::unique_ptr<BBVI> bbvi_obj;
    if (!mini_batch.has_value())
        bbvi_obj = std::make_unique<BBVI>(posterior, q_list, batch_size, optimizer, iterations, learning_rate,
                                          record_elbo, quiet_progress);
    else
        bbvi_obj = std::make_unique<BBVIM>(posterior, _neg_logposterior, q_list, mini_batch.value(), optimizer,
                                           iterations, learning_rate, mini_batch.value(), record_elbo, quiet_progress);

    BBVIReturnData data{bbvi_obj->run(false)};
    std::transform(data.final_ses.begin(), data.final_ses.end(), data.final_ses.begin(),
                   [](double x) { return exp(x); });
    _latent_variables.set_z_values(data.final_means, "BBVI", data.final_ses);

    for (size_t i{0}; i < _latent_variables.get_z_list().size(); i++)
        _latent_variables.get_z_list()[i].set_q(data.q[i]->clone());

    _latent_variables.set_estimation_method("BBVI");

    ModelOutput output{_categorize_model_output(data.final_means)};

    // LatentVariables latent_variables_store = _latent_variables; // No sense

    return new BBVIResults{_data_name,        output.X_names, _model_name,         _model_type,       _latent_variables,
                           output.Y,          _index,         _multivariate_model, _neg_logposterior, "BBVI",
                           _z_hide,           _max_lag,       data.final_ses,      output.theta,      output.scores,
                           data.elbo_records, output.states,  output.states_var};
}

LaplaceResults* TSM::_laplace_fit(const std::function<double(Eigen::VectorXd)>& obj_type) {
    // Get Mode and Inverse Hessian information
    MLEResults* y{dynamic_cast<MLEResults*>(fit("PML", false))};

    assert(y->get_ihessian().size() > 0 && "No Hessian information - Laplace approximation cannot be performed");
    _latent_variables.set_estimation_method("Laplace");
    ModelOutput output{_categorize_model_output(_latent_variables.get_z_values())};

    return new LaplaceResults(_data_name, output.X_names, _model_name, _model_type, _latent_variables, output.Y, _index,
                              _multivariate_model, obj_type, "Laplace", _z_hide, _max_lag, y->get_ihessian(),
                              output.theta, output.scores, output.states, output.states_var);
}

MCMCResults* TSM::_mcmc_fit(double scale, std::optional<size_t> nsims, bool printer, const std::string& method,
                            std::optional<Eigen::MatrixXd>& cov_matrix, std::optional<bool> map_start,
                            std::optional<bool> quiet_progress) {
    scale = 2.38 / std::sqrt(_z_no);

    // Get Mode and Inverse Hessian information
    Eigen::VectorXd starting_values;
    if (_model_type == "GPNARX" || _model_type == "GRP" || _model_type == "GP" || map_start) {
        MLEResults* y{dynamic_cast<MLEResults*>(fit("PML", false))};
        starting_values = y->get_z().get_z_values();

        Eigen::VectorXd ses = y->get_ihessian().diagonal().cwiseAbs();
        ses.unaryExpr([](double v) { return std::isnan(v) ? 1.0 : v; });
        cov_matrix.emplace(ses.asDiagonal());
    } else
        starting_values = _latent_variables.get_z_starting_values();

    assert(method == "M-H" && "Method not recognized!");
    MetropolisHastings sampler{MetropolisHastings(_neg_logposterior, scale, nsims.value(), starting_values, cov_matrix,
                                                  2, true, quiet_progress.value())};
    Sample sample = sampler.sample();

    _latent_variables.set_z_values(sample.mean_est, "M-H", std::nullopt, sample.chain.row(0));
    if (_latent_variables.get_z_list().size() == 1) {
        auto transform{_latent_variables.get_z_list()[0].get_prior()->get_transform()};
        sample.mean_est[0]     = transform(sample.mean_est[0]);
        sample.median_est[0]   = transform(sample.median_est[0]);
        sample.upper_95_est[0] = transform(sample.upper_95_est[0]);
        sample.lower_95_est[0] = transform(sample.lower_95_est[0]);
    } else
        for (Eigen::Index i{0}; i < sample.chain.size(); i++) {
            auto transform{_latent_variables.get_z_list()[i].get_prior()->get_transform()};
            sample.mean_est[i]     = transform(sample.mean_est[i]);
            sample.median_est[i]   = transform(sample.median_est[i]);
            sample.upper_95_est[i] = transform(sample.upper_95_est[i]);
            sample.lower_95_est[i] = transform(sample.lower_95_est[i]);
        }

    _latent_variables.set_estimation_method("M-H");

    ModelOutput output{_categorize_model_output(sample.mean_est)};

    return new MCMCResults(_data_name, output.X_names, _model_name, _model_type, _latent_variables, output.Y, _index,
                           _multivariate_model, _neg_logposterior, "Metropolis Hastings", _z_hide, _max_lag,
                           sample.chain, sample.mean_est, sample.median_est, sample.upper_95_est, sample.lower_95_est,
                           output.theta, output.scores, output.states, output.states_var);
}

MLEResults* TSM::_ols_fit() {return nullptr;}

MLEResults* TSM::_optimize_fit(const std::string& method, const std::function<double(Eigen::VectorXd)>& obj_type,
                               const std::optional<Eigen::MatrixXd>& cov_matrix, const std::optional<size_t> iterations,
                               const std::optional<size_t> nsims, const std::optional<StochOptim>& optimizer,
                               const std::optional<uint8_t> batch_size, const std::optional<size_t> mininbatch,
                               const std::optional<bool> map_start, const std::optional<double> learning_rate,
                               const std::optional<bool> record_elbo, const std::optional<bool> quiet_progress,
                               const std::optional<bool> preopt_search, const std::optional<Eigen::VectorXd>& start) {
    // Starting values - Check to see if model has preoptimize method, if not, simply use default starting values
    Eigen::VectorXd phi;
    bool preoptimized{false};
    if (start != std::nullopt)
        phi = start.value();
    else {
        // Possible call to  _preoptimize_model() (from subclass)
        phi = _latent_variables.get_z_starting_values();
    }

    // Optimize using L-BFGS-B
    Posterior function{obj_type};
    // the python solver equivalent has no bounds
    cppoptlib::solver::Lbfgsb<Posterior> solver(
            Eigen::VectorXd::Constant(phi.size(), std::numeric_limits<double>::min()),
            Eigen::VectorXd::Constant(phi.size(), std::numeric_limits<double>::max()));
    auto [p, solver_state] = solver.Minimize(function, phi);

    if (preoptimized) {
        auto [p2, solver_state2] = solver.Minimize(function, _latent_variables.get_z_starting_values());
        if (_neg_loglik(p2.x) < _neg_loglik(p.x))
            p = p2;
    }

    ModelOutput output{_categorize_model_output(p.x)};

    // Check that matrix is non-singular, act accordingly
    Eigen::MatrixXd ihessian{hessian(obj_type, p.x).inverse()};
    Eigen::MatrixXd ses{ihessian.diagonal().cwiseAbs().array().pow(0.5)};
    _latent_variables.set_z_values(p.x, method, ses);
    // _latent_variables.set_z_values(p.x, method);

    _latent_variables.set_estimation_method(method);

    return new MLEResults(_data_name, output.X_names, _model_name, _model_type, _latent_variables, p.x, output.Y,
                          _index, _multivariate_model, obj_type, method, _z_hide, _max_lag, ihessian, output.theta,
                          output.scores, output.states, output.states_var);
}

Results* TSM::fit(std::string method, bool printer, std::optional<Eigen::MatrixXd>& cov_matrix,
                  const std::optional<size_t> iterations, const std::optional<size_t> nsims,
                  const std::optional<StochOptim>& optimizer, const std::optional<uint8_t> batch_size,
                  const std::optional<size_t> mininbatch, const std::optional<bool> map_start,
                  const std::optional<double> learning_rate, const std::optional<bool> record_elbo,
                  const std::optional<bool> quiet_progress) {
    if (method.empty())
        method = _default_method;
    assert(std::find(_supported_methods.begin(), _supported_methods.end(), method) != _supported_methods.end() &&
           "Method not supported!");

    if (method == "MLE")
        return _optimize_fit(method, _neg_loglik, cov_matrix, iterations, nsims, optimizer, batch_size, mininbatch,
                             map_start, learning_rate, record_elbo, quiet_progress);

    else if (method == "PML")
        return _optimize_fit(method, _neg_logposterior, cov_matrix, iterations, nsims, optimizer, batch_size,
                             mininbatch, map_start, learning_rate, record_elbo, quiet_progress);

    else if (method == "M-H")
        return _mcmc_fit(1.0, nsims, true, "M-H", cov_matrix, map_start, quiet_progress);
    else if (method == "Laplace")
        return _laplace_fit(_neg_logposterior);

    else if (method == "BBVI") {
        std::function<double(Eigen::VectorXd, std::optional<size_t>)> posterior;
        if (!mininbatch) {
            posterior = change_function_params(_neg_logposterior);
        } else
            posterior = _mb_neg_logposterior;
        return _bbvi_fit(posterior);
    } else if (method == "OLS")
        return _ols_fit();
}

std::vector<size_t> TSM::shift_dates(size_t n) {
    assert(_index.size() > 0);
    assert(_index.size() > _max_lag);
    std::vector<size_t> date_index(_index.begin() + _max_lag, _index.end());
    if (date_index.size() > 1) {
        for (size_t i{0}; i < n; i++)
            date_index.push_back(date_index.back() + (date_index.back() - date_index.at(date_index.size() - 2)));
    } else {
        for (size_t i{0}; i < n; i++)
            date_index.push_back(date_index.back() + 1);
    }
    return std::move(date_index);
}

Eigen::VectorXd TSM::transform_z() const {
    return _latent_variables.get_z_values(true);
}

void TSM::plot_z(const std::optional<std::vector<size_t>>& indices, size_t width, size_t height) {
    _latent_variables.plot_z(indices, width, height);
}

void TSM::adjust_prior(const std::vector<size_t>& index, Family& prior) {
    _latent_variables.adjust_prior(index, prior);
}

Eigen::MatrixXd TSM::draw_latent_variables(size_t nsims) {
    assert(_latent_variables.get_estimation_method());
    assert(_latent_variables.get_estimation_method().value() == "BBVI" ||
           _latent_variables.get_estimation_method().value() == "M-H");
    if (_latent_variables.get_estimation_method().value() == "BBVI") {
        std::vector<Family*> q_vec = _latent_variables.get_z_approx_dist();
        Eigen::MatrixXd output(q_vec.size(), nsims);
        size_t r = 0;
        for (Family* f : q_vec) {
            output.row(r) = f->draw_variable_local(nsims);
            r++;
        }
        return output;
    } else {
        std::vector<LatentVariable> lvs = _latent_variables.get_z_list();
        size_t cols                     = 0;
        if (!lvs.empty())
            cols = lvs.at(0).get_sample().value().size();
        Eigen::MatrixXd chain(lvs.size(), cols);
        for (size_t i{0}; i < lvs.size(); i++) {
            // Check that the samples exists (since they are optional)
            assert(lvs.at(i).get_sample());
            // Check that the samples have the same size
            assert(lvs.at(i).get_sample().value().size() == cols);
            chain.row(i) = lvs.at(i).get_sample().value();
        }
        std::vector<size_t> ind;
        unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
        std::default_random_engine generator(seed);
        std::uniform_int_distribution<size_t> distribution{0, cols};
        for (size_t n{0}; n < nsims; n++)
            ind.push_back(distribution(generator));
        return chain(Eigen::all, ind);
    }
}

void TSM::set_model(std::function<std::tuple<Eigen::VectorXd, Eigen::VectorXd>(Eigen::VectorXd)> model) {
    _model = model;
}

void TSM::set_mb_model(std::function<std::tuple<Eigen::VectorXd, Eigen::VectorXd>(Eigen::VectorXd, size_t)> mb_model) {
    _mb_model = mb_model;
}

void TSM::set_neg_loglik(std::function<double(Eigen::VectorXd)> neg_loglik) {
    _neg_loglik = neg_loglik;
}

void TSM::set_mb_neg_loglik(std::function<double(Eigen::VectorXd, std::optional<size_t>)> mb_neg_loglik) {
    _mb_neg_loglik = mb_neg_loglik;
}