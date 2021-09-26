#include "tsm.hpp"

Posterior::Posterior(const std::function<double(Eigen::VectorXd)>& posterior) : _posterior{_posterior} {}

Posterior::scalar_t Posterior::operator()(const vector_t& x) const {
    return _posterior(x);
}

TSM::TSM(const std::string& model_type) : _model_type{model_type}, _latent_variables{model_type} {};

BBVIResults* TSM::_bbvi_fit(const std::function<double(Eigen::VectorXd, std::optional<size_t>)>& posterior,
                            const std::string& optimizer, size_t iterations, bool map_start, size_t batch_size,
                            std::optional<size_t> mini_batch, double learning_rate, bool record_elbo,
                            bool quiet_progress, const Eigen::VectorXd& start) {
    Eigen::VectorXd phi{(start.size() > 0) ? start : _latent_variables.get_z_starting_values()}; // If user supplied

    Eigen::VectorXd start_loc;
    if ((_model_type != "GPNARX" || _model_type != "GPR" || _model_type != "GP" || _model_type != "GASRank") &&
        !mini_batch.has_value()) {
        // TODO: Controllare che il procedimento sia corretto e i bounds!
        Posterior function{[posterior](Eigen::VectorXd x) { return posterior(x, std::nullopt); }};
        cppoptlib::solver::Lbfgsb<Posterior> solver(Eigen::VectorXd::Zero(phi.size()),
                                                    Eigen::VectorXd::Ones(phi.size()));
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

MLEResults* TSM::_optimize_fit(const std::function<double(Eigen::VectorXd)>& obj_type,
                               const std::optional<Eigen::MatrixXd>& cov_matrix, const std::optional<size_t> iterations,
                               const std::optional<size_t> nsims, const std::optional<StochOptim> optimizer,
                               const std::optional<u_int8_t> batch_size, const std::optional<size_t> mininbatch,
                               const std::optional<bool> map_start, const std::optional<double> learning_rate,
                               const std::optional<bool> record_elbo, const std::optional<bool> quiet_progress,
                               const std::optional<bool> preopt_search, const std::optional<Eigen::VectorXd> start) {

    std::string method;
    if (obj_type == _neg_loglik) // TODO: Serve un modo metodo per confrontare le funzioni
        method = "MLE";
    else
        method = "PML";

    // Starting values - Check to see if model has preoptimize method, if not, simply use default starting values
    Eigen::VectorXd phi;
    bool preoptimized{false};
    if (start != std::nullopt)
        phi = start.value();
    else {
        // TODO: _preoptimize_model() non trovato -> Si può eliminare l'if?
        if (preopt_search) {
            // Eigen::VectorXd phi = _preoptimize_model(_latent_variables.get_z_starting_values(), method);
            // bool preoptimized{true};
            // TODO: _preoptimize_model() non trovato
            phi = _latent_variables.get_z_starting_values();
        } else
            phi = _latent_variables.get_z_starting_values();
    }

    // Optimize using L-BFGS-B
    // TODO: Controllare che il procedimento sia corretto e i bounds!
    Posterior function{obj_type};
    cppoptlib::solver::Lbfgsb<Posterior> solver(Eigen::VectorXd::Zero(phi.size()), Eigen::VectorXd::Ones(phi.size()));
    auto [p, solver_state] = solver.Minimize(function, phi);

    if (preoptimized) {
        auto [p2, solver_state2] = solver.Minimize(function, _latent_variables.get_z_starting_values());
        if (_neg_loglik(p2.x) < _neg_loglik(p.x))
            p = p2;
    }

    ModelOutput output{_categorize_model_output(p.x)};

    // Check that matrix is non-singular, act accordingly
    // TRY
    Eigen::MatrixXd ihessian{(nd.Hessian(obj_type)(p.x)).inverse()}; // TODO: Find a function to compute Hessian
    Eigen::MatrixXd ses{ihessian.diagonal().cwiseAbs().array().pow(0.5)};
    _latent_variables.set_z_values(p.x, method, ses);
    // EXCEPT
    _latent_variables.set_z_values(p.x, method);

    _latent_variables.set_estimation_method(method);

    return new MLEResults(_data_name, output.X_names, _model_name, _model_type, _latent_variables, p.x, output.Y,
                          _index, _multivariate_model, obj_type, method, _z_hide, _max_lag, ihessian, output.theta,
                          output.scores, output.states, output.states_var);
}

Results* TSM::fit(std::string method, const std::optional<Eigen::MatrixXd>& cov_matrix,
                  const std::optional<size_t> iterations, const std::optional<size_t> nsims,
                  const std::optional<StochOptim> optimizer, const std::optional<u_int8_t> batch_size,
                  const std::optional<size_t> mininbatch, const std::optional<bool> map_start,
                  const std::optional<double> learning_rate, const std::optional<bool> record_elbo,
                  const std::optional<bool> quiet_progress) {
    if (method == "")
        method = _default_method;
    assert(std::find(_supported_methods.begin(), _supported_methods.end(), method) != _supported_methods.end() &&
           "Method not supported!");

    if (method == "MLE")
        return _optimize_fit(_neg_loglik, cov_matrix, iterations, nsims, optimizer, batch_size, mininbatch, map_start,
                             learning_rate, record_elbo, quiet_progress);

    else if (method == "PML")
        return _optimize_fit(_neg_logposterior, cov_matrix, iterations, nsims, optimizer, batch_size, mininbatch,
                             map_start, learning_rate, record_elbo, quiet_progress);

    else if (method == "M-H")
        return _mcmc_fit(1.0, nsims, true, "M-H",

                         cov_matrix, map_start, quiet_progress);
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
