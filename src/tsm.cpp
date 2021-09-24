#include "tsm.hpp"

#include <memory>

TSM::TSM(const std::string& model_type) : _model_type{model_type}, _latent_variables{model_type} {};

BBVIResults TSM::_bbvi_fit(const std::function<double(Eigen::VectorXd, std::optional<size_t>)>& posterior,
                           const std::string& optimizer, size_t iterations, bool map_start, size_t batch_size,
                           std::optional<size_t> mini_batch, double learning_rate, bool record_elbo,
                           bool quiet_progress, const Eigen::VectorXd& start) {
    Eigen::VectorXd phi{(start.size() > 0) ? start : _latent_variables.get_z_starting_values()}; // If user supplied

    Eigen::VectorXd start_loc;
    if ((_model_type != "GPNARX" || _model_type != "GPR" || _model_type != "GP" || _model_type != "GASRank") &&
        !mini_batch.has_value()) {
        // Eigen::VectorXd p = optimize.minimize(posterior, phi, method='L-BFGS-B') --> lbfgs search from d lib // PML
        // starting values
        // start_loc = 0.8 * p + 0.2 * phi;
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

    return {_data_name,        output.x_names, _model_name,         _model_type,       _latent_variables,
            output.y,          _index,         _multivariate_model, _neg_logposterior, "BBVI",
            _z_hide,           _max_lag,       data.final_ses,      output.theta,      output.scores,
            data.elbo_records, output.states,  output.states_var};
}
