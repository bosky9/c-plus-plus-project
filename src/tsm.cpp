#include "tsm.hpp"

TSM::TSM(std::string model_type) : _model_type{model_type}, _latent_variables{model_type} {};

BBVIResults TSM::bbvi_fit(const std::function<double(Eigen::VectorXd)>& posterior, const std::string& optimizer,
                          size_t iterations, bool map_start, size_t batch_size, std::optional<size_t> mini_batch,
                          double learning_rate, bool record_elbo, bool quiet_progress, Eigen::VectorXd start) {
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
        if (typeid(approx_dist).name() == "Normal") {
            _latent_variables.get_z_list()[i].get_q()->vi_change_param(0, start_loc[i]);
            if (start_ses.size() == 0)
                _latent_variables.get_z_list()[i].get_q()->vi_change_param(1, std::exp(-3.0));
            else
                _latent_variables.get_z_list()[i].get_q()->vi_change_param(1, start_ses[i]);
        }
    }

    std::vector<Family> q_list;
    for (size_t i{0}; i < _latent_variables.get_z_list().size(); i++)
        q_list.push_back(*_latent_variables.get_z_list()[i].get_q());

    BBVI bbvi_obj;
    if (!mini_batch.has_value())
        bbvi_obj =
                BBVI{posterior, q_list, batch_size, optimizer, iterations, learning_rate, record_elbo, quiet_progress};
    else
        bbvi_obj = BBVIM{posterior,  _neg_logposterior, q_list,     mini_batch,  optimizer,
                         iterations, learning_rate,     mini_batch, record_elbo, quiet_progress};
}